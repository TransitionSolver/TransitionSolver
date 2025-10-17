"""
Check dictionaries equal within tolerance
=========================================
"""

import json
import numpy as np


np.set_printoptions(legacy='1.25')


def key2int(d):
    """
    Interpret dictionary keys as integers - avoid ambiguity in JSON
    """
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


class NumpyEncoder(json.JSONEncoder):
    """
    Serialisable numpy arrays
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def approx_equal(a, b, rtol, atol):
    """
    @returns Whether equal within tolerances
    """
    try:
        return np.allclose(a, b, rtol=rtol, atol=atol)
    except TypeError:
        return a == b


def isclose(result, file_name, rtol=1e-3, atol=0., ignore=None, generate_baseline=False):
    """
    @param result Dictionary of results from program
    @param file_name Name of JSON file containing expected results
    @param ignore Any dictionary keys to ignore
    """
    if generate_baseline:
        with open(file_name, "w") as f:
            json.dump(result, f, cls=NumpyEncoder)

    with open(file_name, "r") as f:
        expected = json.load(f, object_hook=key2int)

    if ignore is None:
        ignore = []

    return not assert_isclose(result, expected, rtol, atol, ignore)


def is_list_dict(l):
    """
    @returns Whether list is a list of dictionaries
    """
    return isinstance(l, list) and all([isinstance(e, dict) for e in l])


def assert_isclose(result, expected, rtol, atol, ignore):
    """
    Walk through data and perform check
    """
    for k in set(expected) - set(ignore):
        if is_list_dict(expected[k]):
            for a, b in zip(result[k], expected[k]):
                assert_isclose(a, b, rtol, atol, ignore)
        elif isinstance(expected[k], dict):
            assert_isclose(result[k], expected[k], rtol, atol, ignore)
        elif not approx_equal(result[k], expected[k], rtol, atol):
            raise AssertionError(f"Disagreement for {k}. Expected {expected[k]}; found {result[k]}")
