"""
Check dictionaries equal within tolerance
=========================================
"""

import json
import numpy as np


def key2int(d):
    """
    Interpret dictionary keys as integers - avoid ambiguity in JSON
    """
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


def isclose(result, file_name, rtol=1e-3, atol=0., ignore=None):
    """
    @param result Dictionary of results from program
    @param file_name Name of JSON file containing expected results
    @param ignore Any top-level dictionary keys to ignore
    """
    with open(file_name, "r") as f:
        expected = json.load(f, object_hook=key2int)

    if ignore is None:
        ignore = []

    def assert_approx_equal(a, b):
        try:
            assert np.allclose(a, b, rtol=rtol, atol=atol)
        except TypeError:
            assert a == b

    for i in ignore:
        del expected[i]

    return _isclose(result, expected, assert_approx_equal)


def is_list_dict(l):
    """
    @returns Whether list is a list of dictionaries
    """
    return isinstance(l, list) and all([isinstance(e, dict) for e in l])


def _isclose(result, expected, assert_equal):
    """
    Walk through data and perform check
    """
    if not isinstance(expected, dict):
        return assert_equal(result, expected)

    for k in expected:
        if isinstance(expected[k], dict):
            for l in expected[k]:
                _isclose(result[k][l], expected[k][l], assert_equal)
        elif is_list_dict(expected[k]):
            for a, b in zip(result[k], expected[k]):
                _isclose(a, b, assert_equal)
        else:
            _isclose(result[k], expected[k], assert_equal)

    return True
