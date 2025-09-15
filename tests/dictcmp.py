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

    def approx_equal(a, b):
        try:
            return np.all(np.isclose(a, b, rtol=rtol, atol=atol))
        except TypeError:
            return a == b

    for k in set(expected) - set(ignore):
        if isinstance(expected[k], dict):
            for l in expected[k]:
                assert approx_equal(result[k][l], expected[k][l])
        else:
            assert approx_equal(result[k], expected[k])

    return True
