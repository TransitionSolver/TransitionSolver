"""
Check dictionaries equal within tolerance
=========================================
"""

import json_numpy as json
from deepdiff import DeepDiff
import numpy as np


def key2int(d):
    """
    @returns Dictionary with keys as integers - avoid ambiguity in JSON
    """
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


def assert_deep_equal(result, file_name, significant_digits=3, generate_baseline=False, **kwargs):
    """
    @param result Dictionary of results from program
    @param file_name Name of JSON file containing expected results
    """
    if generate_baseline:
        with open(file_name, "w") as f:
            json.dump(result, f)

    with open(file_name, "r") as f:
        expected = json.load(f, object_hook=key2int)

    diff = DeepDiff(expected, result, number_format_notation="e", significant_digits=significant_digits, ignore_type_subclasses=True, ignore_numeric_type_changes=True,
                    ignore_order=True, ignore_type_in_groups=[(list, np.ndarray), (float, np.float64), (bool, np.bool)], **kwargs)

    assert not diff, diff.pretty()
