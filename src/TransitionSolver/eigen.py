"""
Convert Python list-like objects to Eigen
=========================================
"""

from pathlib import Path
import numpy as np

import cppyy

EIGEN_VECTOR_INCLUDE = Path("eigen3") / "Eigen" / "Core"
cppyy.include(str(EIGEN_VECTOR_INCLUDE))


def vector(list_):
    """
    @param list_ List-like, iterable Python object
    @returns Eigen::VectorXf object
    """
    vec = cppyy.gbl.Eigen.Vector[float, len(list_)]()
    for i, e in enumerate(list_):
        vec[i] = e
    return vec


def to_numpy(arr):
    """
    @param arr Eigen::MatrixXf object
    @returns np.array of same object
    """
    np_arr = np.empty((arr.rows(), arr.cols()), dtype=float)
    for i in range(arr.rows()):
        for j in range(arr.cols()):
            np_arr[i, j] = arr[i, j]
    return np.squeeze(np_arr)
