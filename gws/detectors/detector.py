"""
Represent a gravitational wave detector
========================================
"""

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    @abstractmethod
    def __call__(self, f):
        """
        :param f: Frequencies in Hz
        :return: Sensitivity at those frequencies ($\Omega h^2$)
        """


class FromDiskDetector(Detector):
    """
    Detector data from disk
    """

    def __init__(self, file_name):
        self.f, self.omega_h2 = np.loadtxt(file_name, unpack=True)

    def __call__(self, f):
        return np.interp(f, self.f, self.omega_h2, left=0, right=0)
