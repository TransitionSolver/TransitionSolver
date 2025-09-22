"""
Represent a gravitational wave detector
========================================
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import quad


class Detector(ABC):

    detector_time = 1.
    channels = 1

    @abstractmethod
    def __call__(self, f):
        """
        :param f: Frequencies in Hz
        :return: Sensitivity at those frequencies ($\Omega h^2$)
        """
        
    def SNR(self, signal, a=1e-11, b=1e3):
        """
        @returns Signal to noise ratio for detector and given signal
        """
        def integrand(log_f):
            f = np.exp(log_f)
            return f * (signal(f) / self(f))**2

        integral = quad(integrand, np.log(a), np.log(b))[0]
        return (self.detection_time * self.channels * integral)**0.5


class FromDiskDetector(Detector):
    """
    Detector data from disk
    """

    def __init__(self, file_name, detector_time=1., channels=1):
        self.f, self.omega_h2 = np.loadtxt(file_name, unpack=True)
        self.detector_time = detector_time
        self.channels = channels

    def __call__(self, f):
        return np.interp(f, self.f, self.omega_h2, left=0, right=0)
