"""
Represent a gravitational wave detector
========================================
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


SECONDS_PER_YEAR = 31556952


class Detector(ABC):

    def __init__(self, detection_time=1. * SECONDS_PER_YEAR, channels=1, label=None):
        self.detection_time = detection_time
        self.channels = channels
        self.label = label

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

    def __init__(self, file_name, **kwargs):
        self.f, self.omega_h2 = np.loadtxt(file_name, unpack=True)
        self._interp = interp1d(self.f, self.omega_h2, fill_value="extrapolate")
        super().__init__(self, **kwargs)

    def __call__(self, f):
        return self._interp(f)
