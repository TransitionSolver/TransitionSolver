"""
LISA detector
=============
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos, sin

from .detector import Detector, FromDiskDetector, SECONDS_PER_YEAR


LITTLE_H = 0.67
THIS = Path(os.path.dirname(os.path.abspath(__file__)))


class LISA(Detector):
    """
    Based on data and equations in 10.1088/1475-7516/2019/11/017
    
    Reconstructing the spectral shape of a stochastic gravitational wave background with LISA
    e-Print: 1906.09244 [astro-ph.CO]
    DOI: 10.1088/1475-7516/2019/11/017
    Published in: JCAP 11 (2019), 017
    """
    def __call__(self, f):

        L = 2.5e9
        c = 2.998e8
        P = 15
        A = 3
        H0 = 2.2e-18

        P_oms = (1e-12)**2 * P**2 * (1 + (2e-3 / f)**4) * (2*pi*f/c)**2
        P_acc = (1e-15)**2 * A**2 * (1 + (4e-4 / f)**2) * \
            (2*pi*f/c)**2 * (1 + (f/(8e-3))**4) / (2*pi*f)**4
        P_n = 16*sin(2*pi*f*L/c)**2 * (P_oms + (3 + cos(4*pi*f*L/c)) * P_acc)
        R_n = 16*sin(2*pi*f*L/c)**2 * 0.3 * (2*pi*f*L/c)**2 / \
            (1 + 0.6*(2*pi*f*L/c)**2)
        S_n = P_n / R_n
        Omega_s = 4*pi**2 / (3*H0**2) * f**3 * S_n

        # Compute h^2 Omega(f), since the GW signals also have the prefactor of h^2
        return LITTLE_H**2 * Omega_s


lisa = LISA(detection_time=3. * SECONDS_PER_YEAR, channels=1)
# Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower
lisa_thrane = FromDiskDetector(THIS / 'LISA_sensitivity_spectrum.txt')
lisa_thrane_1_yr = FromDiskDetector(THIS / 'LISA_sensitivity_spectrum_1_yr.txt')
# Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower, with the 2019 sensitivity equation
lisa_thrane_2019 = FromDiskDetector(THIS / 'LISA_sensitivity_spectrum_2019.txt')
lisa_thrane_2019_snr_1 = FromDiskDetector(THIS / 'LISA_sensitivity_spectrum_2019_SNR_1.txt')
lisa_thrane_2019_snr_10 = FromDiskDetector(THIS / 'LISA_sensitivity_spectrum_2019_SNR_10.txt')
