from gws.detectors.GWDetector import GWDetector
import math
import numpy as np
import matplotlib.pyplot as plt


# TODO: remember that we normally plot h^2 * Omega, not just Omega, where h~0.67


class LISA(GWDetector):
    def __init__(self):
        super().__init__()

        # Convert years to seconds. Gregorian calendar has 31556952 seconds in a year.
        self.detectionTime = 3. * 31556952
        self.numIndependentChannels = 1

    def constructSensitivityCurve(self, frequencies):
        # Curve taken from "Reconstructing the spectral shape of a stochastic gravitational wave background with LISA".
        L = 2.5*10**9
        c = 2.998*10**8
        p = 15
        a = 3
        H0 = 2.2*10**-18

        P_oms = lambda f, P: (10**-12)**2 * P*P * (1 + (2*10**-3 / f)**4) * (2*np.pi*f/c)**2
        P_acc = lambda f, A: (10**-15)**2 * A*A * (1 + (4*10**-4 / f)**2) * (2*np.pi*f/c)**2 * (1 + (f/(8*10**-3))**4)\
                                / (2*np.pi*f)**4
        P_n = lambda f: 16*np.sin(2*np.pi*f*L/c)**2 * (P_oms(f,p) + (3 + np.cos(4*np.pi*f*L/c)) * P_acc(f,a))
        R_n = lambda f: 16*np.sin(2*np.pi*f*L/c)**2 * 0.3 * (2*np.pi*f*L/c)**2 / (1 + 0.6*(2*np.pi*f*L/c)**2)
        S_n = lambda f: P_n(f) / R_n(f)
        Omega_s = lambda f: 4*np.pi**2 / (3*H0**2) * f**3 * S_n(f)

        #P = P_n(frequencies)
        #Po = P_oms(frequencies, p)
        #Pa = P_acc(frequencies, a)
        #R = R_n(frequencies)
        #S = S_n(frequencies)
        #O = Omega_s(frequencies)

        h = 0.67

        self.sensitivityCurve = np.empty(shape=(2, len(frequencies)))
        self.sensitivityCurve[0] = frequencies
        # Compute h^2 Omega(f), since the GW signals also have the prefactor of h^2.
        self.sensitivityCurve[1] = h*h * Omega_s(frequencies)


if __name__ == "__main__":
    detector = LISA()
    detector.constructSensitivityCurve(np.logspace(-4, -1, 400))

    plt.loglog(detector.sensitivityCurve[0], detector.sensitivityCurve[1])

    # Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower.
    detector.loadSensitivityCurveFromFile('LISA sensitivity spectrum.txt')
    plt.loglog(detector.sensitivityCurve[0], detector.sensitivityCurve[1])

    # Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower, with the 2019 sensitivity equation.
    detector.loadSensitivityCurveFromFile('LISA sensitivity spectrum 2019.txt')
    plt.loglog(detector.sensitivityCurve[0], detector.sensitivityCurve[1])

    # Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower, with the 2019 sensitivity equation.
    detector.loadSensitivityCurveFromFile('LISA sensitivity spectrum 2019 PLS (lisa_plot) SNR=1.txt')
    plt.loglog(detector.sensitivityCurve[0], detector.sensitivityCurve[1])

    # Compare the equation to the PLS from Thrane's Plcurves, LISA_noisepower, with the 2019 sensitivity equation.
    detector.loadSensitivityCurveFromFile('LISA sensitivity spectrum 2019 PLS (lisa_plot) SNR=10.txt')
    plt.loglog(detector.sensitivityCurve[0], detector.sensitivityCurve[1])

    plt.legend(['Eq', 'PLS', 'PLS of Eq',  '2019 PLS (SNR=1)', '2019 PLS (SNR=10)'])
    plt.show()
