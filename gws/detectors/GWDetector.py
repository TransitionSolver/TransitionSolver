import numpy as np
from typing import List


class GWDetector:
    """
    The base class for gravitational wave detectors. Subclasses should override constructSensitivityCurve unless the
    sensitivity curve will be loaded from a file. The main use of this class is to construct and store the sensitivity
    curve for a detector, and provide convenience functions for interpolation between sampled frequencies.
    """

    sensitivityCurve: List[List[float]]
    detectionTime: float
    numIndependentChannels: float

    def __init__(self):
        # This is expected to contain at least two numpy arrays once the sensitivity curve has been loaded or
        # constructed. The first array should be the list of sampled frequencies. The second array should be the
        # detector noise/sensitivity in the form of h^2 Omega, where Omega is the fractional energy density.
        self.sensitivityCurve = []

    def loadSensitivityCurveFromFile(self, fileName):
        """
        Loads the sensitivity curve, including sampled frequencies, from the input file. The file should have at least
        two columns of numbers, where the first column contains the sampled frequency values, and the second column
        contains the noise/sensitivity values at those frequencies.

        :param fileName: A string representing the name of the file to load from.
        :return: N/A
        """

        # Unpack transposes the matrix so that the 0th dimension corresponds to the columns in the file.
        self.sensitivityCurve = np.loadtxt(fileName, unpack=True)

    def constructSensitivityCurve(self, frequencies):
        """
        Constructs the detector's noise/sensitivity curve. The frequencies are stored in the first element of
        sensitivityCurve, while the noise values at those frequencies are stored in the second element. To be overridden
        in sub-classes.

        :param frequencies: A numpy array of frequencies (in Hz) to be sampled.
        :return: N/A
        """

        return

    def getSensitivity(self, frequencies):
        """
        Returns the noise values at the input frequencies to sample. Interpolation is used between the frequencies
        sampled to construct the noise curve (which is assumed to have been loaded or constructed).

        :param frequencies: A numpy array of frequencies to sample the noise curve at.
        :return: A numpy array of noise values interpolated from the constructed noise curve to match the desired
            frequencies to sample.
        """

        return np.interp(frequencies, self.sensitivityCurve[0], self.sensitivityCurve[1], left=0, right=0)

    def interpolateSensitivityCurve(self, newFrequencies):
        """
        Identical to getSensitivity, except the detector's sensitivity curve is overridden with the new interpolated
        curve which is not returned.

        :param frequencies: A numpy array of frequencies to sample the noise curve at.
        :return: N/A
        """

        #newSensitivityCurve = np.interp(newFrequencies, self.sensitivityCurve[0], self.sensitivityCurve[1])
        newSensitivityCurve = self.getSensitivity(newFrequencies)

        self.sensitivityCurve = np.empty(shape=(2, len(newFrequencies)))
        self.sensitivityCurve[0] = newFrequencies
        self.sensitivityCurve[1] = newSensitivityCurve
