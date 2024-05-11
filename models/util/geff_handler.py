from __future__ import annotations
from typing import Callable, List, Tuple, Union
import csv
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


# Currently using digitised curves from top panels of Fig 3 from https://arxiv.org/pdf/1609.04979.pdf.


# Only used for debugging, say to plot the data.
def getRawGeffCurveFromFile(filename: str) -> Tuple[List[float], List[float]]:
    x: List[float] = []
    y: List[float] = []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            x.append(float(line[0]))
            y.append(float(line[1]))
            if y[-1] < 0:
                y[-1] = 0.

    return x, y


# Constructs and returns a function that takes as input z=T/m (temperature over particle mass) and returns the effective
# degrees of freedom for the corresponding particle. The function is derived from digitised data stored in the input
# file. Any z values outside the data's domain are clamped to the data boundaries. That is, if z < zMin, treat it as
# z = zMin, and similarly for z > zMax. Any negative degrees of freedom in the data is set to zero.
def getGeffCurveFromFile(filename: str) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    zData: List[float] = []
    geff: List[float] = []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            zData.append(float(line[0]))
            geff.append(float(line[1]))
            # Don't allow negative degrees of freedom. Instead, set to zero.
            if geff[-1] < 0:
                geff[-1] = 0.

    # Make sure we have an increasing sequence of zData values, otherwise the CubicSpline will complain.
    if zData[0] > zData[-1]:
        zData.reverse()
        geff.reverse()

    # Create a cubic spline from the data. This is effectively the function that will be returned.
    interpolator = CubicSpline(zData, geff)

    # Find the boundaries of the data so we can clamp the output to match the domain.
    zMin = zData[0]
    geffMin = geff[0]
    zMax = zData[-1]
    geffMax = geff[-1]

    # Return a clamped version of the cubic spline such that any input values outside the domain are instead treated as
    # the closest boundary of the domain.
    # TODO: why can't we just set extrapolate=False when creating the CubicSpline?
    def clampedInterpolator(z: Union[float, np.ndarray]) -> np.ndarray:
        z = np.array(z)
        mask_underflow = z <= zMin
        mask_overflow = z >= zMax
        mask_inrange = np.logical_and(z > zMin, z < zMax)
        result = np.zeros(shape=z.shape)
        result[mask_underflow] = geffMin
        result[mask_overflow] = geffMax
        result[mask_inrange] = interpolator(z[mask_inrange])
        mask_negative = np.logical_and(mask_inrange, result < 0.)
        result[mask_negative] = 0.
        return result
        #if z <= zMin:
        #    return geffMin
        #if z >= zMax:
        #    return geffMax
        #return max(0., interpolator(z))

    return clampedInterpolator


def getGeffCurve_boson() -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    return getGeffCurveFromFile('models/util/boson_geff.csv')


def getGeffCurve_fermion() -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    return getGeffCurveFromFile('models/util/fermion_geff.csv')


def getGeffCurve_total() -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    return getGeffCurveFromFile('models/util/SM_geff.csv')


if __name__ == "__main__":
    x_, y_ = getRawGeffCurveFromFile('models/util/SM_geff.csv')
    #myFunc = getGeffCurve_fermion()
    myFunc = getGeffCurve_total()
    #xSamples = np.linspace(x_[0], x_[-1], 1000)
    xSamples = np.logspace(np.log10(x_[0]), np.log10(x_[-1]), 1000)
    ySamples = [myFunc(xSample) for xSample in xSamples]
    plt.scatter(x_, y_)
    plt.plot(xSamples, ySamples)
    plt.xscale('log')
    plt.margins(0)
    plt.show()
