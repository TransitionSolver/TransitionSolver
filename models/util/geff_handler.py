from __future__ import annotations
from typing import Callable, List, Tuple
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


def getGeffCurveFromFile(filename: str) -> Callable[[float], float]:
    x: List[float] = []
    y: List[float] = []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            x.append(float(line[0]))
            y.append(float(line[1]))
            if y[-1] < 0:
                y[-1] = 0.

    # Make sure we have an increasing sequence of x values, otherwise the CubicSpline will complain.
    if x[0] > x[-1]:
        x.reverse()
        y.reverse()

    interpolator = CubicSpline(x, y)

    # Find the boundaries of the data so we can clamp the output to match the domain.
    xMin = x[0]
    yMin = y[0]
    xMax = x[-1]
    yMax = y[-1]

    def clampedInterpolator(z: float):
        z = np.array(z)
        mask_underflow = z <= xMin
        mask_overflow = z >= xMax
        mask_inrange = np.logical_and(z > xMin, z < xMax)
        result = np.zeros(shape=z.shape)
        result[mask_underflow] = yMin
        result[mask_overflow] = yMax
        result[mask_inrange] = interpolator(z[mask_inrange])
        mask_negative = np.logical_and(mask_inrange, result < 0.)
        result[mask_negative] = 0.
        return result
        #if z <= xMin:
        #    return yMin
        #if z >= xMax:
        #    return yMax
        #return max(0., interpolator(z))

    return clampedInterpolator


def getGeffCurve_boson() -> Callable[[float], float]:
    return getGeffCurveFromFile('models/util/boson_geff.csv')


def getGeffCurve_fermion() -> Callable[[float], float]:
    return getGeffCurveFromFile('models/util/fermion_geff.csv')


def getGeffCurve_total() -> Callable[[float], float]:
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
