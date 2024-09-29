from __future__ import annotations

from typing import List


class IntegrationHelper:
    x: List[float]
    data: List[float]
    bInitialised: bool
    numPreparationPoints: int = 0
    bDebug: bool  # TODO: currently not set to true anywhere.

    def __init__(self, *args, **kwargs):
        self.bDebug = False
        self.bInitialised = False
        self.init(*args, **kwargs)
        #self.initialiseIntegration()

    def init(self, *args, **kwargs):
        return

    def initialiseIntegration(self):
        self.bInitialised = True
        #self.x = [self.x[0]]
        #self.data = [self.data[0]]

    def integrate(self, newx: float) -> float:
        if len(self.x) == self.numPreparationPoints:
            self.initialiseIntegration()

        # This can only happen in custom integration helpers. The provided integration helpers add to x upon creation.
        if len(self.x) == 0:
            self.x.append(newx)
            self.data.append(0.)
            return 0.

        if self.bInitialised:
            return self.integrateProper(newx)
        else:
            return self.integrateNaive(newx)

    # Should append newx to self.x, and the result to self.data.
    def integrateProper(self, newx: float) -> float:
        self.x.append(newx)
        self.data.append(0.)
        return self.data[-1]

    # Should append newx to self.x, and the result to self.data.
    def integrateNaive(self, newx: float) -> float:
        self.x.append(newx)
        self.data.append(0.)
        return self.data[-1]

    #def undo(self):
    #    self.x.pop()
    #    self.data.pop()

    def setRestorePoint(self):
        raise NotImplementedError('IntegrationHelper.setRestorePoint must be overridden if it is to be used.')

    def restore(self):
        raise NotImplementedError('IntegrationHelper.restore must be overridden if it is to be used.')


# Designed specifically to aid in evaluating integrals of the form
#   I(x) = int_x^a dx' f(x') [int_x^x' dx'' g(x'')]^3 ,
# where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm, where n is the number
# of evaluations, assuming n is also the step size in the integration. However, this can be reduced to O(n) using a
# recurrence relation between I(x_i) and I(x_{i+1}).
# This class assumes a new, smaller* x is passed to the integration for each interval.
# *smaller after it has been transformed using sampleTransformationFunction.
class CubedNestedIntegrationHelper(IntegrationHelper):
    def init(self, firstX, outerFunction, innerFunction, sampleTransformationFunction):
        #self.x = initialPoints
        self.x = [firstX]
        self.data = [0.]

        self.f = outerFunction
        self.g = innerFunction
        self.tr = sampleTransformationFunction
        #self.i = 0
        self.numPreparationPoints = 3

        self.restorePoint = {}
        self.L2 = 0.
        self.L3 = 0.
        self.L6 = 0.
        self.S1 = 0.
        self.S2 = 0.
        self.S3 = 0.
        self.S4 = 0.
        self.S5 = 0.
        self.S6 = 0.

    # TODO: probably easier to just create a copy of the object at this rate. The current method is more efficient but
    #  more cumbersome.
    def setRestorePoint(self):
        self.restorePoint = \
        {
            'L2': self.L2,
            'L3': self.L3,
            'L6': self.L6,
            'S1': self.S1,
            'S2': self.S2,
            'S3': self.S3,
            'S4': self.S4,
            'S5': self.S5,
            'S6': self.S6,
            'length': len(self.x)
        }

    def restore(self):
        self.L2 = self.restorePoint['L2']
        self.L3 = self.restorePoint['L3']
        self.L6 = self.restorePoint['L6']
        self.S1 = self.restorePoint['S1']
        self.S2 = self.restorePoint['S2']
        self.S3 = self.restorePoint['S3']
        self.S4 = self.restorePoint['S4']
        self.S5 = self.restorePoint['S5']
        self.S6 = self.restorePoint['S6']
        self.x = self.x[:self.restorePoint['length']]
        self.data = self.data[:self.restorePoint['length']]

    def initialiseIntegration(self):
        super().initialiseIntegration()

        self.L2 = 0.
        self.L3 = 0.
        self.L6 = 0.

        # Aliases for concision.
        x = self.x
        f = self.f
        g = self.g
        tr = self.tr

        #xn1 = tr(x[0])
        #xn2 = tr(x[1])
        #xn3 = tr(x[2])

        dxn2 = tr(x[0]) - tr(x[1])
        dxn3 = tr(x[1]) - tr(x[2])
        #dxn3 = tr(x[2]) - tr(x[3])

        fn1 = f(x[0])
        fn2 = f(x[1])
        fsn2 = fn2 + fn1
        gsn3 = g(x[2]) + g(x[1])
        gsn2 = g(x[1]) + g(x[0])

        #self.data.append(f(x[0]) * gsn2**3 * dx1**4 / (16.0 * x[1]**3))
        #self.data.append((x[1]/x[2])**3 * self.data[1] + gsn3*dx2*(f1*gsn3**2*dx2**2*(dx1 + dx2)
        #    + 3*f0*gsn2*dx1**2*(gsn3*dx2 + gsn2*dx1)) / (16.0 * x[2]**3))
        self.data.append(f(x[0]) * gsn2**3 * dxn2**4 / (16.0)) # * xn2**3))
        # This is using the final form of Delta I_i
        #otherMethod = (xn2/xn3)**3 * self.data[1] + gsn3*dxn3*(fn2*gsn3**2*dxn3**2*(dxn2 + dxn3)
        #    + 3*fn1*gsn2*dxn2**2*(gsn3*dxn3 + gsn2*dxn2)) / (16.0 * xn3**3)
        self.data.append((fn2*gsn3**3*dxn3**3*(dxn3 + dxn2) + fn1*(gsn3*dxn3 + gsn2*dxn2)**3*dxn2) / (16.0)) # * xn3**3))
        #print(abs((otherMethod - self.data[-1]) / self.data[-1]))

        self.S1 = (gsn3*dxn3)**3 * fsn2*dxn2
        self.S2 = 0.
        self.S3 = 0.
        self.S4 = 3*fn1*gsn3**2*dxn3**2*gsn2*dxn2**2
        self.S5 = 3*fn1*gsn3*dxn3*gsn2**2*dxn2**3
        self.S6 = 0.

        #self.i = 3

    def integrateProper(self, newx: float) -> float:
        if self.bDebug and not self.bInitialised:
            print('CubedNestedIntegrationHelper.integrate called when bInitialised is False.')

        self.x.append(newx)

        # Aliases for concision.
        x = self.x
        f = self.f
        g = self.g
        tr = self.tr
        #i = self.i
        i = len(self.x)-1

        xi = tr(x[i])
        xi1 = tr(x[i-1])
        xi2 = tr(x[i-2])

        dxi = xi1 - xi
        dxi1 = xi2 - xi1
        dxi2 = tr(x[i-3]) - xi2

        fi1 = f(x[i-1])
        fi2 = f(x[i-2])
        fi3 = f(x[i-3])
        fsi1 = fi1 + fi2
        fsi2 = fi2 + fi3
        gsi = g(x[i]) + g(x[i-1])
        gsi1 = g(x[i-1]) + g(x[i-2])
        gsi2 = g(x[i-2]) + g(x[i-3])

        """
        # We want to use L2(i+1) in L3(i), so do L3 before updating L2.
        self.L3 += gsi2*self.L2#*dxi2*self.L2
        self.L2 += fsi2#*dxi2
        self.L6 += fi3*gsi2#*dxi2**2

        #factor = (gsi*dxi)/(gsi1*dxi1)
        factor = gsi/gsi1

        self.S1 = factor**3 * self.S1 + fsi1*gsi**3#*dxi**3*dxi1
        self.S2 = factor**2 * self.S2 + 3*gsi**2*gsi1 * self.L2 #*dxi**2*dxi1 * self.L2
        self.S3 = factor    * self.S3 + 3*gsi*gsi1 * (2*self.L3 + gsi1*self.L2)#*dxi*dxi1*(2*self.L3 + gsi1*dxi1*self.L2)
        self.S4 = factor**2 * self.S4 + 3*fi2*gsi**2*gsi1#*dxi**2*dxi1**2
        self.S5 = factor    * self.S5 + 3*fi2*gsi*gsi1**2#*dxi*dxi1**3
        self.S6 = factor    * self.S6 + 6*gsi*gsi1 * self.L6 #*dxi*dxi1 * self.L6

        Si = (self.S1 + self.S2 + self.S3 + self.S4 + self.S5 + self.S6) * dxi**4

        Ti = fi1*gsi**3*dxi**4

        #self.data.append((xi1/xi)**3 * self.data[-1] + (Si + Ti) / (16.0*xi**3))
        self.data.append(self.data[-1] + (Si + Ti) / 16.0)
        """

        # We want to use L2(i+1) in L3(i), so do L3 before updating L2.
        self.L3 += gsi2*dxi2*self.L2
        self.L2 += fsi2*dxi2
        self.L6 += fi3*gsi2*dxi2**2

        factor = (gsi*dxi)/(gsi1*dxi1)

        self.S1 = factor**3 * self.S1 + fsi1*gsi**3*dxi**3*dxi1
        self.S2 = factor**2 * self.S2 + 3*gsi**2*gsi1*dxi**2*dxi1 * self.L2
        self.S3 = factor    * self.S3 + 3*gsi*gsi1*dxi*dxi1*(2*self.L3 + gsi1*dxi1*self.L2)
        self.S4 = factor**2 * self.S4 + 3*fi2*gsi**2*gsi1*dxi**2*dxi1**2
        self.S5 = factor    * self.S5 + 3*fi2*gsi*gsi1**2*dxi*dxi1**3
        self.S6 = factor    * self.S6 + 6*gsi*gsi1*dxi*dxi1 * self.L6

        Si = self.S1 + self.S2 + self.S3 + self.S4 + self.S5 + self.S6

        Ti = fi1*gsi**3*dxi**4

        #self.data.append((xi1/xi)**3 * self.data[-1] + (Si + Ti) / (16.0*xi**3))
        self.data.append(self.data[-1] + (Si + Ti) / 16.0)

        #self.i += 1

        return self.data[-1]

    # Uses the naive integration method. Returns the result and doesn't store the result in self.data. Used only for
    # temporary integration when the integration handler doesn't have enough data to be initialised.
    def integrateNaive(self, newx: float) -> float:
        result: float = 0

        self.x.append(newx)

        # Reverse the list of decreasing inputs to make integration more intuitive.
        x = self.x[::-1]
        x_tr = [self.tr(xx) for xx in x]
        f = self.f
        g = self.g

        for i in range(len(x)-1):
            dxi = abs(x_tr[i+1] - x_tr[i])
            innerIntegral = 0
            for j in range(i):
                dxj = abs(x_tr[j+1] - x_tr[j])
                innerIntegral += dxj*(g(x[j]) + g(x[j+1]))
            innerIntegralExtra = dxi*(g(x[i]) + g(x[i+1]))
            result += dxi*(f(x[i])*innerIntegral**3 + f(x[i+1])*(innerIntegral + innerIntegralExtra)**3)

        self.data.append(result/16)

        return self.data[-1]


# Designed specifically to aid in evaluating integrals of the form
#   I(x) = [int_x^a dx' f(x') int_x^x' dx'' g(x'')] / [int_x^a dx' h(x')] ,
# where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm,
# where n is the number of evaluations, assuming n is also the step size in the integration. However, this can be
# reduced to O(n) using a recurrence relation between I(x_i) and I(x_{i+1}).
# This class assumes a new, smaller x is passed to the integration for each interval (where x is smaller after it has
# been transformed using sampleTransformationFunction).
# This integration scheme requires two initial x points before the main recurrence relations can be used.
class LinearNestedNormalisedIntegrationHelper(IntegrationHelper):
    def init(self, firstX, outerFunction, innerFunction, normalisationFunction, sampleTransformationFunction):
        #self.x = initialPoints
        self.x = [firstX]
        self.data = [0.]

        self.f = outerFunction
        self.g = innerFunction
        self.h = normalisationFunction
        self.tr = sampleTransformationFunction
        #self.i = 0
        self.numPreparationPoints = 2

        self.restorePoint = {}

        self.L = 0.
        self.A = 0.
        self.B = 0.

    def setRestorePoint(self):
        self.restorePoint = {
            'L': self.L,
            'A': self.A,
            'B': self.B,
            'length': len(self.x)
        }

    def restore(self):
        self.L = self.restorePoint['L']
        self.A = self.restorePoint['A']
        self.B = self.restorePoint['B']
        self.x = self.x[:self.restorePoint['length']]
        self.data = self.data[:self.restorePoint['length']]

    def initialiseIntegration(self):
        super().initialiseIntegration()

        self.L = 0.
        self.A = 0.
        self.B = 0.

        # Aliases for concision.
        x = self.x
        f = self.f
        g = self.g
        h = self.h
        tr = self.tr

        dxn2 = abs(tr(x[0]) - tr(x[1]))

        #fn1 = f(x[0])
        #fn2 = f(x[1])
        gsn2 = g(x[1]) + g(x[0])

        self.A = 0.25*f(x[0])*gsn2*dxn2**2
        # self.L is still 0 for point n-2.
        self.B = 0.5*(h(x[1]) + h(x[0]))*dxn2

        self.data.append(self.A/self.B)

        #self.i = 2

    def integrateProper(self, newx: float) -> float:
        if self.bDebug and not self.bInitialised:
            print('CubedNestedIntegrationHelper.integrate called when bInitialised is False.')

        self.x.append(newx)

        x = self.x
        f = self.f
        g = self.g
        h = self.h
        tr = self.tr
        #i = self.i
        i = len(self.x)-1

        xi = tr(x[i])
        xi1 = tr(x[i-1])

        dxi = abs(tr(x[i-1]) - tr(x[i]))
        dxi1 = abs(tr(x[i-2]) - tr(x[i-1]))

        fi1 = f(x[i-1])
        fsi1 = fi1 + f(x[i-2])
        gsi = g(x[i]) + g(x[i-1])
        hsi = h(x[i]) + h(x[i-1])

        self.L += fsi1*dxi1

        #self.A = (xi1/xi) * self.A + 0.25*dxi*gsi*(self.L + fi1*dxi)
        self.A = self.A + 0.25*dxi*gsi*(self.L + fi1*dxi)

        self.B = self.B + 0.5*hsi*dxi

        self.data.append(self.A/self.B)

        #self.i += 1

        return self.data[-1]

    # Uses the naive integration method. Returns the result and doesn't store the result in self.data. Used only for
    # temporary integration when the integration handler doesn't have enough data to be initialised.
    def integrateNaive(self, newx: float) -> float:
        numerator: float = 0
        denominator: float = 0

        self.x.append(newx)

        # Reverse the list of decreasing inputs to make integration more intuitive. This will not be a bottleneck.
        x = self.x[::-1]
        x_tr = [self.tr(xx) for xx in x]
        f = self.f
        g = self.g
        h = self.h

        for i in range(len(x)-1):
            dxi = abs(x_tr[i+1] - x_tr[i])
            innerIntegral = 0
            for j in range(i):
                dxj = abs(x_tr[j+1] - x_tr[j])
                innerIntegral += dxj*(g(x[j]) + g(x[j+1]))
            innerIntegralExtra = dxi*(g(x[i]) + g(x[i+1]))
            numerator += 0.25*dxi*(f(x[i])*innerIntegral + f(x[i+1])*(innerIntegral + innerIntegralExtra))

            denominator += 0.5*dxi*(h(x[i]) + h(x[i+1]))

        self.data.append(numerator/denominator)

        return self.data[-1]


def test1():
    print('test1:')
    import numpy as np
    a = 2
    b = 3
    firstPoints = np.linspace(a, b, 1000)[::-1]
    outFun = lambda x: x
    inFun = lambda x: x + 1
    trans = lambda x: x
    integrator = CubedNestedIntegrationHelper(firstPoints[0], outFun, inFun, trans)
    integrator.x = list(firstPoints[:-1])
    print('naive:    ', integrator.integrateNaive(firstPoints[-1]))
    exactResult = ((a - b)**4 * (35*a**4 + 4*a**3*(38 + 35*b)
                                 + a**2*(224 + 608*b + 210*b**2)
                                 + 4*a*(28 + 224*b + 170*b**2 + 35 * b ** 3)
                                 + b*(448 + 560*b + 240*b**2 + 35*b**3))) / 2240
    integrator = CubedNestedIntegrationHelper(firstPoints[0], outFun, inFun, trans)
    lastResult = 0.
    for x in firstPoints[1:]:
        lastResult = integrator.integrate(x)
    print('efficient:', lastResult)
    print('exact:    ', exactResult)

    integrator = LinearNestedNormalisedIntegrationHelper(firstPoints[0], outFun, inFun, outFun, trans)
    integrator.x = list(firstPoints[:-1])
    print('naive:    ', integrator.integrateNaive(firstPoints[-1]))
    exactResult = (b*(8 + 3*b) + a*(8*a/(a + b) - 3*(4 + a))) / 12
    integrator = LinearNestedNormalisedIntegrationHelper(firstPoints[0], outFun, inFun, outFun, trans)
    # integrator.initialiseIntegration()
    lastResult = 0.
    for x in firstPoints[1:]:
        lastResult = integrator.integrate(x)
    print('efficient:', lastResult)
    print('exact:    ', exactResult)


def test2():
    print('test2:')

    import numpy as np
    a = 2
    b = 3
    firstPoints = np.linspace(a, b, 1000)[::-1]
    outFun = lambda x: (x+4)*x
    inFun = lambda x: x*x
    trans = lambda x: x
    integrator = CubedNestedIntegrationHelper(firstPoints[0], outFun, inFun, trans)
    integrator.x = list(firstPoints[:-1])
    print('naive:    ', integrator.integrateNaive(firstPoints[-1]))
    exactResult = (486*a**11 + 55*a**12 - 220*a**9*b**2*(6 + b) - 110*a**3*b**8*(9 + 2*b) + 66*a**6*b**5*(24 + 5*b)
                   + 5*b**11*(48 + 11*b)) / 17820
    integrator = CubedNestedIntegrationHelper(firstPoints[0], outFun, inFun, trans)
    lastResult = 0.
    for x in firstPoints[1:]:
        lastResult = integrator.integrate(x)
    print('efficient:', lastResult)
    print('exact:    ', exactResult)

    integrator = LinearNestedNormalisedIntegrationHelper(firstPoints[0], outFun, inFun, outFun, trans)
    integrator.x = list(firstPoints[:-1])
    print('naive:    ', integrator.integrateNaive(firstPoints[-1]))
    exactResult = (36*a**5 + 5*a**6 - 10*a**3*b**2*(6 + b) + b**5*(24 + 5*b)) / (30*(b**2*(6 + b) - a**2*(6 + a)))
    integrator = LinearNestedNormalisedIntegrationHelper(firstPoints[0], outFun, inFun, outFun, trans)
    # integrator.initialiseIntegration()
    lastResult = 0.
    for x in firstPoints[1:]:
        lastResult = integrator.integrate(x)
    print('efficient:', lastResult)
    print('exact:    ', exactResult)


if __name__ == "__main__":
    test1()
    test2()
