class IntegrationHelper:
    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)
        self.initialiseIntegration()

    def init(self, *args, **kwargs):
        return

    def initialiseIntegration(self):
        return


# Designed specifically to aid in evaluating integrals of the form
#   I(x) = x^-3 * {int_x^a dx' f(x') [int_x^x' dx'' g(x'')]^3} ,
# where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm, where n is the number
# of evaluations, assuming n is also the step size in the integration. However, this can be reduced to O(n) using a
# recurrence relation between I(x_i) and I(x_{i+1}).
# This class assumes a new, smaller* x is passed to the integration for each interval.
# *smaller after it has been transformed using sampleTransformationFunction.
class CubedNestedIntegrationHelper(IntegrationHelper):
    def init(self, initialPoints, outerFunction, innerFunction, sampleTransformationFunction):
        self.x = initialPoints
        self.f = outerFunction
        self.g = innerFunction
        self.tr = sampleTransformationFunction
        self.i = 0

        self.data = [0.0]

    def initialiseIntegration(self):
        self.L2 = 0
        self.L3 = 0
        self.L6 = 0

        x = self.x
        f = self.f
        g = self.g
        tr = self.tr

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
        self.S2 = 0
        self.S3 = 0
        self.S4 = 3*fn1*gsn3**2*dxn3**2*gsn2*dxn2**2
        self.S5 = 3*fn1*gsn3*dxn3*gsn2**2*dxn2**3
        self.S6 = 0

        self.i = 3

    def integrate(self, newx: float):
        x = self.x
        f = self.f
        g = self.g
        tr = self.tr
        i = self.i

        x.append(newx)

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

        self.i += 1


# Designed specifically to aid in evaluating integrals of the form
#   I(x) = x^-1 {[int_x^a dx' f(x') int_x^x' dx'' g(x'')] / [int_x^a dx' h(x')]} ,
# where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm,
# where n is the number of evaluations, assuming n is also the step size in the integration. However, this can be
# reduced to O(n) using a recurrence relation between I(x_i) and I(x_{i+1}).
# This class assumes a new, smaller x is passed to the integration for each interval (where x is smaller after it has
# been transformed using sampleTransformationFunction).
# This integration scheme requires two initial x points before the main recurrence relations can be used.
class LinearNestedNormalisedIntegrationHelper(IntegrationHelper):
    def init(self, initialPoints, outerFunction, innerFunction, normalisationFunction, sampleTransformationFunction):
        self.x = initialPoints
        self.f = outerFunction
        self.g = innerFunction
        self.h = normalisationFunction
        self.tr = sampleTransformationFunction
        self.i = 0

        self.data = [0.0]

    def initialiseIntegration(self):
        self.L = 0
        self.A = 0
        self.B = 0

        x = self.x
        f = self.f
        g = self.g
        h = self.h
        tr = self.tr

        dxn2 = tr(x[0]) - tr(x[1])

        fn1 = f(x[0])
        fn2 = f(x[1])
        gsn2 = g(x[1]) + g(x[0])

        self.A = 0.25/tr(x[1])*f(x[0])*gsn2*dxn2**2
        # self.L is still 0 for point n-2.
        self.B = 0.5*(h(x[1]) + h(x[0]))*dxn2

        self.data.append(self.A/self.B)

        self.i = 2

    def integrate(self, newx: int):
        x = self.x
        f = self.f
        g = self.g
        h = self.h
        tr = self.tr
        i = self.i

        x.append(newx)

        xi = tr(x[i])
        xi1 = tr(x[i-1])

        dxi = tr(x[i-1]) - tr(x[i])
        dxi1 = tr(x[i-2]) - tr(x[i-1])

        fi1 = f(x[i-1])
        fsi1 = fi1 + f(x[i-2])
        gsi = g(x[i]) + g(x[i-1])
        hsi = h(x[i]) + h(x[i-1])

        self.L += fsi1*dxi1

        self.A = (xi1/xi) * self.A + 0.25/xi*dxi*gsi*(self.L + fi1*dxi)

        self.B = self.B + 0.5*hsi*dxi

        self.data.append(self.A/self.B)

        self.i += 1
