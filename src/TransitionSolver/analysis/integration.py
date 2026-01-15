"""
Integration helper methods
==========================
"""


class CubedNestedIntegrationHelper:
    """
    Designed specifically to aid in evaluating integrals of the form
       I(x) = x^-3 * {int_x^a dx' self.f(x') [int_x^x' dx'' self.g(x'')]^3} ,
    where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm, where n is the number
    of evaluations, assuming n is also the step size in the integration. However, this can be reduced to O(n) using a
    recurrence relation between I(x_i) and I(x_{i+1}).
    This class assumes a new, smaller* x is passed to the integration for each interval.
    *smaller after it has been transformed using transform.
    """

    def __init__(self, points, outer_function, inner_function, transform):
        self.x = points
        self.f = outer_function
        self.g = inner_function
        self.tr = transform
        self.i = 0

        self.data = [0.]

        self.L2 = 0
        self.L3 = 0
        self.L6 = 0

        dxn2 = self.tr(self.x[0]) - self.tr(self.x[1])
        dxn3 = self.tr(self.x[1]) - self.tr(self.x[2])

        fn1 = self.f(self.x[0])
        fn2 = self.f(self.x[1])
        fsn2 = fn2 + fn1

        gsn3 = self.g(self.x[2]) + self.g(self.x[1])
        gsn2 = self.g(self.x[1]) + self.g(self.x[0])

        self.data.append(fn1 * gsn2**3 * dxn2**4 / (16.0))
        self.data.append((fn2*gsn3**3*dxn3**3*(dxn3 + dxn2) + fn1*(gsn3*dxn3 + gsn2*dxn2)**3*dxn2) / (16.0))

        self.S1 = (gsn3*dxn3)**3 * fsn2*dxn2
        self.S2 = 0
        self.S3 = 0
        self.S4 = 3*fn1*gsn3**2*dxn3**2*gsn2*dxn2**2
        self.S5 = 3*fn1*gsn3*dxn3*gsn2**2*dxn2**3
        self.S6 = 0

        self.i = 3

    def integrate(self, x: float):

        self.x.append(x)

        xi = self.tr(self.x[self.i])
        xi1 = self.tr(self.x[self.i-1])
        xi2 = self.tr(self.x[self.i-2])

        dxi = xi1 - xi
        dxi1 = xi2 - xi1
        dxi2 = self.tr(self.x[self.i-3]) - xi2

        fi1 = self.f(self.x[self.i-1])
        fi2 = self.f(self.x[self.i-2])
        fi3 = self.f(self.x[self.i-3])
        fsi1 = fi1 + fi2
        fsi2 = fi2 + fi3
        gsi = self.g(self.x[self.i]) + self.g(self.x[self.i-1])
        gsi1 = self.g(self.x[self.i-1]) + self.g(self.x[self.i-2])
        gsi2 = self.g(self.x[self.i-2]) + self.g(self.x[self.i-3])

        # We want to use L2(i+1) in L3(i), so do L3 before updating L2
        self.L3 += gsi2*dxi2*self.L2
        self.L2 += fsi2*dxi2
        self.L6 += fi3*gsi2*dxi2**2

        factor = (gsi*dxi)/(gsi1*dxi1)

        self.S1 = factor**3 * self.S1 + fsi1*gsi**3*dxi**3*dxi1
        self.S2 = factor**2 * self.S2 + 3*gsi**2*gsi1*dxi**2*dxi1 * self.L2
        self.S3 = factor * self.S3 + 3*gsi*gsi1*dxi*dxi1*(2*self.L3 + gsi1*dxi1*self.L2)
        self.S4 = factor**2 * self.S4 + 3*fi2*gsi**2*gsi1*dxi**2*dxi1**2
        self.S5 = factor * self.S5 + 3*fi2*gsi*gsi1**2*dxi*dxi1**3
        self.S6 = factor * self.S6 + 6*gsi*gsi1*dxi*dxi1 * self.L6

        Si = self.S1 + self.S2 + self.S3 + self.S4 + self.S5 + self.S6

        Ti = fi1*gsi**3*dxi**4

        self.data.append(self.data[-1] + (Si + Ti) / 16.0)

        self.i += 1


class LinearNestedNormalisedIntegrationHelper:
    """
    Designed specifically to aid in evaluating integrals of the form
       I(x) = x^-1 {[int_x^a dx' self.f(x') int_x^x' dx'' self.g(x'')] / [int_x^a dx' self.h(x')]} ,
    where it must be evaluated for several values of x. Naively this would be an O(n^3) algorithm,
    where n is the number of evaluations, assuming n is also the step size in the integration. However, this can be
    reduced to O(n) using a recurrence relation between I(x_i) and I(x_{i+1}).
    This class assumes a new, smaller x is passed to the integration for each interval (where x is smaller after it has
    been transformed using transform).
    This integration scheme requires two initial x points before the main recurrence relations can be used.
    """

    def __init__(self, points, outer_function, inner_function, normalisation_function, transform):
        self.x = points
        self.f = outer_function
        self.g = inner_function
        self.h = normalisation_function
        self.tr = transform

        self.data = [0.0]
        self.L = 0

        dxn2 = self.tr(self.x[0]) - self.tr(self.x[1])
        gsn2 = self.g(self.x[1]) + self.g(self.x[0])

        self.A = 0.25/self.tr(self.x[1])*self.f(self.x[0])*gsn2*dxn2**2
        # self.L is still 0 for point n-2
        self.B = 0.5*(self.h(self.x[1]) + self.h(self.x[0]))*dxn2

        self.data.append(self.A/self.B)

        self.i = 2

    def integrate(self, x: int):
        self.x.append(x)

        xi = self.tr(self.x[self.i])
        xi1 = self.tr(self.x[self.i-1])

        dxi = self.tr(self.x[self.i-1]) - self.tr(self.x[self.i])
        dxi1 = self.tr(self.x[self.i-2]) - self.tr(self.x[self.i-1])

        fi1 = self.f(self.x[self.i-1])
        fsi1 = fi1 + self.f(self.x[self.i-2])
        gsi = self.g(self.x[self.i]) + self.g(self.x[self.i-1])
        hsi = self.h(self.x[self.i]) + self.h(self.x[self.i-1])

        self.L += fsi1*dxi1

        self.A = (xi1/xi) * self.A + 0.25/xi*dxi*gsi*(self.L + fi1*dxi)

        self.B = self.B + 0.5*hsi*dxi

        self.data.append(self.A/self.B)

        self.i += 1
