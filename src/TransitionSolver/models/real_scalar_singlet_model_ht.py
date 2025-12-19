from .analysable_potential import AnalysablePotential
import numpy as np


class RealScalarSingletModel_HT(AnalysablePotential):
    def init(self, c1, b3, theta, vs, ms, muh=0, mus=0, lh=0, ls=0, c2=0):
        # We have two fields: h and s.
        self.Ndim = 2

        self.ndof = 107.75
        self.raddof = self.ndof  # The dof that we don't explicitly include in VT. These must be accounted for in the
        #                          radiation for the free energy density.

        self.c1 = c1
        self.b3 = b3
        self.theta = theta
        self.vs = vs
        self.ms2 = ms**2

        if muh != 0:
            self.muh = muh
            self.mus = mus
            self.lh = lh
            self.ls = ls
            self.c2 = c2

        # Trig functions evaluated for theta (at the VEV and T=0), to avoid future evaluations.
        self.cos2Theta = np.cos(2*self.theta)
        self.sin2Theta = np.sin(2*self.theta)
        self.cosSqTheta = np.cos(self.theta)**2
        self.sinSqTheta = np.sin(self.theta)**2

        self.setCouplingsAndScale()

        # Used to determine the overall scale of the problem. This is used (for example) to estimate what is a
        # reasonable step size for a small offset in field space (e.g. for derivatives). Of course, in such an
        # application, a small fraction of this scale is used as the offset.
        self.fieldScale = self.vh
        self.temperatureScale = 300

        self.bValid = True

        if muh == 0:
            self.constrainParameters()

        self.Tmax = 300.0
        self.forbidPhaseCrit = lambda X: True if (X[0] < -1.0) else False  # don't let h go too negative

    def setCouplingsAndScale(self):
        # From PDG (28/05/2021).
        mW = 80.379
        mZ = 91.1876
        mt = 162.5  # Using running mass rather than pole mass.
        alpha = 1 / 137.036
        self.mh2 = 125.1**2

        e = np.sqrt(4*np.pi*alpha)
        sinWeinbergAngle = np.sqrt(1 - (mW/mZ)**2)
        cosWeinbergAngle = mW/mZ

        self.gp = e/cosWeinbergAngle
        self.g = e/sinWeinbergAngle

        self.vh = 2*mW/self.g
        vhZ = 2*mZ/np.sqrt(self.g**2 + self.gp**2)

        self.yt = np.sqrt(2)*mt/self.vh

        self.renormScale = mZ
        self.renormScaleSq = mZ*mZ

    def getParameterPoint(self):
        return [self.c1, self.b3, self.theta, self.vs, np.sqrt(self.ms2), self.muh, self.mus, self.lh, self.ls, self.c2]

    # Tree-level potential.
    def V0(self, X):
        X = np.array(X)
        h = X[..., 0]
        s = X[..., 1]

        return self.muh*h**2 + self.mus*s**2 + self.lh*h**4 + self.ls*s**4 + self.c1*h**2*s + self.c2*h**2*s**2\
            + self.b3*s**3

    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]

        Pih = self.debyeCorrectionHiggs(T) if T > 0 else 0.0
        Pis = self.debyeCorrectionSinglet(T) if T > 0 else 0.0

        return (self.muh + 0.5*Pih)*h**2 + (self.mus + 0.5*Pis)*s**2\
            + self.lh*h**4 + self.ls*s**4 + self.c1*h**2*s + self.c2*h**2*s**2 + self.b3*s**3

    def approxZeroTMin(self):
        return np.array([[self.vh, self.vs]])

    def calculateMixingAngle_tree(self, X):
        h = X[..., 0]
        s = X[..., 1]

        arg = (8*self.c2*h*s + 4*self.c1*h) / (self.mh2 - self.ms2)

        if arg < -1 or arg > 1:
            return None
        return 0.5*np.arcsin(arg)

    def calculateMixingAngle_tan(self, X, T):
        mhh = self.d2Vdh2(X) + (self.debyeCorrectionHiggs(T) if T > 0 else 0.0)
        mhs = self.d2Vdhds(X)
        mss = self.d2Vds2(X) + (self.debyeCorrectionSinglet(T) if T > 0 else 0.0)

        return self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

    def calculateMixingAngle_tan_supplied(self, mhh, mhs, mss):
        return 0.5*np.arctan(2*mhs/(mhh - mss))

    def getTachyonRatio(self):
        vev = np.array([self.vh, self.vs])

        return self.d2Vdhds(vev)**2 / (self.d2Vdh2(vev) * self.d2Vds2(vev))

    def debyeCorrectionHiggs(self, T):
        return (3/16*self.g**2 + self.gp**2/16 + self.yt**2/4 + 2*self.lh + self.c2/6)*T**2

    def debyeCorrectionSinglet(self, T):
        return (2*self.c2/3 + self.ls)*T**2

    def constrainParameters(self):
        # Update muh, mus, lh, ls and c2 to satisfy the constraints.
        self.lh = 1/(8*self.vh**2) * (self.mh2*self.cosSqTheta + self.ms2*self.sinSqTheta)
        self.ls = 1/(8*self.vs**2) * (self.mh2*self.sinSqTheta + self.ms2*self.cosSqTheta + self.c1*self.vh**2/self.vs
            - 3*self.b3*self.vs)
        self.c2 = 1/(8*self.vh*self.vs) * ((self.mh2 - self.ms2)*self.sin2Theta - 4*self.c1*self.vh)
        self.muh = -2*self.lh*self.vh**2 - self.c1*self.vs - self.c2*self.vs**2
        self.mus = -1/(2*self.vs) * (4*self.ls*self.vs**3 + self.c1*self.vh**2 + 2*self.c2*self.vh**2*self.vs
            + 3*self.b3*self.vs**2)

        self.bValid = self.calculateMixingAngle_tan(np.array([self.vh, self.vs]), 0) is not None

    def d2Vdh2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.muh + 12*self.lh*h**2 + 2*self.c1*s + 2*self.c2*s**2

    def d2Vds2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.mus + 12*self.ls*s**2 + 2*self.c2*h**2 + 6*self.b3*s

    def d2Vdhds(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.c1*h + 4*self.c2*h*s

    def couldBeSubcritical(self):
        vs2 = ((-3*self.b3 + np.sqrt(9*self.b3**2 - 32*self.ls*self.mus)) / (8*self.ls))
        origin = self.V0(np.array([0.0, 0.0]))
        newPoint = self.V0(np.array([0.0, vs2]))
        vev = self.V0(np.array([0.0, self.vs]))
        return (vs2 > self.vs) and vev < newPoint and vev < origin, vs2
