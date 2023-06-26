import numpy as np
from models.analysable_potential import AnalysablePotential


class SMplusCubic(AnalysablePotential):
    # kap = [-1.87,-1.88,-1.89,-1.9,-1.91,-1.92]*125.**2 /246.
    def init(self,kap=-121.95,yt=0.9946,g=0.6535,g1=0.35):
        self.v = 246.
        self.v2 = self.v**2
        self.mh = 125.

        # The number of scalar fields in the model.
        self.Ndim = 1
        self.ndof = 106.75
        # We account for 23.5 dof in the one-loop corrections, so 106.75-23.5 = 83.25 of them needed to be accounted for
        # in the radiation term.
        self.raddof = 83.25
        # Don't do use simple dof, we'll calculate them via temp and field
        self.bUseSimpleDOF = False

        # setting gs energy via the potential at zero temp
        self.groundStateEnergy = self.Vtot(self.v, 0.)

        # Set the temp and field scale by the EW VEV
        # TODO: these need to be set correctly. vh is copied from the real scalar singlet model but there is vh here.
        self.fieldScale = self.v
        # Reducing temperature scale because we want to consider hundreds of MeV too.
        self.temperatureScale = 0.1*self.v

        # set this to zero since we do want to consider very low temps here
        self.minimumTemperature = 0

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = self.v**2

        self.kap = kap
        self.mu2 = .5*(self.mh**2 + self.v*self.kap)
        self.lam = .5/self.v2 *(self.mh**2 - self.v*self.kap)
        self.yt = yt
        self.g = g
        self.g1 = g1

    def forbidPhaseCrit(self, X):
        return (np.array([X])[...,0] < -5.0).any()

    def V0(self, X):
        X = np.asanyarray(X)
        rho = X[...,0]
        r = -.5*self.mu2*rho**2 + self.kap*rho**3 / 3 + .25*self.lam*rho**4
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        rho = X[...,0]
        rhoSq = rho**2
        TSq = T**2
        g2Sq = self.g**2
        g1Sq = self.g1**2

        h2 = 3*self.lam*rhoSq + 2*self.kap*rho - self.mu2 + TSq*(self.lam/4 + g2Sq + (g2Sq + g1Sq)/16 + self.yt**2/4)
        W2 = g2Sq/4*rhoSq + 11/6*g2Sq*TSq
        a = (g2Sq + g1Sq)/4*rho**2 + 11/6*(g2Sq + g1Sq)*TSq
        Delta = np.sqrt(a**2 - 11/3*g1Sq*g2Sq*(11/3 + rhoSq)*TSq)
        Z2 = 0.5*(a + Delta)
        ph2 = 0.5*(a - Delta)

        M = np.array([h2, W2, Z2, ph2])
        M = np.rollaxis(M, 0, len(M.shape))

        dof = np.array([1, 6, 3, 3])
        c = np.array([1.5, 5/6, 5/6])

        return M, dof, c

    def fermion_massSq(self, X):
        X = np.array(X)
        rho = X[...,0]

        m12 = self.yt**2/2 * rho**2

        massSq = np.array([m12])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))

        dof = np.array([12])

        return massSq, dof

    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model, and we want to include both of them.
        return [np.array([0]), np.array([self.v])]
