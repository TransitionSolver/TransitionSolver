import numpy as np
from models.analysable_potential import AnalysablePotential


class SMplusCubic(AnalysablePotential):
    # kap = [-1.87,-1.88,-1.89,-1.9,-1.91,-1.92]*125.**2 /246.
    # Previously, default value for kap was -121.95. Now no default value.
    # The idea is that if only kap is supplied, the muSq and lam are extracted at the one-loop level. Otherwise, they
    # are taken as input, presumably from a previous run of the one-loop extraction (e.g. read from saved benchmark).
    def init(self, kap, muSq=0, lam=0, mu0Sq=0, yt=0.9946, g=0.6535, g1=0.35, bDebugIteration=False):
        self.v = 246.
        self.vSq = self.v**2
        self.mh = 125.
        self.mhSq = self.mh**2

        # The number of scalar fields in the model.
        self.Ndim = 1
        self.ndof = 106.75
        # We account for 21.5 dof in the one-loop corrections, so 106.75-21.5 = 85.25 of them need to be accounted for
        # in the radiation term.
        self.raddof = 85.25
        # Don't do use simple dof, we'll calculate them via temp and field
        self.bUseSimpleDOF = False

        # Set the temp and field scale by the EW VEV
        self.fieldScale = self.v
        # Reducing temperature scale because we want to consider hundreds of MeV too.
        self.temperatureScale = 0.1*self.v

        # set this to zero since we do want to consider very low temps here
        self.minimumTemperature = 0

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = self.vSq

        self.yt = yt
        self.g = g
        self.g1 = g1

        self.kap = kap
        # This is only tree-level extraction. We need one-loop extraction.
        # self.mu2 = .5*(self.mh**2 + self.v*self.kap)
        # self.lam = .5/self.v2 *(self.mh**2 - self.v*self.kap)

        # Everything below here in this function is not necessary for the model in PhaseTracer.

        self.bValid = True

        if muSq != 0:
            self.muSq = muSq
            self.lam = lam

            if mu0Sq != 0:
                self.mu0Sq = mu0Sq

        if muSq == 0:
            self.constrainParameters(bDebugIteration)
        elif mu0Sq == 0:
            self.calculateQuadraticsForMasses()

        # setting gs energy via the potential at zero temp
        self.groundStateEnergy = self.Vtot(np.array([self.v]), 0.)

    # Tree-level potential.
    def V0(self, X):
        X = np.asanyarray(X)
        rho = X[...,0]
        r = -.5*self.muSq*rho**2 + self.kap*rho**3 / 3 + .25*self.lam*rho**4
        return r

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        rho = X[...,0]
        rhoSq = rho**2
        TSq = T**2
        g2Sq = self.g**2
        g1Sq = self.g1**2

        h2 = 3*self.lam*rhoSq + 2*self.kap*rho - self.mu0Sq + TSq*(self.lam/4 + g2Sq + (g2Sq + g1Sq)/16 + self.yt**2/4)
        W2_L = g2Sq/4*rhoSq + 11/6*g2Sq*TSq
        W2_T = g2Sq/4*rhoSq
        a = (g2Sq + g1Sq)/4*rho**2 + 11/6*(g2Sq + g1Sq)*TSq
        Delta = np.sqrt(a**2 - 11/3*g1Sq*g2Sq*(11/3 + rhoSq)*TSq)
        Z2_L = 0.5*(a + Delta)
        Z2_T = (g2Sq + g1Sq)/4*rho**2 
        ph2_L = 0.5*(a - Delta)

        M = np.array([h2, W2_L, W2_T, Z2_L, Z2_T, ph2_L])
        M = np.rollaxis(M, 0, len(M.shape))

        dof = np.array([1, 2, 4, 1, 2, 1])
        c = np.array([1.5, 5/6, 5/6, 5/6, 5/6, 5/6])

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

    """
    ====================================================================================================================
    One-loop extraction of Lagrangian parameters using EWSB and Higgs mass.
    Not necessary in PhaseTracer because there we take all parameters as input.
    ====================================================================================================================
    """

    def calculateQuadraticsForMasses(self):
        # Tree-level extraction.
        self.mu0Sq = 0.5*(self.mhSq + self.kap*self.v)

    def HiggsMass(self):
        return -self.muSq + 2*self.kap*self.v + 3*self.lam*self.vSq + self.d2VCW(np.array([self.v]))

    def constrainParameters(self, bDebugIteration=False):
        self.muSq, self.lam, self.mu0Sq = 0, 0, 0
        self.constrainParametersAtTreeLevel()
        self.muSqPrev, self.lamPrev, self.mu0SqPrev = self.muSq, self.lam, self.mu0Sq

        hasConverged = False
        i = 0
        iterMax = 50

        while not hasConverged:
            if i > iterMax:
                if bDebugIteration:
                    print("Failed to converge after", iterMax, "iterations.")

                self.bValid = False
                return

            if bDebugIteration:
                print(str(i) + '. ' + str(self.muSq) + '\t' + str(self.lam))
                print('Higgs mass:', np.sqrt(abs(self.HiggsMass())))
            i += 1

            self.constrainParametersAtOneLoop()

            if not self.bValid:
                if bDebugIteration:
                    print("Invalid at one-loop after iteration:", i)
                return

            if not self.checkParameterMagnitudes():
                self.bValid = False

                if bDebugIteration:
                    print("Invalid at one-loop after iteration:", i)

                # Rollback to the previous parameter values so we can report the unstable set of values before they
                # diverge.
                self.rollbackIteration()
                return

            if bDebugIteration:
                print(str(abs(self.muSq - self.muSqPrev)) + '\t' + str(abs(self.lam - self.lamPrev)))
            hasConverged = self.checkConvergence()

            self.muSqPrev, self.lamPrev, self.mu0SqPrev = self.muSq, self.lam, self.mu0Sq

        if bDebugIteration:
            print("Converged after", i, "iterations")
            print(str(i) + '. ' + str(self.muSq) + '\t' + str(self.lam) + '\t' + str(self.mu0Sq))
            print('Higgs mass:', np.sqrt(abs(self.HiggsMass())))

    def constrainParametersAtTreeLevel(self):
        self.lam = (self.mhSq - self.kap*self.v) / (2*self.vSq)
        self.muSq = self.kap*self.v + self.lam*self.vSq

        # Store the quadratic parameters calculated from the quadratic parameters using tree-level EWSB. These are used
        # for masses that enter the Coleman-Weinberg potential.
        self.mu0Sq = self.muSq

    def constrainParametersAtOneLoop(self):
        vev = np.array([self.v])
        dVCW = self.dVCW(vev)
        d2VCW = self.d2VCW(vev)

        self.lam = (self.mhSq - self.kap*self.v + dVCW/self.v - d2VCW) / (2*self.vSq)
        self.muSq = self.kap*self.v + self.lam*self.vSq + dVCW/self.v

        # Store the quadratic parameters calculated from the quadratic parameters using tree-level EWSB. These are used
        # for masses that enter the Coleman-Weinberg potential.
        self.mu0Sq = self.kap*self.v + self.lam*self.vSq

    def checkParameterMagnitudes(self):
        quadraticMaxVal = 1e10
        quarticMaxVal = 100

        if abs(self.muSq) > quadraticMaxVal or abs(self.lam) > quarticMaxVal:
            # print('Diverging parameters, potential not valid.')
            return False

        return True

    def rollbackIteration(self):
        self.muSq, self.lam, self.mu0Sq = self.muSqPrev, self.lamPrev, self.mu0SqPrev

    def checkConvergence(self):
        tol = 1e-5
        return abs(self.muSq - self.muSqPrev) < tol and abs(self.lam - self.lamPrev) < tol

    def forbidPhaseCrit(self, X):
        return (np.array([X])[...,0] < -5.0).any()

    def dVCW(self, X, T=0):
        m, n, c = self.boson_massSq(X, T)
        dm = self.d_boson_massSq(X)

        y = 0

        for i in range(len(m)):
            if m[i] > 1e-10:
                y += n[i]*m[i]*dm[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)

        m, n = self.fermion_massSq(X)
        dm = self.d_fermion_massSq(X)
        c = 1.5

        for i in range(len(m)):
            if m[i] > 1e-10:
                y -= n[i]*m[i]*dm[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)

        return y/(32*np.pi**2)

    def d2VCW(self, X, T=0):
        m, n, c = self.boson_massSq(X, T)
        dm = self.d_boson_massSq(X)
        d2m = self.d2_boson_massSq(X)

        y = 0

        for i in range(len(m)):
            if m[i] > 1e-10:
                y += n[i]*((dm[i]**2 + m[i]*d2m[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)
                    + dm[i]**2)

        m, n = self.fermion_massSq(X)
        dm = self.d_fermion_massSq(X)
        d2m = self.d2_fermion_massSq(X)
        c = 1.5

        for i in range(len(m)):
            if m[i] > 1e-10:
                y -= n[i]*((dm[i]**2 + m[i]*d2m[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)
                    + dm[i]**2)

        return y/(32*np.pi**2)

    def d_boson_massSq(self, X):
        X = np.array(X)
        rho = X[...,0]
        g2Sq = self.g**2
        g1Sq = self.g1**2

        h2 = 6*self.lam*rho + 2*self.kap
        W2_L = W2_T = g2Sq/2*rho
        Z2_L = Z2_T = (g2Sq + g1Sq)/2*rho
        ph2_L = 0

        M = np.array([h2, W2_L, W2_T, Z2_L, Z2_T, ph2_L])
        M = np.rollaxis(M, 0, len(M.shape))

        return M

    def d2_boson_massSq(self, X):
        g2Sq = self.g**2
        g1Sq = self.g1**2

        h2 = 6*self.lam
        W2_L = W2_T = g2Sq/2
        Z2_L = Z2_T = (g2Sq + g1Sq)/2
        ph2_L = 0

        M = np.array([h2, W2_L, W2_T, Z2_L, Z2_T, ph2_L])
        M = np.rollaxis(M, 0, len(M.shape))

        return M

    def d_fermion_massSq(self, X):
        X = np.array(X)
        rho = X[...,0]

        m12 = self.yt**2 * rho

        massSq = np.array([m12])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))

        return massSq

    def d2_fermion_massSq(self, X):
        m12 = self.yt**2

        massSq = np.array([m12])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))

        return massSq


if __name__ == "__main__":
    potential = SMplusCubic(-1.9*125**2/246, bDebugIteration=True)

