from models.analysable_potential import AnalysablePotential
import numpy as np


class MassMatrixData:
    def __init__(self, mhh, mhs, mss, theta):
        self.mhh = mhh
        self.mhs = mhs
        self.mss = mss
        self.theta = theta

        self.sinSq = np.sin(theta)**2
        self.cosSq = 1 - self.sinSq
        self.sin2 = np.sin(2*theta)
        self.cos2 = self.cosSq - self.sinSq


class RealScalarSingletModel(AnalysablePotential):
    def init(self, c1, b3, theta, vs, ms, muh=0, mus=0, lh=0, ls=0, c2=0, muh0=0, mus0=0, bDebugIteration=False,
            bStoreParameterEvolution=False, bSkipOneLoop=False):
        # We have two fields: h and s.
        self.Ndim = 2

        self.ndof = 107.75
        self.raddof = 80.25  # The dof that we don't explicitly include in VT. These must be accounted for in the
        #                      radiation for the free energy density.
        self.bUseSimpleDOF = False

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

            if muh0 != 0:
                self.muh0 = muh0
                self.mus0 = mus0

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

        # Setting this to 1e-10 causes issues where the Goldstone mass can be just above 1e-10 and then we include the
        # massShiftHack term when avoiding the divergence from the Goldstone at the VEV.
        self.minMassThreshold = 1e-5
        self.massShiftHack = 1

        self.bUseGoldstoneResummation = True

        self.bValid = True

        self.parameterEvolution = []

        if muh == 0:
            self.constrainParameters(bDebugIteration, bStoreParameterEvolution=bStoreParameterEvolution,
                bSkipOneLoop=bSkipOneLoop)
        elif muh0 == 0:
            self.calculateQuadraticsForMasses()

        self.Tmax = 300.0
        self.forbidPhaseCrit = lambda X: True if (X[0] < -1.0) else False  # don't let h go too negative

        self.groundStateEnergy = self.Vtot([self.vh, self.vs], 0.)

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
        #vhZ = 2*mZ/np.sqrt(self.g**2 + self.gp**2)

        self.yt = np.sqrt(2)*mt/self.vh

        # Avoid squaring in the future.
        self.gp2 = self.gp**2
        self.g2 = self.g**2
        self.yt2 = self.yt**2

        self.renormScale = mZ
        self.renormScaleSq = mZ*mZ

    def getParameterPoint(self):
        return [self.c1, self.b3, self.theta, self.vs, np.sqrt(self.ms2), self.muh, self.mus, self.lh, self.ls, self.c2,
         self.muh0, self.mus0]

    def freeEnergyDensity(self, X, T):
        return self.Vtot(X, T, include_radiation=False) - np.pi**2/90*self.raddof*T**4

    # Tree-level potential.
    def V0(self, X):
        X = np.array(X)
        h = X[..., 0]
        s = X[..., 1]

        return self.muh*h**2 + self.mus*s**2 + self.lh*h**4 + self.ls*s**4 + self.c1*h**2*s + self.c2*h**2*s**2\
            + self.b3*s**3

    # Purely for debugging, not actually called for standard evaluation of the potential.
    def VCW(self, X, T):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions)

    # Purely for debugging, not actually called for standard evaluation of the potential.
    def VTherm(self, X, T):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        return self.V1T(bosons, fermions, T)

    def approxZeroTMin(self):
        return np.array([[self.vh, self.vs]])

    def calculateMixingAngle(self, X):
        h = X[..., 0]
        s = X[..., 1]

        d2VCWdhds = self.d2VCWdhds(X)
        arg = (8*self.c2*h*s + 4*self.c1*h + 2*d2VCWdhds) / (self.mh2 - self.ms2)

        if arg < -1 or arg > 1:
            return None
        return 0.5*np.arcsin(arg)

    def calculateMixingAngle_tree(self, X):
        h = X[..., 0]
        s = X[..., 1]

        arg = (8*self.c2*h*s + 4*self.c1*h) / (self.mh2 - self.ms2)

        if arg < -1 or arg > 1:
            return None
        return 0.5*np.arcsin(arg)

    def calculateMixingAngle_tree_tan(self, X, T):
        mhh = self.d2V0mdh2(X) + self.debyeCorrectionHiggs(T)
        mhs = self.d2V0mdhds(X)
        mss = self.d2V0mds2(X) + self.debyeCorrectionSinglet(T)

        return self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

    def calculateMixingAngle_tan_supplied(self, mhh, mhs, mss):
        arg = 2*mhs/(mhh - mss)
        if len(arg.shape) > 0:
            arg[abs(arg) < 1e-10] = 0.0
        else:
            if abs(arg) < 1e-10:
                arg = 0.0
        return 0.5*np.arctan(arg)

    def calculateMixingAngle_tan(self, X):
        mhh = self.d2Vdh2(X)
        mhs = self.d2Vdhds(X)
        mss = self.d2Vds2(X)

        return self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

    def getTachyonRatio(self):
        vev = np.array([self.vh, self.vs])

        return self.d2V0mdhds(vev)**2 / (self.d2V0mdh2(vev) * self.d2V0mds2(vev))

    def debyeCorrectionHiggs(self, T):
        return (3/16*self.g2 + self.gp2/16 + self.yt2/4 + 2*self.lh + self.c2/6)*T**2

    def debyeCorrectionSinglet(self, T):
        return (2*self.c2/3 + self.ls)*T**2

    # Following https://arxiv.org/pdf/1903.09642.pdf.
    def deltaMSq(self, T, phi_t, phi_f):
        # Bosons in true phase.
        mb_t, dofb, _ = self.boson_massSq(phi_t, T)
        # Bosons in false phase.
        mb_f, _, _ = self.boson_massSq(phi_f, T)
        # Fermions in true phase.
        mf_t, doff = self.fermion_massSq(phi_t)
        # Fermions in false phase.
        mf_f, _ = self.fermion_massSq(phi_f)

        delta = 0

        for i in range(len(mb_t)):
            delta += dofb[i]*(mb_t[i] - mb_f[i])

        for i in range(len(mf_t)):
            delta += 0.5*doff[i]*(mf_t[i] - mf_f[i])

        return delta

    # Following https://arxiv.org/pdf/2007.15586.pdf.
    def gSqDeltaM(self, T, phi_t, phi_f):
        # Bosons in true phase.
        mb_t, dof, cb = self.boson_massSq(phi_t, T)
        # Bosons in false phase.
        mb_f, _, _ = self.boson_massSq(phi_f, T)

        # These have the form g^2 * Ni * \Delta Mi.
        delta = 0.25*self.g2 * dof[1] * (np.sqrt(mb_t[1]) - np.sqrt(mb_f[1]))  # W bosons
        delta += 0.25*(self.g2 + self.gp2) * dof[2] * (np.sqrt(mb_t[2]) - np.sqrt(mb_f[2]))  # Z boson

        # NOTE: we assume the zero temperature coupling for the Z boson, neglecting mixing with the photon. In
        # https://arxiv.org/pdf/1703.08215.pdf they ignore thermal masses, so this is probably the best we can do.

        return delta

    # Following https://arxiv.org/pdf/1903.09642.pdf.
    def gSq(self):
        # Field location and temperature don't matter.
        _, dof, _ = self.boson_massSq(self.approxZeroTMin(), 0.)
        gsq = 0.25*self.g2 * dof[1]
        gsq += 0.25*(self.g2 + self.gp2) * dof[2]

        return gsq

    """
        ----------------------------------------------------------------------------------------------------------------

        Mass eigenvalues and their derivatives.

        ----------------------------------------------------------------------------------------------------------------
    """

    # The eigenvalues of the tree-level mass matrix. These are used as masses in the Coleman-Weinberg potential.
    def massEigenvalues(self, X, T=0, massMatrix=None):
        if massMatrix is None:
            Th = self.debyeCorrectionHiggs(T)
            Ts = self.debyeCorrectionSinglet(T)

            mhh = self.d2V0mdh2(X) + Th
            mhs = self.d2V0mdhds(X)
            mss = self.d2V0mds2(X) + Ts

            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        m1 = MM.mhh*MM.cosSq + MM.mhs*MM.sin2 + MM.mss*MM.sinSq
        m2 = MM.mhh*MM.sinSq - MM.mhs*MM.sin2 + MM.mss*MM.cosSq

        return m1, m2

    def massEigenvalues_characteristic(self, X):
        mhh = self.d2V0mdh2(X)
        mhs = self.d2V0mdhds(X)
        mss = self.d2V0mds2(X)

        a = mhh + mss
        b = np.sqrt((mhh - mss)**2 + 4*mhs**2)

        return 0.5*(a + b), 0.5*(a - b)

    # The eigenvalues of the one-loop mass matrix. These are used to check if the masses are reproduced after
    # constraining the parameters in the potential.
    def massEigenvalues_full(self, X, T=0, massMatrix=None):
        if massMatrix is None:
            Th = self.debyeCorrectionHiggs(T)
            Ts = self.debyeCorrectionSinglet(T)

            mhh = self.d2Vdh2(X) + Th
            mhs = self.d2Vdhds(X)
            mss = self.d2Vds2(X) + Ts

            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        m1 = MM.mhh*MM.cosSq + MM.mhs*MM.sin2 + MM.mss*MM.sinSq
        m2 = MM.mhh*MM.sinSq - MM.mhs*MM.sin2 + MM.mss*MM.cosSq

        return m1, m2

    # Returns the Higgs-like mass eigenvalue.
    def HiggsMass(self, X, massMatrix=None):
        m1, m2 = self.massEigenvalues_full(X, massMatrix=massMatrix)

        return m1

    # Returns the singlet-like mass eigenvalue.
    def SingletMass(self, X, massMatrix=None):
        m1, m2 = self.massEigenvalues_full(X, massMatrix=massMatrix)

        return m2

    # Returns the Higgs-like tree-level mass eigenvalue.
    def HiggsMass_tree(self, X, massMatrix=None):
        m1, m2 = self.massEigenvalues(X, massMatrix=massMatrix)

        return m1

    # Returns the singlet-like tree-level mass eigenvalue.
    def SingletMass_tree(self, X, massMatrix=None):
        m1, m2 = self.massEigenvalues(X, massMatrix=massMatrix)

        return m2

    """ FIRST DERIVATIVES """

    def d_massEigenvalues_dh(self, X, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        dmhhdh = self.d3V0mdh3(X)
        dmssdh = self.d3V0mdhds2(X)
        dmhsdh = self.d3V0mdh2ds(X)

        dtdh = (dmhsdh*(MM.mhh - MM.mss) - MM.mhs*(dmhhdh - dmssdh)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)

        m1 = dmhhdh*MM.cosSq + dmhsdh*MM.sin2 + dmssdh*MM.sinSq\
            + dtdh*(-MM.mhh*MM.sin2 + 2*MM.mhs*MM.cos2 + MM.mss*MM.sin2)
        m2 = dmhhdh*MM.sinSq - dmhsdh*MM.sin2 + dmssdh*MM.cosSq\
            + dtdh*(MM.mhh*MM.sin2 - 2*MM.mhs*MM.cos2 - MM.mss*MM.sin2)

        return m1, m2

    def d_massEigenvalues_ds(self, X, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        dmhhds = self.d3V0mdh2ds(X)
        dmssds = self.d3V0mds3(X)
        dmhsds = self.d3V0mdhds2(X)

        dtds = (dmhsds*(MM.mhh - MM.mss) - MM.mhs*(dmhhds - dmssds)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)

        m1 = dmhhds*MM.cosSq + dmhsds*MM.sin2 + dmssds*MM.sinSq\
            + dtds*(-MM.mhh*MM.sin2 + 2*MM.mhs*MM.cos2 + MM.mss*MM.sin2)
        m2 = dmhhds*MM.sinSq - dmhsds*MM.sin2 + dmssds*MM.cosSq\
            + dtds*(MM.mhh*MM.sin2 - 2*MM.mhs*MM.cos2 - MM.mss*MM.sin2)

        return m1, m2

    """ SECOND DERIVATIVES """

    def d2_massEigenvalues_dh2(self, X, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        dmhhdh = self.d3V0mdh3(X)
        dmssdh = self.d3V0mdhds2(X)
        dmhsdh = self.d3V0mdh2ds(X)

        d2mhhdh2 = self.d4V0mdh4(X)
        d2mssdh2 = self.d4V0mdh2ds2(X)
        d2mhsdh2 = self.d4V0mdh3ds(X)

        dtdh = (dmhsdh*(MM.mhh - MM.mss) - MM.mhs*(dmhhdh - dmssdh)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)

        d2tdh2 = (((MM.mhh - MM.mss)**2 + 4*MM.mhs**2) * (d2mhsdh2*(MM.mhh - MM.mss) - MM.mhs*(d2mhhdh2 - d2mssdh2))
            - 2*(dmhsdh*(MM.mhh - MM.mss) - MM.mhs*(dmhhdh - dmssdh)) * ((dmhhdh - dmssdh)*(MM.mhh - MM.mss)
            + 4*dmhsdh*MM.mhs)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)**2

        thetaPart = 2*dtdh*(-dmhhdh*MM.sin2 + 2*dmhsdh*MM.cos2 + dmssdh*MM.sin2)\
            + d2tdh2*(-MM.mhh*MM.sin2 + 2*MM.mhs*MM.cos2 + MM.mss*MM.sin2)\
            + 2*dtdh**2*(-MM.mhh*MM.cos2 - 2*MM.mhs*MM.sin2 + MM.mss*MM.cos2)

        m1 = d2mhhdh2*MM.cosSq + d2mhsdh2*MM.sin2 + d2mssdh2*MM.sinSq + thetaPart
        m2 = d2mhhdh2*MM.sinSq - d2mhsdh2*MM.sin2 + d2mssdh2*MM.cosSq - thetaPart

        return m1, m2

    def d2_massEigenvalues_ds2(self, X, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        dmhhds = self.d3V0mdh2ds(X)
        dmssds = self.d3V0mds3(X)
        dmhsds = self.d3V0mdhds2(X)

        d2mhhds2 = self.d4V0mdh2ds2(X)
        d2mssds2 = self.d4V0mds4(X)
        d2mhsds2 = self.d4V0mdhds3(X)

        dtds = (dmhsds*(MM.mhh - MM.mss) - MM.mhs*(dmhhds - dmssds)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)

        d2tds2 = (((MM.mhh - MM.mss)**2 + 4*MM.mhs**2) * (d2mhsds2*(MM.mhh - MM.mss) - MM.mhs*(d2mhhds2 - d2mssds2))
            - 2*(dmhsds*(MM.mhh - MM.mss) - MM.mhs*(dmhhds - dmssds)) * ((dmhhds - dmssds)*(MM.mhh - MM.mss)
            + 4*dmhsds*MM.mhs)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)**2

        thetaPart = 2*dtds*(-dmhhds*MM.sin2 + 2*dmhsds*MM.cos2 + dmssds*MM.sin2)\
            + d2tds2*(-MM.mhh*MM.sin2 + 2*MM.mhs*MM.cos2 + MM.mss*MM.sin2)\
            + 2*dtds**2*(-MM.mhh*MM.cos2 - 2*MM.mhs*MM.sin2 + MM.mss*MM.cos2)

        m1 = d2mhhds2*MM.cosSq + d2mhsds2*MM.sin2 + d2mssds2*MM.sinSq + thetaPart
        m2 = d2mhhds2*MM.sinSq - d2mhsds2*MM.sin2 + d2mssds2*MM.cosSq - thetaPart

        return m1, m2

    def d2_massEigenvalues_dhds(self, X, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix

        dmhhdh = self.d3V0mdh3(X)
        dmssdh = self.d3V0mdhds2(X)
        dmhsdh = self.d3V0mdh2ds(X)

        dmhhds = self.d3V0mdh2ds(X)
        dmssds = self.d3V0mds3(X)
        dmhsds = self.d3V0mdhds2(X)

        d2mhhdhds = self.d4V0mdh3ds(X)
        d2mssdhds = self.d4V0mdhds3(X)
        d2mhsdhds = self.d4V0mdh2ds2(X)

        dtdh = (dmhsdh*(MM.mhh - MM.mss) - MM.mhs*(dmhhdh - dmssdh)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)
        dtds = (dmhsds*(MM.mhh - MM.mss) - MM.mhs*(dmhhds - dmssds)) / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)

        d2tdhds = (((MM.mhh - MM.mss)**2 + 4*MM.mhs**2) * (d2mhsdhds*(MM.mhh - MM.mss) + dmhsdh*(dmhhds - dmssds)
            - dmhsds*(dmhhdh - dmssdh) - MM.mhs*(d2mhhdhds - d2mssdhds)) - 2*(dmhsdh*(MM.mhh - MM.mss)
            - MM.mhs*(dmhhdh - dmssdh)) * ((dmhhdh - dmssdh)*(MM.mhh - MM.mss) + 4*dmhsdh*MM.mhs))\
            / ((MM.mhh - MM.mss)**2 + 4*MM.mhs**2)**2

        thetaPart = dtds*(-dmhhdh*MM.sin2 + 2*dmhsdh*MM.cos2 + dmssdh*MM.sin2)\
            + dtdh*(-dmhhds*MM.sin2 + 2*dmhsds*MM.cos2 + dmssds*MM.sin2)\
            + d2tdhds*(-MM.mhh*MM.sin2 + 2*MM.mhs*MM.cos2 + MM.mss*MM.sin2)\
            + 2*dtdh*dtds*(-MM.mhh*MM.cos2 - 2*MM.mhs*MM.sin2 + MM.mss*MM.cos2)

        m1 = d2mhhdhds*MM.cosSq + d2mhsdhds*MM.sin2 + d2mssdhds*MM.sinSq + thetaPart
        m2 = d2mhhdhds*MM.sinSq - d2mhsdhds*MM.sin2 + d2mssdhds*MM.cosSq - thetaPart

        return m1, m2

    """ THIRD DERIVATIVES"""

    def d3jkl_massEigenvalues(self, X, j, k, l, massMatrix=None):
        if massMatrix is None:
            mhh = self.d2V0mdh2(X)
            mss = self.d2V0mds2(X)
            mhs = self.d2V0mdhds(X)
            theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

            massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        MM = massMatrix
        mhh = MM.mhh
        mss = MM.mss
        mhs = MM.mhs
        s2 = MM.sin2
        c2 = MM.cos2

        dj_mhh = self.d3jkl_V0m(X, 0, 0, j)
        dk_mhh = self.d3jkl_V0m(X, 0, 0, k)
        dl_mhh = self.d3jkl_V0m(X, 0, 0, l)
        dj_mss = self.d3jkl_V0m(X, 1, 1, j)
        dk_mss = self.d3jkl_V0m(X, 1, 1, k)
        dl_mss = self.d3jkl_V0m(X, 1, 1, l)
        dj_mhs = self.d3jkl_V0m(X, 0, 1, j)
        dk_mhs = self.d3jkl_V0m(X, 0, 1, k)
        dl_mhs = self.d3jkl_V0m(X, 0, 1, l)
        d2jk_mhh = self.d4jklm_V0m(X, 0, 0, j, k)
        d2jl_mhh = self.d4jklm_V0m(X, 0, 0, j, l)
        d2kl_mhh = self.d4jklm_V0m(X, 0, 0, k, l)
        d2jk_mss = self.d4jklm_V0m(X, 1, 1, j, k)
        d2jl_mss = self.d4jklm_V0m(X, 1, 1, j, l)
        d2kl_mss = self.d4jklm_V0m(X, 1, 1, k, l)
        d2jk_mhs = self.d4jklm_V0m(X, 0, 1, j, k)
        d2jl_mhs = self.d4jklm_V0m(X, 0, 1, j, l)
        d2kl_mhs = self.d4jklm_V0m(X, 0, 1, k, l)

        dj_t = (dj_mhs*(mhh - mss) - mhs*(dj_mhh - dj_mss)) / ((mhh - mss)**2 + 4*mhs**2)
        dk_t = (dk_mhs*(mhh - mss) - mhs*(dk_mhh - dk_mss)) / ((mhh - mss)**2 + 4*mhs**2)
        dl_t = (dl_mhs*(mhh - mss) - mhs*(dl_mhh - dl_mss)) / ((mhh - mss)**2 + 4*mhs**2)

        d2jk_t = (((mhh - mss)**2 + 4*mhs**2) * (d2jk_mhs*(mhh - mss) + dj_mhs*(dk_mhh - dk_mss)
            - dk_mhs*(dj_mhh - dj_mss) - mhs*(d2jk_mhh - d2jk_mss)) - 2*(dj_mhs*(mhh - mss) - mhs*(dj_mhh - dj_mss))
            * ((dk_mhh - dk_mss)*(mhh - mss) + 4*dk_mhs*mhs)) / ((mhh - mss)**2 + 4*mhs**2)**2
        d2jl_t = (((mhh - mss)**2 + 4*mhs**2) * (d2jl_mhs*(mhh - mss) + dj_mhs*(dl_mhh - dl_mss)
            - dl_mhs*(dj_mhh - dj_mss) - mhs*(d2jl_mhh - d2jl_mss)) - 2*(dj_mhs*(mhh - mss) - mhs*(dj_mhh - dj_mss))
            * ((dl_mhh - dl_mss)*(mhh - mss) + 4*dl_mhs*mhs)) / ((mhh - mss)**2 + 4*mhs**2)**2
        d2kl_t = (((mhh - mss)**2 + 4*mhs**2) * (d2kl_mhs*(mhh - mss) + dk_mhs*(dl_mhh - dl_mss)
            - dl_mhs*(dk_mhh - dk_mss) - mhs*(d2kl_mhh - d2kl_mss)) - 2*(dk_mhs*(mhh - mss) - mhs*(dk_mhh - dk_mss))
            * ((dl_mhh - dl_mss)*(mhh - mss) + 4*dl_mhs*mhs)) / ((mhh - mss)**2 + 4*mhs**2)**2

        A = (mhh - mss)**2 + 4*mhs**2
        B = d2jk_mhs*(mhh - mss) + dj_mhs*(dk_mhh - dk_mss) - dk_mhs*(dj_mhh - dj_mss) - mhs*(d2jk_mhh - d2jk_mss)
        C = dj_mhs*(mhh - mss) - mhs*(dj_mhh - dj_mss)
        D = (dk_mhh - dk_mss)*(mhh - mss) + 4*dk_mhs*mhs
        dl_A = 2*(dl_mhh - dl_mss)*(mhh - mss) + 8*dl_mhs*mhs
        # Neglecting d3jkl_mhs*(mhh - mss) and -mhs*(d3jkl_mhh - d3jkl_mss) as they vanish (in dl_B).
        dl_B = d2jk_mhs*(dl_mhh - dl_mss) + d2jl_mhs*(dk_mhh - dk_mss) + dj_mhs*(d2kl_mhh - d2kl_mss)\
            - d2kl_mhs*(dj_mhh - dj_mss) - dk_mhs*(d2jl_mhh - d2jl_mss) - dl_mhs*(d2jk_mhh - d2jk_mss)
        dl_C = d2jl_mhs*(mhh - mss) + dj_mhs*(dl_mhh - dl_mss) - dl_mhs*(dj_mhh - dj_mss) - mhs*(d2jl_mhh - d2jl_mss)
        dl_D = (d2kl_mhh - d2kl_mss)*(mhh - mss) + (dk_mhh - dk_mss)*(dl_mhh - dl_mss) + 4*d2kl_mhs*mhs + 4*dk_mhs*dl_mhs
        d3jkl_t = ((4*C*D/A - B)*dl_A + A*dl_B - 2*D*dl_C - 2*C*dl_D) / (A*A)

        # d3jkl_mhh*cSq + d3jkl_mhs*s2 + d3jkl_mss*cSq  --- vanishes since V0 is only quartic and these are fifth
        # derivatives of V0.
        thetaPart = dl_t * (-d2jk_mhh*s2 + 2*d2jk_mhs*c2 + d2jk_mss*c2)\
            + d2kl_t * (-dj_mhh*s2 + 2*dj_mhs*c2 + dj_mss*s2)\
            + dk_t * (-d2jl_mhh*s2 + 2*d2jl_mhs*c2 + d2jl_mss*s2)\
            + 2*dk_t * dl_t * (-dj_mhh*c2 - 2*dj_mhs*s2 + dj_mss*c2)\
            + d2jl_t * (-dk_mhh*s2 + 2*dk_mhs*c2 + dk_mss*s2)\
            + dj_t * (-d2kl_mhh*s2 + 2*d2kl_mhs*c2 + d2kl_mss*s2)\
            + 2*dj_t * dl_t * (-dk_mhh*c2 - 2*dk_mhs*s2 + dk_mss*c2)\
            + d3jkl_t * (-mhh*s2 + 2*mhs*c2 + mss*s2)\
            + d2jk_t * (-dl_mhh*s2 + 2*dl_mhs*c2 + dl_mss*s2)\
            + 2*d2jk_t * dl_t * (-mhh*c2 - 2*mhs*c2 + mss*c2)\
            + 2*(d2jl_t*dk_t + dj_t*d2kl_t) * (-mhh*c2 - 2*mhs*s2 + mss*c2)\
            + 2*dj_t * dk_t * (-dl_mhh*c2 - 2*dl_mhs*s2 + dl_mss*c2)\
            + 4*dj_t * dk_t * dl_t * (mhh*s2 - 2*mhs*c2 - 2*mss*s2)

        return thetaPart, -thetaPart

    """
        ----------------------------------------------------------------------------------------------------------------

        Goldstone self-energy and its derivatives.

        ----------------------------------------------------------------------------------------------------------------
    """

    def goldstoneMass(self, X, T=0, massMatrix=None):
        # Make a copy of the field input as we modify it if the Higgs field is close to zero. We don't want such changes
        # to persist after this function ends.
        xCopy = X.copy()
        h = xCopy[..., 0]

        # Because we divide by h, we need to shift slightly away from h=0. Shifting by 1e-5 and 1e-8 show no difference,
        # while by 1e-3 shows a 1e-12 relative error and by 1e-1 a 1e-8 relative error.
        hShift = 1e-6 * self.fieldScale

        # TODO: there's probably a better way to do this. We need to update X since that's passed into V0 and VCW.
        if len(h.shape) > 0:
            for i in range(len(h)):
                if (abs(xCopy[..., 0][i]) < hShift).any():
                    hSign = np.sign(xCopy[..., 0][i])
                    if len(hSign.shape) > 1:
                        hSign[hSign == 0] = 1
                    elif hSign == 0:
                        hSign = 1
                    xCopy[..., 0][i] = hShift*hSign
                    h[i] = xCopy[..., 0][i]
        else:
            if abs(h) < hShift:
                hSign = np.sign(h)
                h = hShift*(1 if hSign == 0 else hSign)
                xCopy[0] = h

        if self.bUseGoldstoneResummation:
            mass = 1/h * (self.dV0mdh(xCopy) + self.dj_VCW(xCopy, 0, T=0, massMatrix=massMatrix, ignoreGoldstone=True))\
                + (self.debyeCorrectionHiggs(T) if T > 0.0 else 0.0)
        else:
            mass = 1/h * self.dV0mdh(xCopy) + (self.debyeCorrectionHiggs(T) if T > 0.0 else 0.0)

        return mass

    def dj_goldstoneMass(self, X, j, massMatrix=None):
        xCopy = X.copy()
        h = xCopy[..., 0]

        # Because we divide by h, we need to shift slightly away from h=0. Shifting by 1e-5 and 1e-8 show no difference,
        # while by 1e-3 shows a 1e-12 relative error and by 1e-1 a 1e-8 relative error.
        hShift = 1e-6 * self.fieldScale

        if len(h.shape) > 0:
            for i in range(len(h)):
                if (abs(xCopy[..., 0][i]) < hShift).any():
                    hSign = np.sign(xCopy[..., 0][i])
                    if len(hSign.shape) > 1:
                        hSign[hSign == 0] = 1
                    elif hSign == 0:
                        hSign = 1
                    xCopy[..., 0][i] = hShift*hSign
                    h[i] = xCopy[..., 0][i]
        else:
            if abs(h) < hShift:
                hSign = np.sign(h)
                h = hShift*(1 if hSign == 0 else hSign)
                xCopy[0] = h

        if self.bUseGoldstoneResummation:
            d2hj_VCW = self.d2jk_VCW(xCopy, 0, j, T=0, massMatrix=massMatrix, ignoreGoldstone=True)
        else:
            d2hj_VCW = 0.0

        returnVal = self.d2jk_V0m(xCopy, 0, j) + d2hj_VCW

        if j == 0:
            returnVal -= self.goldstoneMass(xCopy, T=0, massMatrix=None)

        return returnVal / h

    def d2jk_goldstoneMass(self, X, j, k, massMatrix=None):
        xCopy = X.copy()
        h = xCopy[..., 0]

        # Because we divide by h, we need to shift slightly away from h=0. Shifting by 1e-5 and 1e-8 show no difference,
        # while by 1e-3 shows a 1e-12 relative error and by 1e-1 a 1e-8 relative error.
        hShift = 1e-6 * self.fieldScale

        if len(h.shape) > 0:
            for i in range(len(h)):
                if (abs(xCopy[..., 0][i]) < hShift).any():
                    hSign = np.sign(xCopy[..., 0][i])
                    if len(hSign.shape) > 1:
                        hSign[hSign == 0] = 1
                    elif hSign == 0:
                        hSign = 1
                    xCopy[..., 0][i] = hShift*hSign
                    h[i] = xCopy[..., 0][i]
        else:
            if abs(h) < hShift:
                hSign = np.sign(h)
                h = hShift*(1 if hSign == 0 else hSign)
                xCopy[0] = h

        if self.bUseGoldstoneResummation:
            d3hjk_VCW = self.d3jkl_VCW(xCopy, 0, j, k, T=0, massMatrix=massMatrix, ignoreGoldstone=True)
        else:
            d3hjk_VCW = 0.0

        returnVal = self.d3jkl_V0m(xCopy, 0, j, k) + d3hjk_VCW

        if j == 0:
            returnVal -= self.dj_goldstoneMass(xCopy, j, massMatrix=massMatrix)

        if k == 0:
            returnVal -= self.dj_goldstoneMass(xCopy, k, massMatrix=massMatrix)

        return returnVal / h

    """
        ----------------------------------------------------------------------------------------------------------------

        Boson masses and their derivatives.

        ----------------------------------------------------------------------------------------------------------------
    """

    def boson_massSq(self, X, T, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        X = np.array(X)
        h = X[..., 0]
        #s = X[..., 1]
        h2 = h**2
        #s2 = s**2
        T2 = T**2

        # Thermal corrections.
        mW_therm = 11/6*self.g2*T2
        # The thermal corrections for mass eigenvalues, Z boson and photon are handled in their respective
        # eigenvalue equations.

        # Neutral gauge mass matrix eigenvalues (for Z boson and photon masses).
        a = (self.g2 + self.gp2)*(3*h2 + 22*T2)
        b = np.sqrt(9*(self.g2 + self.gp2)**2*h2**2 + 44*T2*(self.g2 - self.gp2)**2*(3*h2 + 11*T2))
        mZ = (a + b)/24
        mPh = (a - b)/24

        # Scalar mass eigenvalues (for Higgs and singlet masses).
        m1, m2 = self.massEigenvalues(X, T, massMatrix)

        mW = self.g2*h2/4 + mW_therm

        if ignoreGoldstone:
            mgb = 0.0
        else:
            # TODO: Given that we now use the massMatrix in both the eigs and the Goldstone self-energy we should really
            #  calculate it if it's None.
            mgb = self.goldstoneMass(X, T, massMatrix=massMatrix)

        # Important to use mp or another mass that has a thermal correction so that massSq gets the right shape (in the
        # the event X is a single point and T is an array).
        massSq = np.empty(mZ.shape + (Nboson,))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        # 3 dof for photon.
        dof = np.array([3, 6, 3, 3, *[1]*self.Ndim])
        c = np.array([1.5, 5/6, 5/6, 5/6, *[1.5]*self.Ndim])

        return massSq, dof, c

    def d_boson_massSq_dh(self, X, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        X = np.array(X)
        h = X[..., 0]

        mW = self.g2*h/2
        mZ = (self.g2 + self.gp2)*h/2
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            mgb = self.dj_goldstoneMass(X, 0, massMatrix=massMatrix)

        m1, m2 = self.d_massEigenvalues_dh(X, massMatrix)

        massSq = np.empty(mZ.shape + (Nboson,))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    def d_boson_massSq_ds(self, X, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        X = np.array(X)
        #s = X[..., 1]

        mW = 0
        mZ = 0
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            mgb = self.dj_goldstoneMass(X, 1, massMatrix=massMatrix)

        m1, m2 = self.d_massEigenvalues_ds(X, massMatrix)

        massSq = np.empty(m1.shape + (Nboson,))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    def d2_boson_massSq_dh2(self, X, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        mW = self.g2/2
        mZ = (self.g2 + self.gp2)/2
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            mgb = self.d2jk_goldstoneMass(X, 0, 0, massMatrix=massMatrix)

        m1, m2 = self.d2_massEigenvalues_dh2(X, massMatrix)

        massSq = np.empty(shape=(m1.shape + (Nboson,)))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    def d2_boson_massSq_ds2(self, X, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        mW = 0
        mZ = 0
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            mgb = self.d2jk_goldstoneMass(X, 1, 1, massMatrix=massMatrix)

        m1, m2 = self.d2_massEigenvalues_ds2(X, massMatrix)

        massSq = np.empty(shape=(m1.shape + (Nboson,)))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    def d2_boson_massSq_dhds(self, X, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        mW = 0
        mZ = 0
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            mgb = self.d2jk_goldstoneMass(X, 0, 1, massMatrix=massMatrix)

        m1, m2 = self.d2_massEigenvalues_dhds(X, massMatrix)

        massSq = np.empty(shape=(m1.shape + (Nboson,)))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    # In this model, only the scalars have finite third derivatives.
    def d3jkl_boson_massSq(self, X, j, k, l, massMatrix=None, ignoreGoldstone=False):
        Nboson = 4 + self.Ndim

        mW = 0
        mZ = 0
        mPh = 0

        if ignoreGoldstone:
            mgb = 0.0
        else:
            print('<d3jkl_boson_massSq> Attempted to calculate third derivative of Goldstone boson mass. We should'
                'ignore the Goldstone boson contribution when taking the third derivative of the Coleman-Weinberg'
                'potential! Ignoring this contribution...')
            mgb = 0.0

        m1, m2 = self.d3jkl_massEigenvalues(X, j, k, l, massMatrix=massMatrix)

        massSq = np.empty(shape=(m1.shape + (Nboson,)))

        massSq[..., 0] = mgb
        massSq[..., 1] = mW
        massSq[..., 2] = mZ
        massSq[..., 3] = mPh
        massSq[..., 4] = m1
        massSq[..., 5] = m2

        return massSq

    """
        ----------------------------------------------------------------------------------------------------------------

        Fermion masses and their derivatives.

        ----------------------------------------------------------------------------------------------------------------
    """

    def fermion_massSq(self, X):
        Nfermions = 1

        X = np.array(X)
        h = X[..., 0]
        h2 = h**2

        # Top quark.
        mt = self.yt2*h2/2

        massSq = np.empty(np.asarray(mt).shape + (Nfermions,))

        massSq[..., 0] = mt

        dof = np.array([12])

        return massSq, dof

    def d_fermion_massSq_dh(self, X):
        Nfermions = 1

        X = np.array(X)
        h = X[..., 0]

        # Top quark.
        mt = self.yt2*h

        massSq = np.empty(np.asarray(mt).shape + (Nfermions,))

        massSq[..., 0] = mt

        return massSq

    def d_fermion_massSq_ds(self, X):
        Nfermions = 1

        X = np.array(X)

        massSq = np.empty(X[..., 0].shape + (Nfermions,))

        massSq[..., 0] = 0

        return massSq

    def d2_fermion_massSq_dh2(self, X):
        Nfermions = 1

        # Top quark.
        mt = self.yt2

        massSq = np.empty(shape=Nfermions)

        massSq[..., 0] = mt

        return massSq

    def d2_fermion_massSq_ds2(self, X):
        Nfermions = 1

        X = np.array(X)

        massSq = np.empty(X[..., 0].shape + (Nfermions,))

        massSq[..., 0] = 0

        return massSq

    def d2_fermion_massSq_dhds(self, X):
        Nfermions = 1

        X = np.array(X)

        massSq = np.empty(X[..., 0].shape + (Nfermions,))

        massSq[..., 0] = 0

        return massSq

    # In this model, no fermions have a finite third derivative.
    def d3jkl_fermion_massSq(self, X, j, k, l):
        Nfermions = 1

        X = np.array(X)

        massSq = np.empty(X[..., 0].shape + (Nfermions,))

        massSq[..., 0] = 0

        return massSq

    """
        ----------------------------------------------------------------------------------------------------------------

        The algorithm for constraining the parameters. We constrain the quadratic parameters muh and mus (the
        coefficients of h^2 and s^2) and the quartic parameters lh and ls (the coefficients of h^4 and s^4).
        
        We first constrain the parameters at tree-level, neglecting the one-loop corrections to the potential.
        We then constrain the parameters at one-loop, using the tree-level constrained parameter values. This shifts the
        parameter values. The one-loop constraints are applied again on these corrected parameters until convergence.

        ----------------------------------------------------------------------------------------------------------------
    """

    def constrainParameters(self, bDebugIteration=False, bStoreParameterEvolution=False, bSkipTreeLevel=False, bSkipOneLoop=False):
        if not bSkipTreeLevel:
            self.muh, self.mus, self.lh, self.ls, self.c2, self.muh0, self.mus0 = 0, 0, 0, 0, 0, 0, 0

            self.constrainParametersAtTreeLevel()

        vev = np.array([self.vh, self.vs])

        self.muhPrev, self.musPrev, self.lhPrev, self.lsPrev, self.c2Prev, self.muh0Prev, self.mus0Prev =\
            self.muh, self.mus, self.lh, self.ls, self.c2, self.muh0, self.mus0
        self.thetaPrev = self.calculateMixingAngle_tree_tan(vev, 0)

        if bStoreParameterEvolution:
            self.parameterEvolution.append([self.muh, self.mus, self.lh, self.ls, self.c2, self.thetaPrev])

        if bSkipOneLoop:
            return

        if self.thetaPrev is None:
            if bDebugIteration:
                print('Tree-level mixing angle is invalid at tree-level.')
            self.bValid = False
            return

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
                print(str(i) + '. ' + str(self.muh) + '\t' + str(self.mus) + '\t' + str(self.lh) + '\t' + str(self.ls)
                    + '\t' + str(self.c2))
                print('Higgs mass:', np.sqrt(abs(self.HiggsMass(vev))))
                print('Singlet mass:', np.sqrt(abs(self.SingletMass(vev))))
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
                print(str(abs(self.muh - self.muhPrev)) + '\t' + str(abs(self.mus - self.musPrev))
                    + '\t' + str(abs(self.lh - self.lhPrev)) + '\t' + str(abs(self.ls - self.lsPrev)))
            hasConverged = self.checkConvergence()

            self.muhPrev, self.musPrev, self.lhPrev, self.lsPrev, self.c2Prev, self.muh0, self.mus0 =\
                self.muh, self.mus, self.lh, self.ls, self.c2, self.muh0, self.mus0
            self.thetaPrev = self.calculateMixingAngle_tan(vev)

            if self.thetaPrev is None:
                if bDebugIteration:
                    print('Tree-level mixing angle is invalid at one-loop after iteration:', i)
                self.bValid = False
                return

            if bStoreParameterEvolution:
                self.parameterEvolution.append([self.muh, self.mus, self.lh, self.ls, self.c2, self.thetaPrev])

        if bDebugIteration:
            print("Converged after", i, "iterations")
            print(str(i) + '. ' + str(self.muh) + '\t' + str(self.mus) + '\t' + str(self.lh) + '\t' + str(self.ls)
                + '\t' + str(self.c2))
            print('Higgs mass:', np.sqrt(abs(self.HiggsMass(vev))))
            print('Singlet mass:', np.sqrt(abs(self.SingletMass(vev))))

    def checkParameterMagnitudes(self):
        quadraticMaxVal = 1e10
        quarticMaxVal = 100

        if abs(self.muh) > quadraticMaxVal or abs(self.mus) > quadraticMaxVal or abs(self.lh) > quarticMaxVal \
                or abs(self.ls) > quarticMaxVal or abs(self.c2) > quarticMaxVal:
            # print('Diverging parameters, potential not valid.')
            return False

        return True

    def rollbackIteration(self):
        self.muh, self.mus, self.lh, self.ls, self.c2, self.muh0, self.mus0, self.theta =\
            self.muhPrev, self.musPrev, self.lhPrev, self.lsPrev, self.c2Prev, self.muh0Prev, self.mus0Prev, self.thetaPrev

    def checkConvergence(self):
        tol = 1e-5
        return abs(self.muh - self.muhPrev) < tol and abs(self.mus - self.musPrev) < tol \
            and abs(self.lh - self.lhPrev) < tol and abs(self.ls - self.lsPrev) < tol \
            and abs(self.c2 - self.c2Prev) < tol

    # Updates muh, mus, lh, ls and c2 to satisfy tree-level constraints.
    def constrainParametersAtTreeLevel(self):
        self.lh = 1/(8*self.vh**2) * (self.mh2*self.cosSqTheta + self.ms2*self.sinSqTheta)
        self.ls = 1/(8*self.vs**2) * (self.mh2*self.sinSqTheta + self.ms2*self.cosSqTheta + self.c1*self.vh**2/self.vs
            - 3*self.b3*self.vs)
        self.c2 = 1/(8*self.vh*self.vs) * ((self.mh2 - self.ms2)*self.sin2Theta - 4*self.c1*self.vh)
        self.muh = -2*self.lh*self.vh**2 - self.c1*self.vs - self.c2*self.vs**2
        self.mus = -1/(2*self.vs) * (4*self.ls*self.vs**3 + self.c1*self.vh**2 + 2*self.c2*self.vh**2*self.vs
            + 3*self.b3*self.vs**2)

        # Store the quadratic parameters calculated from the quadratic parameters using tree-level EWSB. These are used
        # for masses that enter the Coleman-Weinberg potential.
        self.muh0 = self.muh
        self.mus0 = self.mus

    # Updates muh, mus, lh, ls and c2 to satisfy one-loop constraints.
    def constrainParametersAtOneLoop(self):
        vev = np.array([self.vh, self.vs])

        mhh = self.d2V0mdh2(vev)
        mss = self.d2V0mds2(vev)
        mhs = self.d2V0mdhds(vev)

        theta = self.calculateMixingAngle_tan_supplied(mhh, mhs, mss)

        massMatrix = MassMatrixData(mhh, mhs, mss, theta)

        dVCWdh = self.dVCWdh(vev, 0, massMatrix)
        dVCWds = self.dVCWds(vev, 0, massMatrix)
        d2VCWdh2 = self.d2VCWdh2(vev, 0, massMatrix)
        d2VCWds2 = self.d2VCWds2(vev, 0, massMatrix)
        d2VCWdhds = self.d2VCWdhds(vev, 0, massMatrix)

        self.lh = 1/(8*self.vh**2) * (self.mh2*self.cosSqTheta + self.ms2*self.sinSqTheta + dVCWdh/self.vh - d2VCWdh2)
        self.ls = 1/(8*self.vs**2) * (self.mh2*self.sinSqTheta + self.ms2*self.cosSqTheta + self.c1*self.vh**2/self.vs
                    - 3*self.b3*self.vs + dVCWds/self.vs - d2VCWds2)
        self.c2 = 1/(8*self.vh*self.vs) * ((self.mh2 - self.ms2)*self.sin2Theta - 4*self.c1*self.vh - 2*d2VCWdhds)
        self.muh = -2*self.lh*self.vh**2 - self.c1*self.vs - self.c2*self.vs**2 - 0.5/self.vh*dVCWdh
        self.mus = -1/(2*self.vs) * (4*self.ls*self.vs**3 + self.c1*self.vh**2 + 2*self.c2*self.vh**2*self.vs
                    + 3*self.b3*self.vs**2 + dVCWds)

        # Store the quadratic parameters calculated from the quadratic parameters using tree-level EWSB. These are used
        # for masses that enter the Coleman-Weinberg potential.
        self.muh0 = -2*self.lh*self.vh**2 - self.c1*self.vs - self.c2*self.vs**2
        self.mus0 = -1/(2*self.vs) * (4*self.ls*self.vs**3 + self.c1*self.vh**2 + 2*self.c2*self.vh**2*self.vs
                    + 3*self.b3*self.vs**2)

    def calculateQuadraticsForMasses(self):
        self.muh0 = -2*self.lh*self.vh**2 - self.c1*self.vs - self.c2*self.vs**2
        self.mus0 = -1/(2*self.vs) * (4*self.ls*self.vs**3 + self.c1*self.vh**2 + 2*self.c2*self.vh**2*self.vs
                    + 3*self.b3*self.vs**2)

    """
    ----------------------------------------------------------------------------------------------------------------
    
    Derivatives of the Coleman-Weinberg potential up to second order.
    
    ----------------------------------------------------------------------------------------------------------------
    """

    def dVCWdh(self, X, T=0, massMatrix=None, ignoreGoldstone=False):
        m, n, c = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        dmdh = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)

        y = 0

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                continue
            else:
                y += n[i]*m[i]*dmdh[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)

        m, n = self.fermion_massSq(X)
        dmdh = self.d_fermion_massSq_dh(X)
        c = 1.5

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                continue
            else:
                y -= n[i]*m[i]*dmdh[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)

        return y/(32*np.pi**2)

    def dVCWds(self, X, T=0, massMatrix=None, ignoreGoldstone=False):
        m, n, c = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        dmds = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)

        y = 0

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                continue
            else:
                y += n[i]*m[i]*dmds[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)

        m, n = self.fermion_massSq(X)
        dmds = self.d_fermion_massSq_ds(X)
        c = 1.5

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                continue
            else:
                y -= n[i]*m[i]*dmds[i]*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)

        return y/(32*np.pi**2)

    def d2VCWdh2(self, X, T=0, massMatrix=None, ignoreGoldstone=False):
        m, n, c = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        dmdh = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
        d2mdh2 = self.d2_boson_massSq_dh2(X, massMatrix, ignoreGoldstone)

        y = 0

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y += n[i]*dmdh[i]**2*(np.log(self.massShiftHack) - c[i] + 1.5)
            else:
                y += n[i]*((dmdh[i]**2 + m[i]*d2mdh2[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)
                    + dmdh[i]**2)

        m, n = self.fermion_massSq(X)
        dmdh = self.d_fermion_massSq_dh(X)
        d2mdh2 = self.d2_fermion_massSq_dh2(X)
        c = 1.5

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y -= n[i]*dmdh[i]**2*(np.log(self.massShiftHack) - c + 1.5)
            else:
                y -= n[i]*((dmdh[i]**2 + m[i]*d2mdh2[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)
                    + dmdh[i]**2)

        return y/(32*np.pi**2)

    def d2VCWds2(self, X, T=0, massMatrix=None, ignoreGoldstone=False):
        m, n, c = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        dmds = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
        d2mds2 = self.d2_boson_massSq_ds2(X, massMatrix, ignoreGoldstone)

        y = 0

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y += n[i]*dmds[i]**2*(np.log(self.massShiftHack) - c[i] + 1.5)
            else:
                y += n[i]*((dmds[i]**2 + m[i]*d2mds2[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)
                    + dmds[i]**2)

        m, n = self.fermion_massSq(X)
        dmds = self.d_fermion_massSq_ds(X)
        d2mds2 = self.d2_fermion_massSq_ds2(X)
        c = 1.5

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y -= n[i]*dmds[i]**2*(np.log(self.massShiftHack) - c + 1.5)
            else:
                y -= n[i]*((dmds[i]**2 + m[i]*d2mds2[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)
                    + dmds[i]**2)

        return y/(32*np.pi**2)

    def d2VCWdhds(self, X, T=0, massMatrix=None, ignoreGoldstone=False):
        m, n, c = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        dmdh = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
        dmds = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
        d2mdhds = self.d2_boson_massSq_dhds(X, massMatrix, ignoreGoldstone)

        y = 0

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y += n[i]*dmdh[i]*dmds[i]*(np.log(self.massShiftHack) - c[i] + 1.5)
            else:
                y += n[i]*((dmdh[i]*dmds[i] + m[i]*d2mdhds[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c[i] + 0.5)
                    + dmdh[i]*dmds[i])

        m, n = self.fermion_massSq(X)
        dmdh = self.d_fermion_massSq_dh(X)
        dmds = self.d_fermion_massSq_ds(X)
        d2mdhds = self.d2_fermion_massSq_dhds(X)
        c = 1.5

        for i in range(len(m)):
            if abs(m[i]) < self.minMassThreshold:
                y -= n[i]*dmdh[i]*dmds[i]*(np.log(self.massShiftHack) - c + 1.5)
            else:
                y -= n[i]*((dmdh[i]*dmds[i] + m[i]*d2mdhds[i])*(np.log(np.abs(m[i]/self.renormScaleSq)) - c + 0.5)
                    + dmdh[i]*dmds[i])

        return y/(32*np.pi**2)

    def dj_VCW(self, X, j, T=0, massMatrix=None, ignoreGoldstone=False):
        mb, nb, cb = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        mf, nf = self.fermion_massSq(X)
        cf = 1.5

        if j == 0:
            dj_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_dh(X)
        else:
            dj_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_ds(X)

        y = np.sum(nb*mb*dj_mb*(np.log(abs(mb/self.renormScaleSq) + 1e-100) - cb + 0.5), axis=-1)

        y -= np.sum(nf*mf*dj_mf*(np.log(abs(mf/self.renormScaleSq) + 1e-100) - cf + 0.5), axis=-1)

        return y/(32*np.pi**2)

    def d2jk_VCW(self, X, j, k, T=0, massMatrix=None, ignoreGoldstone=False):
        mb, nb, cb = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        mf, nf = self.fermion_massSq(X)
        cf = 1.5

        if j == 0:
            dj_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_dh(X)
        else:
            dj_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_ds(X)

        if k == j:
            dk_mb = dj_mb
            dk_mf = dj_mf
        elif j == 0:
            dk_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dk_mf = self.d_fermion_massSq_dh(X)
        else:
            dk_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dk_mf = self.d_fermion_massSq_ds(X)

        total = j+k
        if total > 2:
            print('<d2jk_VCW> Error: Attempted to take derivative in field directions:', j, k)
            return 0.0
        if total == 0:
            d2jk_mb = self.d2_boson_massSq_dh2(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_dh2(X)
        elif total == 1:
            d2jk_mb = self.d2_boson_massSq_dhds(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_dhds(X)
        else:
            d2jk_mb = self.d2_boson_massSq_ds2(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_ds2(X)

        y = 0

        for i in range(len(mb)):
            if abs(mb[i]) < self.minMassThreshold:
                y += nb[i]*dj_mb[i]*dk_mb[i]*(np.log(self.massShiftHack) - cb[i] + 1.5)
            else:
                y += nb[i]*((dj_mb[i]*dk_mb[i] + mb[i]*d2jk_mb[i])*(np.log(np.abs(mb[i]/self.renormScaleSq)) - cb[i]
                    + 0.5) + dj_mb[i]*dk_mb[i])

        for i in range(len(mf)):
            if abs(mf[i]) < self.minMassThreshold:
                y -= nf[i]*dj_mf[i]*dk_mf[i]*(np.log(self.massShiftHack) - cf + 1.5)
            else:
                y -= nf[i]*((dj_mf[i]*dk_mf[i] + mf[i]*d2jk_mf[i])*(np.log(np.abs(mf[i]/self.renormScaleSq)) - cf + 0.5)
                    + dj_mf[i]*dk_mf[i])

        return y/(32*np.pi**2)

    def d3jkl_VCW(self, X, j, k, l, T=0, massMatrix=None, ignoreGoldstone=False):
        mb, nb, cb = self.boson_massSq(X, T, massMatrix, ignoreGoldstone)
        mf, nf = self.fermion_massSq(X)
        cf = 1.5

        if j == 0:
            dj_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_dh(X)
        else:
            dj_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dj_mf = self.d_fermion_massSq_ds(X)

        if k == j:
            dk_mb = dj_mb
            dk_mf = dj_mf
        elif k == 0:
            dk_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dk_mf = self.d_fermion_massSq_dh(X)
        else:
            dk_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dk_mf = self.d_fermion_massSq_ds(X)

        if l == j:
            dl_mb = dj_mb
            dl_mf = dj_mf
        elif l == k:
            dl_mb = dk_mb
            dl_mf = dk_mf
        elif l == 0:
            dl_mb = self.d_boson_massSq_dh(X, massMatrix, ignoreGoldstone)
            dl_mf = self.d_fermion_massSq_dh(X)
        else:
            dl_mb = self.d_boson_massSq_ds(X, massMatrix, ignoreGoldstone)
            dl_mf = self.d_fermion_massSq_ds(X)

        total = j+k+l
        if total > 3:
            print('<d3jkl_VCW> Error: Attempted to take derivative in field directions:', j, k, l)
            return 0.0

        total_jk = j+k
        if total_jk == 0:
            d2jk_mb = self.d2_boson_massSq_dh2(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_dh2(X)
        elif total_jk == 1:
            d2jk_mb = self.d2_boson_massSq_dhds(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_dhds(X)
        else:
            d2jk_mb = self.d2_boson_massSq_ds2(X, massMatrix, ignoreGoldstone)
            d2jk_mf = self.d2_fermion_massSq_ds2(X)

        total_kl = k+l
        if total_kl == total_jk:
            d2kl_mb = d2jk_mb
            d2kl_mf = d2jk_mf
        elif total_kl == 0:
            d2kl_mb = self.d2_boson_massSq_dh2(X, massMatrix, ignoreGoldstone)
            d2kl_mf = self.d2_fermion_massSq_dh2(X)
        elif total_kl == 1:
            d2kl_mb = self.d2_boson_massSq_dhds(X, massMatrix, ignoreGoldstone)
            d2kl_mf = self.d2_fermion_massSq_dhds(X)
        else:
            d2kl_mb = self.d2_boson_massSq_ds2(X, massMatrix, ignoreGoldstone)
            d2kl_mf = self.d2_fermion_massSq_ds2(X)

        total_jl = j+l
        if total_jl == total_jk:
            d2jl_mb = d2jk_mb
            d2jl_mf = d2jk_mf
        elif total_jl == total_kl:
            d2jl_mb = d2kl_mb
            d2jl_mf = d2kl_mf
        elif total_jl == 0:
            d2jl_mb = self.d2_boson_massSq_dh2(X, massMatrix, ignoreGoldstone)
            d2jl_mf = self.d2_fermion_massSq_dh2(X)
        elif total_jl == 1:
            d2jl_mb = self.d2_boson_massSq_dhds(X, massMatrix, ignoreGoldstone)
            d2jl_mf = self.d2_fermion_massSq_dhds(X)
        else:
            d2jl_mb = self.d2_boson_massSq_ds2(X, massMatrix, ignoreGoldstone)
            d2jl_mf = self.d2_fermion_massSq_ds2(X)

        d3jkl_mb = self.d3jkl_boson_massSq(X, j, k, l, massMatrix=massMatrix, ignoreGoldstone=ignoreGoldstone)
        d3jkl_mf = self.d3jkl_fermion_massSq(X, j, k, l)

        y = 0

        for i in range(len(mb)):
            if abs(mb[i]) < self.minMassThreshold:
                y += nb[i]*((d2jl_mb[i]*dk_mb[i] + dj_mb[i]*d2kl_mb[i] + dl_mb[i]*d2jk_mb[i])
                    * (np.log(self.massShiftHack) - cb[i] + 1.5) - mb[i]*d3jkl_mb[i]
                    + (dj_mb[i]*dk_mb[i]*dl_mb[i]) / self.massShiftHack)
            else:
                y += nb[i]*((d2jl_mb[i]*dk_mb[i] + dj_mb[i]*d2kl_mb[i] + dl_mb[i]*d2jk_mb[i] + mb[i]*d3jkl_mb[i])
                    * (np.log(np.abs(mb[i]/self.renormScaleSq)) - cb[i] + 1.5) - mb[i]*d3jkl_mb[i]
                    + (dj_mb[i]*dk_mb[i]*dl_mb[i]) / mb[i])

        for i in range(len(mf)):
            if abs(mf[i]) < self.minMassThreshold:
                y -= nf[i]*((d2jl_mf[i]*dk_mf[i] + dj_mf[i]*d2kl_mf[i] + dl_mf[i]*d2jk_mf[i])
                    * (np.log(self.massShiftHack) - cf + 1.5) - mb[i]*d3jkl_mf[i]
                    + (dj_mf[i]*dk_mf[i]*dl_mf[i]) / self.massShiftHack)
            else:
                y -= nf[i]*((d2jl_mf[i]*dk_mf[i] + dj_mf[i]*d2kl_mf[i] + dl_mf[i]*d2jk_mf[i] + mf[i]*d3jkl_mf[i])
                    * (np.log(np.abs(mf[i]/self.renormScaleSq)) - cf + 1.5) - mf[i]*d3jkl_mf[i]
                    + (dj_mf[i]*dk_mf[i]*dl_mf[i]) / mf[i])

        return y/(32*np.pi**2)

    """
        ----------------------------------------------------------------------------------------------------------------

        Derivatives of the tree-level potential up to fourth order using tree-level EWSB to calculate the quadratic
        parameters from the quartic parameters (i.e. muh from lh and mus from ls). The 'subscript' 'm' in V0m signifies
        these forms are for use in masses that enter the Coleman-Weinberg potential, where we want the parameters to
        be related through EWSB at tree-level only.

        ----------------------------------------------------------------------------------------------------------------
    """

    def dV0mdh(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.muh0*h + 4*self.lh*h**3 + 2*self.c1*h*s + 2*self.c2*h*s**2

    def dV0mds(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.mus0*s + 4*self.ls*s**3 + self.c1*h**2 + 2*self.c2*h**2*s + 3*self.b3*s**2

    def d2V0mdh2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.muh0 + 12*self.lh*h**2 + 2*self.c1*s + 2*self.c2*s**2

    def d2V0mds2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.mus0 + 12*self.ls*s**2 + 2*self.c2*h**2 + 6*self.b3*s

    def d2V0mdhds(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.c1*h + 4*self.c2*h*s

    def d3V0mdh3(self, X):
        h = X[..., 0]

        return 24*self.lh*h

    def d3V0mdh2ds(self, X):
        s = X[..., 1]

        return 2*self.c1 + 4*self.c2*s

    def d3V0mdhds2(self, X):
        h = X[..., 0]

        return 4*self.c2*h

    def d3V0mds3(self, X):
        s = X[..., 1]

        return 24*self.ls*s + 6*self.b3

    def d4V0mdh4(self, X):
        return 24*self.lh

    def d4V0mdh3ds(self, X):
        return 0

    def d4V0mdh2ds2(self, X):
        return 4*self.c2

    def d4V0mdhds3(self, X):
        return 0

    def d4V0mds4(self, X):
        return 24*self.ls

    def d2jk_V0m(self, X, j, k):
        total = j+k
        if total == 0:
            return self.d2V0mdh2(X)
        elif total == 1:
            return self.d2V0mdhds(X)
        elif total == 2:
            return self.d2V0mds2(X)
        else:
            print('<d2jk_V0m> Error: Attempted to take derivative in field directions:', j, k)
            return 0.0

    def d3jkl_V0m(self, X, j, k, l):
        total = j+k+l
        if total == 0:
            return self.d3V0mdh3(X)
        elif total == 1:
            return self.d3V0mdh2ds(X)
        elif total == 2:
            return self.d3V0mdhds2(X)
        elif total == 3:
            return self.d3V0mds3(X)
        else:
            print('<d3jkl_V0m> Error: Attempted to take derivative in field directions:', j, k, l)
            return 0.0

    def d4jklm_V0m(self, X, j, k, l, m):
        total = j+k+l
        if total == 0:
            return self.d4V0mdh4(X)
        elif total == 1:
            return self.d4V0mdh3ds(X)
        elif total == 2:
            return self.d4V0mdh2ds2(X)
        elif total == 3:
            return self.d4V0mdhds3(X)
        elif total == 4:
            return self.d4V0mds4(X)
        else:
            print('<d4jklm_V0m> Error: Attempted to take derivative in field directions:', j, k, l, m)
            return 0.0

    """
        ----------------------------------------------------------------------------------------------------------------

        Derivatives of the tree-level potential up to second order using the most recent one-loop constrained
        quadratic parameters muh and mus. These are related to the quartic parameters lh and ls through one-loop EWSB.

        ----------------------------------------------------------------------------------------------------------------
    """

    def dV0dh(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.muh*h + 4*self.lh*h**3 + 2*self.c1*h*s + 2*self.c2*h*s**2

    def dV0ds(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.mus*s + 4*self.ls*s**3 + self.c1*h**2 + 2*self.c2*h**2*s + 3*self.b3*s**2

    def d2V0dh2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.muh + 12*self.lh*h**2 + 2*self.c1*s + 2*self.c2*s**2

    def d2V0ds2(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.mus + 12*self.ls*s**2 + 2*self.c2*h**2 + 6*self.b3*s

    def d2V0dhds(self, X):
        h = X[..., 0]
        s = X[..., 1]

        return 2*self.c1*h + 4*self.c2*h*s

    """
        ----------------------------------------------------------------------------------------------------------------

        Derivatives of the full T=0 one-loop potential up to second order using the most recent one-loop constrained
        quadratic parameters muh and mus. We do not use the derivatives of the full potential in the masses that enter
        the Coleman-Weinberg potential, so we do not have to use muh0 and mus0.

        ----------------------------------------------------------------------------------------------------------------
    """

    def dVdh(self, X, T=0, massMatrix=None):
        return self.dV0dh(X) + self.dVCWdh(X, T, massMatrix)

    def dVds(self, X, T=0, massMatrix=None):
        return self.dV0ds(X) + self.dVCWds(X, T, massMatrix)

    def d2Vdh2(self, X, T=0, massMatrix=None):
        return self.d2V0dh2(X) + self.d2VCWdh2(X, T, massMatrix)

    def d2Vds2(self, X, T=0, massMatrix=None):
        return self.d2V0ds2(X) + self.d2VCWds2(X, T, massMatrix)

    def d2Vdhds(self, X, T=0, massMatrix=None):
        return self.d2V0dhds(X) + self.d2VCWdhds(X, T, massMatrix)

    """
        ----------------------------------------------------------------------------------------------------------------
    
        Derivatives of the full finite temperature one-loop potential up to first order using the most recent one-loop
        constrained quadratic parameters muh and mus. We do not use the derivatives of the full potential in the masses
        that enter the Coleman-Weinberg potential, so we do not have to use muh0 and mus0.
    
        ----------------------------------------------------------------------------------------------------------------
    """

    def dVdh_finiteT(self, X, T, dh):
        Xp = np.add(X, [dh, 0])
        Xm = np.add(X, [-dh, 0])
        return (self.Vtot(Xp, T) - self.Vtot(Xm, T)) / (2*dh)

    def dVds_finiteT(self, X, T, ds):
        Xp = np.add(X, [0, ds])
        Xm = np.add(X, [0, -ds])
        return (self.Vtot(Xp, T) - self.Vtot(Xm, T)) / (2*ds)

    def dVds_finiteT_zeroH(self, s, T):
        point = np.array([0, s])
        dV0ds = self.dV0ds(point)
        dVCWds = self.dVCWds(point, T)
        dVTds = self.dVTds_zeroH_subleading(s, T)

        return dV0ds + dVCWds + dVTds

    def dVds_finiteT_zeroH_approxT(self, s, T):
        ds = 0.1
        Xp = np.array([0, s + ds])
        Xm = np.array([0, s - ds])

        point = np.array([0, s])
        dV0ds = self.dV0ds(point)
        dVCWds = self.dVCWds(point, T)

        T = np.array([T])
        bosonsp = self.boson_massSq(Xp, T)
        bosonsm = self.boson_massSq(Xm, T)
        fermions = [np.array([0]), np.array([0])]
        dVTds = (self.V1T(bosonsp, fermions, T) - self.V1T(bosonsm, fermions, T)) / (2*ds)

        return dV0ds + dVCWds + dVTds

    def dVTds_zeroH(self, s, T):
        if T == 0:
            return 0

        point = np.array([0, s])
        bosons, c, n = self.boson_massSq(point, T)
        dBosonsds = self.d_boson_massSq_ds(point, T)

        return T**2/(24*np.pi)*np.sum(n*dBosonsds*(np.pi - 3/T*np.sign(bosons)*np.sqrt(np.abs(bosons))), axis=-1)

    def dVTds_zeroH_subleading(self, s, T):
        if T == 0:
            return 0

        point = np.array([0, s])
        bosons, c, n = self.boson_massSq(point, T)
        dBosonsds = self.d_boson_massSq_ds(point)
        logab = 1.5 - 2*0.5772156649 + 2*np.log(4*np.pi)

        return T**2/(8*np.pi**2)*np.sum(n*dBosonsds*(np.pi**2/3 - np.pi/T*np.sign(bosons)*np.sqrt(np.abs(bosons))
            - 1/8*bosons/T**2*(np.log(np.abs(bosons)/T**2) - logab - 1)), axis=-1)


# Benchmarking speed of Vtot evaluation.
if __name__ == "__main__":
    _H = np.linspace(0, 300, 100)
    _S = np.linspace(0, 800, 100)
    _T = np.linspace(0, 100, 10)

    _params = np.loadtxt('../output/RSS/RSS_BP5/parameter_point.txt')
    _pot = RealScalarSingletModel(*_params)

    import time
    startTime = time.perf_counter()

    for _h in _H:
        for _s in _S:
            for _t in _T:
                _V = _pot.Vtot(np.array([_h, _s]), _t)

    endTime = time.perf_counter()
    print('Elapsed time:', endTime - startTime)
