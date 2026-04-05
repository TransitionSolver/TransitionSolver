import numpy as np
from .analysable_potential import AnalysablePotential
from cosmoTransitions.finiteT import Jf, Jb

# Add safe versions for J_b and J_f for calling
# workaround for numpy 2 incompatibility in CT
def _Jb_safe(x):
    xa = np.asarray(x)
    if xa.ndim == 0:
        return Jb(xa[None], approx='spline')[0]
    return Jb(xa, approx='spline')

def _Jf_safe(x):
    xa = np.asarray(x)
    if xa.ndim == 0:
        return Jf(xa[None], approx='spline')[0]
    return Jf(xa, approx='spline')

# Compute x log|x| with correct limit x→0 → 0 (avoid log(0) issues)
def xlogx(x):
    x = np.asarray(x)
    with np.errstate(divide='ignore', invalid='ignore'):# ignore log(0) and invalid ops; will fix values afterward
        y = x * np.log(np.abs(x))
    return np.where(x == 0, 0.0, y)# if x == 0, explicitly set the value to 0
# define BL class
class BminusL(AnalysablePotential):
    def init(self, lps,vp,gbl, lr1,lr2,lr3):
        self.Ndim = 1
        self.vp = vp
        self.lr1 = lr1
        self.lr2 = lr2
        self.lr3 = lr3
        self.lps = lps
        self.gbl = gbl
        self.minimumTemperature = 0.01
        self.Tmax = 2*vp
        self.fieldScale = 2*self.vp
        self.temperatureScale = 2* (self.vp)
        self.ndof = 106
        self.raddof = self.ndof
# setting gs energy via the potential at zero temp
        self.groundStateEnergy = self.Vtot(np.array([self.vp]), 0.)
    def forbidPhaseCrit(self, X):
        return any([np.array([X])[..., 0] < -5.0])
     
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([self.vp])]
    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        h = X[..., 0]
        phi_sq = h * h
        MBL2 = 4.0 * self.gbl * self.gbl * (phi_sq)
        MS2 = 0.5 * self.lps * (phi_sq)
        MR12 = 0.5 * self.lr1 * self.lr1 * phi_sq
        MR22 = 0.5 * self.lr2 * self.lr2 * phi_sq
        MR32 = 0.5 * self.lr3 * self.lr3 * phi_sq
        vphi2 = self.vp**2
        T2 = (T * T) + 1e-100
        T4 = T2 * T2
        IPi2 = 0.10132118364233777144  #= 1. / (M_PI * M_PI);
        I12PiSqrt2 = 0.01875658991993971 # = 1. / (12. * M_PI * std::sqrt(2.));
        twoI3Pi = 0.2122065907891938  # 2. / (3. * M_PI);
        I96 = 0.01041666666666667  # 1. / 96.;
        I12 = 0.08333333333333333 # 1. / 12.;
        sixIPi2 = 0.6079271018540266
        B_CW = sixIPi2 * (self.lps * self.lps * I96 + self.gbl * self.gbl * self.gbl * self.gbl
                           - (self.lr1 * self.lr1 * self.lr1 * self.lr1 + self.lr2 * self.lr2 * self.lr2 * self.lr2 + self.lr3 * self.lr3 * self.lr3 * self.lr3) * I96)

        # VzeroT_raw = 0.25 * B_CW * (phi_sq * phi_sq) * (0.5 * np.log(phi_sq) - 0.5 * np.log(vphi2) - 0.25)
        VzeroT_raw = 0.25 * B_CW * (0.5 * vphi2 * phi_sq * xlogx(phi_sq / vphi2)- 0.25 * phi_sq**2)               
        VzeroT = np.where(phi_sq <= 0, 0.0, VzeroT_raw)
        if T == 0.0:
            return VzeroT
        else:
            yb = (IPi2) * (T4 * Jb_safe(MS2 / (T2), approx='spline')) + 1.5 * (IPi2) * (T4 * Jb_safe(MBL2 / (T2), approx='spline'))
            yf = (IPi2) * (T4 * Jf_safe(MR12 / (T2), approx='spline')) + (IPi2) * (T4 * Jf_safe(MR22 / (T2), approx='spline'))+ (IPi2) * (T4 * Jf_safe(MR32 / (T2), approx='spline'))
            y3 = - I12PiSqrt2* T * (self.lps ** (3 / 2)) * ((phi_sq + I12 * T ** 2) ** (3 / 2) - phi_sq ** (3 / 2)) - twoI3Pi * T * (self.gbl ** 3) * ((phi_sq + T2) ** (3 / 2) - phi_sq ** (3 / 2))
            result = VzeroT+yb+yf+y3
        return result
