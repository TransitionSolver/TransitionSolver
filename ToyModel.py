from AnalysablePotential import AnalysablePotential
import numpy as np


# See https://arxiv.org/pdf/1611.05853.pdf or https://arxiv.org/pdf/2212.07559.pdf for details of this model.
class ToyModel(AnalysablePotential):
    def init(self, AonV, v, D, E):
        self.Ndim = 1
        self.v = v
        self.D = D
        self.E = E
        self.setAonV(AonV)

        # Used to determine the overall scale of the problem. This is used (for example) to estimate what is a
        # reasonable step size for a small offset in field space (e.g. for derivatives). Of course, in such an
        # application, a small fraction of this scale is used as the offset.
        self.fieldScale = self.v
        self.temperatureScale = self.v

        # Stop analysing the transition below this temperature.
        self.minimumTemperature = 0.1

        # The number of degrees of freedom in the model. In the Standard Model, this would be 106.75.
        self.ndof = 100.0
        # The number of degrees of freedom that are not included in the one-loop corrections. These need to be accounted
        # for in the free energy density to correctly determine quantities that depend on the energy density (e.g. the
        # Hubble rate). If all degrees of freedom are present in the one-loop corrections, this should be set to zero.
        self.raddof = self.ndof

        # The zero point of the energy density. Typically, this is taken to be the free energy density in the global
        # minimum at zero temperature. If the current state of the Universe corresponds to the global minimum, then this
        # effectively sets the cosmological constant to zero.
        self.groundStateEnergy = self.Vtot([self.v], 0.)

    def setAonV(self, AonV):
        self.AonV = AonV
        # This constraint comes from requiring m_phi = v/2.
        self.l = 0.125 + 1.5*self.AonV
        self.A = AonV*self.v
        self.T0Sq = (self.l - 3*self.AonV) / (2*self.D) * self.v*self.v
        self.rhoV = 0.25*(self.l - 2*self.AonV)*self.v**4

    # The effective potential, containing all field-dependent terms, but perhaps neglecting some temperature-dependent
    # yet field-independent terms.
    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        h = np.asanyarray(X, dtype=float)[..., 0]
        hSq = h*h

        return self.D*(T*T - self.T0Sq)*hSq - (self.E*T + self.A)*hSq*h + 0.25*self.l*hSq*hSq

    # Not used here, but might be useful.
    def getTrueVacuumLocation(self, T):
        return 1.5*(self.E*T + self.A)/self.l * (1. + np.sqrt(1. - 8./9.*self.l*self.D*(T*T - self.T0Sq)
            / ((self.E*T + self.A)*(self.E*T + self.A))))

    # Returns a list of the parameters value that define this potential.
    def getParameterPoint(self):
        return [self.AonV, self.v, self.D, self.E]
