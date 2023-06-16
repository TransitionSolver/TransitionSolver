from cosmoTransitions.generic_potential import generic_potential
import numpy as np
from typing import List, Union
from analysis.phase_structure import Phase


class AnalysablePotential(generic_potential):
    # NOTE: ndof and raddof are now basically default values for the number of degrees of freedom. Temperature- and
    # field-dependence of the number of degrees of freedom can be specified in getDegreesOfFreedom.

    # The number of degrees of freedom that are included in one-loop corrections. E.g. If the top quark is the only
    # fermion included in the one-loop corrections, then ndof should not count the degrees of freedom from the other
    # fermions.
    ndof: float = 106.75

    # The number of degrees of freedom that are treated purely as radiation. These are the light degrees of freedom that
    # are omitted from the effective potential. E.g. often only the top quark is considered in the one-loop corrections,
    # so the remaining quark degrees of freedom should be added to raddof. This quantity is used to account for the
    # radiation contribution of these light degrees of freedom, which is important for correctly determining the
    # free energy density. While light degrees of freedom do not significantly affect the phase structure, they can
    # affect the free energy density.
    raddof: float = 0.

    # The free energy density in the ground state of the theory. This should be determined by evaluating the effective
    # potential in the global minimum of the potential at zero temperature. The ground state energy is important for
    # calculating the Hubble parameter, because we normalise the free energy density such that it vanishes in the
    # ground state (thus setting the cosmological constant to zero).
    groundStateEnergy: float = 0.

    # The characteristic scale of fields in the effective potential. For an electroweak phase transition this could be
    # Higgs VEV. This scale is used to determine whether minima of the potential are distinct or have merged. E.g. for
    # a phase transition involving VEVs of the order of 100 GeV, two minima separated in field space by less than 1 GeV
    # may indicate numerical errors or that one minimum is spurious.
    fieldScale: float = 100.

    # The characteristic scale of temperature in the phase history. For an electroweak phase transition this should be
    # of the order of 100 GeV. This scale is used to determine whether two temperatures are similar. E.g. if the scale
    # is 100, then two temperatures T1=1.5 and T2=1.6 are similar. However, if the scale is 1, then T1 and T2 are not
    # similar. Usually the temperature scale is of the same order as the field scale.
    temperatureScale: float = 100.

    # The minimum temperature at which phase transition analysis should be performed. Below this, the phase transition
    # is no longer considered. This can be set to the scale at which other, later, cosmological events take place. E.g.
    # for the electroweak phase transition, the minimum temperature can be set to the scale of the QCD transition,
    # particularly if one wishes the electroweak phase transition to complete before the QCD transition.
    minimumTemperature: float = 0.1

    # Returns the free energy density, which is equal to the effective potential. However, the effective temperature
    # may neglect light degrees of freedom. These must be factored in here, hence the subtraction of radiation terms
    # for each of the raddof neglected degrees of freedom.
    def freeEnergyDensity(self, X: Union[float, list[float], np.ndarray], T: Union[float, list[float], np.ndarray])\
            -> Union[float, np.ndarray]:
        return self.Vtot(X, T, include_radiation=False) - np.pi**2/90*self.getRadiationDegreesOfFreedom(X, T)*T**4

    # Returns a list of values for each parameter that specifies the potential. E.g. if the tree-level potential is
    # V0 = a*phi^2 + b*phi^4, this function should return [a, b].
    def getParameterPoint(self) -> List[float]:
        return []

    def getDegreesOfFreedom(self, X: Union[float, list[float], np.ndarray] = 0., T: Union[float, list[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        return self.ndof

    def getRadiationDegreesOfFreedom(self, X: Union[float, list[float], np.ndarray] = 0., T: Union[float, list[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        return self.ndof

    # This is used in transition analysis and gravitational wave prediction for the energy density. The 'from' phase is
    # typically passed in.
    def getDegreesOfFreedomInPhase(self, phase: Phase, T: Union[float, list[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        # If the degrees of freedom depend on the field configuration, one could use:
        # return self.getDegreesOfFreedom(phase.findPhaseAtT(T, self), T)
        return self.ndof
