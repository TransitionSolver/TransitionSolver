from __future__ import annotations
import traceback

from cosmoTransitions.generic_potential import generic_potential
import numpy as np
from typing import List, Union, Callable
from analysis.phase_structure import Phase
from models.util import geff_handler


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
    # the Higgs VEV. This scale is used to determine whether minima of the potential are distinct or have merged. E.g.
    # for phase transition involving VEVs of the order of 100 GeV, two minima separated in field space by less than 1
    # GeV may indicate numerical errors or that one minimum is spurious.
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

    # If true, ndof will be used for the degrees of freedom used in e.g. redshifting gravitational waves.
    # If false, the degrees of freedom will be calculated from the field- and temperature-dependent mass spectrum.
    bUseSimpleDOF: bool = True

    # Functions for calculating the effective bosonic and fermionic degrees of freedom.
    geffFunc_boson: Callable[[Union[float, np.ndarray]], np.ndarray]
    geffFunc_fermion: Callable[[Union[float, np.ndarray]], np.ndarray]

    def __init__(self, *args, **dargs):
        self.geffFunc_boson = lambda x: np.array(1.)
        self.geffFunc_fermion = lambda x: np.array(0.875)

        super().__init__(*args, **dargs)

        if not self.bUseSimpleDOF:
            self.geffFunc_boson = geff_handler.getGeffCurve_boson()
            self.geffFunc_fermion = geff_handler.getGeffCurve_fermion()

    def V1T(self, bosons, fermions, T, include_radiation=True):
        if T == 0.:
            return 0.
        else:
            return super().V1T(bosons, fermions, T, include_radiation)

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

    # TODO: this only works if the model follows CosmoTransitions' usual potential implementation, where one-loop
    #  corrections are handled by specify masses in the boson_ and fermion_massSq functions. Also, the temperature-
    #  dependence would only be accounted for correctly if the user includes thermal corrections to the masses in these
    #  functions according to the Parwani method.
    def getDegreesOfFreedom(self, X: Union[float, list[float], np.ndarray] = 0., T: Union[float, list[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        # TODO: This should check bUseSimpleDOF to see if we want to skip this entirely. Currently only the
        #  getDegreesOfFreedomInPhase function checks bUseSimpleDOF, and we never use getDegreesOfFreedom in phase
        #  even though we should. The only time we ever use the dof anywhere is in the GW calcs.

        m2b, nb, _ = self.boson_massSq(X, T)
        m2f, nf = self.fermion_massSq(X)

        dof = self.raddof
        #newMethod = True

        #for i in range(m2b.shape[1]):
        for i in range(len(m2b)):
            #if newMethod:
            mask = m2b[..., i] < 1e-10
            y = np.zeros(shape=m2b[..., i].shape)
            y[..., mask] = np.inf
            y[..., ~mask] = T/np.sqrt(np.abs(m2b[..., i][~mask]))
            """else:
                if abs(m2b[i]) < 1e-10:
                    y = np.inf
                else:
                    y = T/np.sqrt(abs(m2b[i]))"""
            factor = self.geffFunc_boson(y)
            dof += nb[i]*factor

        for i in range(len(m2f)):
            #if newMethod:
            mask = m2f[i] < 1e-10
            y = np.zeros(shape=m2f[i].shape)
            y[mask] = np.inf
            y[~mask] = T/np.sqrt(np.abs(m2f[i][~mask]))
            """else:
                if abs(m2f[i]) < 1e-10:
                    y = np.inf
                else:
                    y = T/np.sqrt(abs(m2f[i]))"""
            factor = self.geffFunc_fermion(y)
            dof += nf[i]*factor

        return dof

    # Degrees of freedom not included in the one-loop corrections that should be treated as radiation.
    def getRadiationDegreesOfFreedom(self, X: Union[float, List[float], np.ndarray] = 0., T: Union[float, List[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        return self.raddof

    def getEntropicDegreesOfFreedom(self, X: Union[float, List[float], np.ndarray] = 0., T: Union[float, List[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        # Make the approximation that g_s ~= g_eff for T > 0.1 MeV. See https://arxiv.org/pdf/1609.04979.pdf, Fig 1.
        return self.getDegreesOfFreedom(X, T)

    # This is used in transition analysis and gravitational wave prediction for the energy density. The 'from' phase is
    # typically passed in.
    def getDegreesOfFreedomInPhase(self, phase: Phase, T: Union[float, List[float],
            np.ndarray] = 0.) -> Union[float, np.ndarray]:
        if self.bUseSimpleDOF:
            return self.ndof
        else:
            return self.getDegreesOfFreedom(phase.findPhaseAtT(T, self), T)
