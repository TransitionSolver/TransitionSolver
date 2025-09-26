"""
Analyse gravitational wave signals
==================================
"""

from __future__ import annotations

import logging
from typing import Callable, Optional
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from analysis import phase_structure
from analysis.phase_structure import PhaseStructure
from models.analysable_potential import AnalysablePotential
from gws import kappa_nu_model, hydrodynamics


KELVIN_TO_GEV = 8.617e-14
GEV_TO_HZ = 1.519e24
T0 = 2.725 * KELVIN_TO_GEV
N_EFF = 3.046
G0 = 2 + 7 / 11 * N_EFF
S0 = 2 * np.pi**2 / 45 * G0 * T0**3
KM_TO_MPC = 3.241e-20
H_OVER_H0 = 1.0 / (100 * KM_TO_MPC / GEV_TO_HZ)
ZP = 10  # Sound wave peak frequency from simulations
OMEGA_SW = 0.012  # From erratum of https://arxiv.org/abs/1704.05871 TABLE IV.

logger = logging.getLogger(__name__)



class AnalyseIndividualTransition:
    """
    Analyze gravitational wave signals from a single transition
    """

    def __init__(
            self,
            phase_structure: PhaseStructure,
            transition_report: dict,
            potential: AnalysablePotential,
            use_bubble_sep=True,
            vw=None,
            sample_index=None,
            rho_t=None,
            kappa_coll=None,
            kappa_turb=0.05,
            use_sound_shell=True):

        self.use_sound_shell = use_sound_shell
        self.kappa_turb = kappa_turb
        self.kappa_coll = kappa_coll
        self.rho_t = rho_t
        self.sample_index = sample_index
        self._vw = vw
        self.use_bubble_sep = use_bubble_sep

        self.transition_report = transition_report
        self.from_phase = phase_structure.phases[transition_report['falsePhase']]
        self.to_phase = phase_structure.phases[transition_report['truePhase']]
        self.potential = potential

        self.hydro_transition_temp = hydrodynamics.make_hydro_vars(
            self.from_phase,
            self.to_phase,
            self.potential,
            self.transition_temp,
            phase_structure.groundStateEnergyDensity)

        self.hydro_reheat_temp = hydrodynamics.make_hydro_vars(
            self.from_phase,
            self.to_phase,
            self.potential,
            self.reheating_temp,
            phase_structure.groundStateEnergyDensity)

    @property
    def redshift_freq(self):
        """
        a1/a0 = (s0/s1)^(1/3) and convert from GeV to Hz
        """
        return (S0 / self.hydro_reheat_temp.entropyDensityTrue)**(1 / 3) * GEV_TO_HZ

    @property
    def redshift_amp(self):
        """
        (a1/a0)^4 (H0/H1)^2 = (s0/s1)^(4/3) * (H0/H1)^2, and absorb h^2 factor
        """
        return (S0 / self.hydro_reheat_temp.entropyDensityTrue)**(4 / 3) * self.hydro_transition_temp.hubble_constant**2 * H_OVER_H0**2

    @property
    def Pf(self):
        return 0.71 if self.sample_index is None else self.transition_report[
            'Pf'][self.sample_index]

    @property
    def upsilon(self):
        # Assume the rotational modes are negligible
        fluid_velocity = (self.kinetic_energy_fraction /
                          self.hydro_transition_temp.adiabatic_index(self.Pf))**0.5
        tau_sw = self.length_scale / fluid_velocity
        return 1. - (1 + 2. * self.hydro_transition_temp.hubble_constant * tau_sw)**-0.5

    @property
    def transition_temp(self) -> float:
        if self.sample_index is None:
            return self.transition_report['Tp']
        return self.transition_report['TSubsample'][self.sample_index]

    @cached_property
    def reheating_temp(self) -> float:
        if self.sample_index is None:
            return self.transition_report['Treh_p']
        return self.calculate_reheating_temp(
            self.transition_report['TSubsample'][self.sample_index], self.rho_t)

    @property
    def peak_frequency_coll(self):
        if self.peak_amplitude_coll == 0:
            return 0.
        return self.redshift_freq * \
            (0.77 * (8 * np.pi)**(1 / 3) * self.vw /
             (2 * np.pi * self.length_scale))

    @property
    def peak_frequency_sw_bubble_separation(self):
        return 1.58 * self.redshift_freq / self.length_scale * ZP / 10

    @property
    def rb(self):
        """
        @returns Ratio of shell thickness and bubble separation
        """
        return abs(self.vw -
                   self.hydro_transition_temp.soundSpeedFalse) / self.vw

    @property
    def peak_frequency_sw_shell_thickness(self):
        return self.peak_frequency_sw_bubble_separation / self.rb

    @property
    def peak_frequency_turb(self):
        return 3.5 * self.redshift_freq / self.length_scale

    @property
    def peak_amplitude_sw(self) -> float:
        """
        Fit from https://arxiv.org/abs/1704.05871 taking account of erratum
        """
        A = 2.061
        return A * OMEGA_SW * self.redshift_amp * self.kinetic_energy_fraction**2 * self.hydro_transition_temp.hubble_constant * \
            self.length_scale / self.hydro_transition_temp.soundSpeedFalse * self.upsilon

    @property
    def peak_amplitude_sw_sound_shell(self) -> float:
        """
        Based on https://arxiv.org/abs/1909.10040
        """
        mu_f = 4.78 - 6.27 * self.rb + 3.34 * self.rb**2
        f = 3. / mu_f / 2.061
        return f * self.peak_amplitude_sw

    @property
    def peak_amplitude_coll(self) -> float:
        """
        Based on https://arxiv.org/abs/2208.11697
        """
        if self.kappa_coll is None:
            return 0.
        A = 5.13e-2
        return A * self.redshift_amp * (self.hydro_transition_temp.hubble_constant * self.length_scale / (
            (8 * np.pi)**(1 / 3) * self.vw))**2 * self.kinetic_energy_fraction**2

    @property
    def peak_amplitude_turb(self) -> float:
        A = 9.
        return A * self.redshift_amp * self.hydro_transition_temp.hubble_constant * self.length_scale * \
            (self.kappa_turb * self.kinetic_energy_fraction)**(3 / 2) * \
            self._unnormalised_spectral_shape_turb(self.peak_frequency_turb)

    def spectral_shape_sw(self, f: float) -> float:
        x = f / self.peak_frequency_sw_bubble_separation
        return x**3 * (7 / (4 + 3 * x**2))**3.5

    def spectral_shape_sw_double_broken(self, f: float):
        """
        From https://arxiv.org/abs/2209.13551 (Eq. 2.11), originally from https://arxiv.org/abs/1909.10040 (Eq. 5.7)
        """
        b = 1
        m = (9 * self.rb**4 + b) / (self.rb**4 + 1)
        x = f / self.peak_frequency_sw_shell_thickness
        return x**9 * ((1 + self.rb**4) / (self.rb**4 + x**4))**((9 - b) /
                                                                 4) * ((b + 4) / (b + 4 - m + m * x**2))**((b + 4) / 2)
    def _unnormalised_spectral_shape_turb(self, f: float) -> float:
        x = f / self.peak_frequency_turb
        return x**3 / ((1 + x)**(11 / 3) * (1 + 8 * np.pi * f /
                       (self.redshift_freq * self.hydro_transition_temp.hubble_constant)))

    def spectral_shape_turb(self, f: float) -> float:
        return self._unnormalised_spectral_shape_turb(
            f) / self._unnormalised_spectral_shape_turb(self.peak_frequency_turb)

    def spectral_shape_coll(self, f):
        a = 2.41
        b = 2.42
        c = 4.08
        x = f / self.peak_frequency_coll
        # Using normalised spectral shape, so A = 5.13e-2 is moved to the
        # amplitude calculation.
        return (a + b)**c / (b * x**(-a / c) + a * x**(b / c))**c

    def gw_total(self, f):
        if self.kappa_coll is not None:
            return self.gw_coll(f)
        return self.gw_sw(f) + self.gw_turb(f)

    def gw_sw(self, f):
        if self.use_sound_shell:
            return self.peak_amplitude_sw_sound_shell * \
                self.spectral_shape_sw_double_broken(f)
        return self.peak_amplitude_sw * self.spectral_shape_sw(f)

    def gw_turb(self, f):
        return self.peak_amplitude_turb * self.spectral_shape_turb(f)

    def gw_coll(self, f):
        return self.peak_amplitude_coll * self.spectral_shape_coll(f)

    # Copied from
    # transition_analysis.TransitionAnalyer.calculateReheatTemperature.
    def calculate_reheating_temp(
            self, T: float, suppliedRho_t: Optional[Callable[[float], float]]) -> float:
        Tmin = self.transition_report['T'][-1]
        Tc = self.transition_report['Tc']
        Tsep = min(0.001 * (Tc - Tmin), 0.5 * (T - Tmin))
        rhof = hydrodynamics.calculateEnergyDensityAtT_singlePhase(
            self.from_phase, self.to_phase, self.potential, T)

        def objective(t):
            if suppliedRho_t is not None:
                rhot = suppliedRho_t(t)
            else:
                rhot = hydrodynamics.calculateEnergyDensityAtT_singlePhase(
                    self.from_phase, self.to_phase, self.potential, t, forFromPhase=False)
            # Conservation of energy => rhof = rhof*Pf + rhot*Pt which is equivalent to rhof = rhot (evaluated at
            # different temperatures, T and Tt (Treh), respectively).
            return rhot - rhof

        # If the energy density of the true vacuum is never larger than the current energy density of the false vacuum
        # even at Tc, then reheating goes beyond Tc.
        if objective(Tc) < 0:
            # Also, check the energy density of the true vacuum when it first appears. If a solution still doesn't exist
            # here, then just report -1.
            if self.to_phase.T[-1] - 2 * \
                    Tsep > Tc and objective(self.to_phase.T[-1] - 2 * Tsep) < 0:
                return -1

            return scipy.optimize.toms748(objective, T, self.to_phase.T[-1] - 2 * Tsep)
        return scipy.optimize.toms748(objective, T, Tc)

    @property
    def vw(self) -> float:
        """
        Using either Chapman-Jouguet hydrodynamical estimate or value from transition report
        """
        if self._vw is not None:
            return self._vw 
        return self.hydro_transition_temp.cj_velocity

    @cached_property
    def kappa_sw(self) -> float:
        """
        @returns Efficiency of sound waves using the kappa-nu model
        """
        # adjust the vw value to avoid numerical instabilities

        vw = self.vw
        vw = max(vw, 1e-6)
        vw = min(vw, 0.999999)

        if vw != self.vw:
            logger.warning(
                "vw adjusted from %s to %s to avoid numerical instability", self.vw, vw)

        kappa_sw = kappa_nu_model(
            self.hydro_transition_temp.soundSpeedSqTrue,
            self.hydro_transition_temp.alpha,
            vw,
            self._vw is None)

        if kappa_sw > 1:
            raise RuntimeError("kappa_sw > 1: {kappa_sw}")

        if kappa_sw <= 0:
            raise RuntimeError("kappa_sw <= 0: {kappa_sw}")

        return kappa_sw

    @property
    def kinetic_energy_fraction(self) -> float:
        """
        @returns Kinetic energy fraction
        """
        if self.hydro_transition_temp.soundSpeedSqTrue <= 0:
            return 0.

        factor = self.kappa_coll if self.kappa_coll is not None else self.kappa_sw
        K = factor * self.hydro_transition_temp.K

        if K > 1:
            logger.warning('K > 1: %s', K)

        if K < 0:
            raise RuntimeError("K < 0: {K}")

        return K

    @property
    def length_scale(self) -> float:
        """
        @returns Characteristic bubble length scale
        """
        key = 'meanBubbleSeparation' if self.use_bubble_sep else 'meanBubbleRadius'
        scale = self.transition_report[key]
        if self.sample_index is None:
            return scale
        return scale[self.sample_index]

    def report(self, *detectors):
        report = {}
        report['Peak amplitude (sw, sound shell)'] = self.peak_amplitude_sw_sound_shell
        report['Peak frequency (sw, bubble separation)'] = self.peak_frequency_sw_bubble_separation
        report['Peak frequency (sw, shell thickness)'] = self.peak_frequency_sw_shell_thickness
        report['Peak amplitude (turb)'] = self.peak_amplitude_turb
        report['Peak frequency (turb)'] = self.peak_frequency_turb
        report['Peak amplitude (coll)'] = self.peak_amplitude_coll
        report['Peak frequency (coll)'] = self.peak_frequency_coll
        report["SNR"] = {d.label: d.SNR(self.gw_total) for d in detectors}
        return report

    def plot(self, frequencies, detectors=None, ptas=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if detectors is not None:
            for detector in detectors:
                ax.loglog(frequencies, detector(frequencies), label=detector.label)

        if ptas is not None:
            for i, pta in enumerate(ptas):
                pta.plot(ax, color=f"C{i}")

        ax.loglog(frequencies, self.gw_total(frequencies), label="total")
        ax.loglog(frequencies, self.gw_sw(frequencies), label="sw")
        ax.loglog(frequencies, self.gw_turb(frequencies), label="turb")
        ax.loglog(frequencies, self.gw_coll(frequencies), label="coll")
        ax.legend(scatterpoints=1)


def extract_relevant_transitions(report: dict) -> list[dict]:
    """
    @returns All transitions that are part of valid transition paths
    """
    relevant = []

    for path in report['paths']:
        if path['valid']:
            relevant += path['transitions']

    return [t for t in report['transitions'] if t['id'] in relevant]


class GWAnalyser:
    """
    Analyze gravitational wave signals from every transition in cosmological history
    """

    def __init__(
            self,
            potential,
            phase_structure_file,
            phase_history,
            force_relevant=False,
            **kwargs):

        _, phase_structure_ = phase_structure.load_data(phase_structure_file)
        relevant_transitions = phase_history['transitions'] if force_relevant else extract_relevant_transitions(
            phase_history)

        if not relevant_transitions:
            raise RuntimeError(
                'No relevant transition detected in the phase history')

        self.gws = {t['id']: AnalyseIndividualTransition(
            phase_structure_, t, potential, **kwargs) for t in relevant_transitions}

    def report(self, *detectors):
        """
        @returns Data on GW spectrum
        """
        reports = {k: v.report(*detectors) for k, v in self.gws.items()}
        reports["SNR"] = {d.label: d.SNR(self.gw_total) for d in detectors}
        return reports

    def plot(self, frequencies=None, detectors=None, ptas=None, show=False):
        """
        @returns Figure of plot of data on GW spectrum
        """
        if frequencies is None:
            frequencies = np.logspace(-11, 3, 1000)

        n = len(self.gws)
        fig, ax = plt.subplots(n)

        if n <= 1:
            ax = [ax]

        for a, gw in zip(ax, self.gws.values()):
            gw.plot(frequencies, detectors, ptas, ax=a)

        if show:
            plt.show()

        return fig

    def gw_total(self, f):
        return np.sum([g.gw_total(f) for g in self.gws.values()], axis=-1)

    def gw_sw(self, f):
        return np.sum([g.gw_sw(f) for g in self.gws.values()], axis=-1)

    def gw_turb(self, f):
        return np.sum([g.gw_turb(f) for g in self.gws.values()], axis=-1)

    def gw_coll(self, f):
        return np.sum([g.gw_coll(f) for g in self.gws.values()], axis=-1)
