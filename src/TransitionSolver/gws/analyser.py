"""
Analyse gravitational wave signals
==================================
"""

from __future__ import annotations

import json
import logging
from functools import cached_property
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.phase_structure import PhaseStructure
from ..models.analysable_potential import AnalysablePotential
from .giese_kappa import kappa_nu_model
from . import hydrodynamics
from ..phasetracer import read_phase_tracer


KELVIN_TO_GEV = 8.617e-14
GEV_TO_HZ = 1.519e24
T0 = 2.725 * KELVIN_TO_GEV
N_EFF = 3.046
G0 = 2 + 7 / 11 * N_EFF
S0 = 2 * np.pi**2 / 45 * G0 * T0**3
KM_TO_MPC = 3.241e-20
H_OVER_H0 = 1.0 / (100 * KM_TO_MPC / GEV_TO_HZ)
ZP = 10  # Sound wave peak frequency from simulations

logger = logging.getLogger(__name__)

GW_TEMPLATE_CHOICES_FILE = "gw_template_choices.json"


def read_gw_template_choices() -> dict:
    template_file = files("TransitionSolver.settings").joinpath(
        GW_TEMPLATE_CHOICES_FILE
    )

    with template_file.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        *,
        sound_wave_template,
        turbulence_template,
        collision_template,
        kappa_coll,
        kappa_turb,
    ):
        self.sound_wave_template = sound_wave_template
        self.turbulence_template = turbulence_template
        self.collision_template = collision_template
        self.kappa_turb = kappa_turb
        self.kappa_coll = kappa_coll
        self.use_bubble_sep = use_bubble_sep

        self.transition_report = transition_report
        self.from_phase = phase_structure.phases[transition_report["false_phase"]]
        self.to_phase = phase_structure.phases[transition_report["true_phase"]]
        self.potential = potential

        self.hydro_transition_temp = hydrodynamics.make_hydro_vars(
            self.from_phase,
            self.to_phase,
            self.potential,
            self.transition_temp,
            phase_structure.ground_state_energy_density,
        )

        self.hydro_redshift_temp = hydrodynamics.make_hydro_vars(
            self.from_phase,
            self.to_phase,
            self.potential,
            self.redshift_temp,
            phase_structure.ground_state_energy_density,
        )

    @property
    def redshift_freq(self):
        """
        a1/a0 = (s0/s1)^(1/3) and convert from GeV to Hz
        """
        return (S0 / self.hydro_redshift_temp.entropyDensityTrue) ** (1 / 3) * GEV_TO_HZ

    @property
    def redshift_amp(self):
        """
        (a1/a0)^4 (H0/H1)^2 = (s0/s1)^(4/3) * (H0/H1)^2, and absorb h^2 factor
        """
        return (
            (S0 / self.hydro_redshift_temp.entropyDensityTrue) ** (4 / 3)
            * self.hydro_transition_temp.hubble_constant**2
            * H_OVER_H0**2
        )

    @property
    def Pf(self):
        return self.transition_report["perc_threshold_pf"]

    @property
    def upsilon(self):
        # Assume the rotational modes are negligible
        fluid_velocity = (
            self.kinetic_energy_fraction
            / self.hydro_transition_temp.adiabatic_index(self.Pf)
        ) ** 0.5
        tau_sw = self.length_scale / fluid_velocity
        return (
            1.0
            - (1 + 2.0 * self.hydro_transition_temp.hubble_constant * tau_sw) ** -0.5
        )

    @property
    def transition_temp(self) -> float:
        return self.transition_report["T_p"]

    @cached_property
    def redshift_temp(self) -> float:
        return self.transition_report["Treh_p"]

    @property
    def bubble_wall_velocity(self) -> float:
        return self.transition_report["bubble_wall_velocity_p"]

    @property
    def peak_frequency_coll_semi_analytic_2022(self):
        if self.peak_amplitude_coll_semi_analytic_2022 == 0:
            return 0.0
        """
        From https://arxiv.org/pdf/2208.11697
        """    
        A = 0.77
        return self.peak_frequency_semi_analytic_2022_general(A)

    @property
    def peak_frequency_sw_bubble_separation(self):
        """
        From https://arxiv.org/pdf/2308.12943
        """
        return 1.58 * self.redshift_freq / self.length_scale * ZP / 12.37
    
    def peak_frequency_semi_analytic_2022_general(self, A):
        """
        From https://arxiv.org/pdf/2208.11697
        """
        return self.redshift_freq * (
            A
            * (8 * np.pi) ** (1 / 3)
            * self.bubble_wall_velocity
            / (2 * np.pi * self.length_scale)
        )
        
    @property
    def peak_frequency_sw_semi_analytic_2022(self):
        """
        From https://arxiv.org/pdf/2208.11697
        """
        A = 0.66
        return self.peak_frequency_semi_analytic_2022_general(A)

    @property
    def rb(self):
        """
        @returns Ratio of shell thickness and bubble separation
        """
        return (
            abs(self.bubble_wall_velocity - self.hydro_transition_temp.soundSpeedFalse)
            / self.bubble_wall_velocity
        )

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
        OMEGA_SW = 0.012
        return (
            A
            * OMEGA_SW
            * self.redshift_amp
            * self.kinetic_energy_fraction**2
            * self.hydro_transition_temp.hubble_constant
            * self.length_scale
            / self.hydro_transition_temp.soundSpeedFalse
            * self.upsilon
        )

    @property
    def peak_amplitude_sw_sound_shell(self) -> float:
        """
        Based on https://arxiv.org/abs/1909.10040
        """
        mu_f = 4.78 - 6.27 * self.rb + 3.34 * self.rb**2
        f = 3.0 / mu_f / 2.061
        return f * self.peak_amplitude_sw
    
    @property
    def peak_amplitude_coll_semi_analytic_2022(self) -> float:
        """
        Based on https://arxiv.org/abs/2208.11697
        """
        if self.collision_template is None:
            return 0.0

        if self.kappa_coll is None:
            raise ValueError(
                "`kappa_coll` must be set when `collision_template` is not None."
            )

        A = 5.13e-2
        return (
            A
            * self.redshift_amp
            * (
                self.hydro_transition_temp.hubble_constant
                * self.length_scale
                / ((8 * np.pi) ** (1 / 3) * self.bubble_wall_velocity)
            )
            ** 2
            * self.scalar_field_energy_fraction**2
        )
    
    @property
    def peak_amplitude_sw_semi_analytic_2022(self) -> float:
        """
        Based on https://arxiv.org/abs/2208.11697
        """
        K = self.kappa_sw * self.hydro_transition_temp.available_energy_fraction
        A = 5.14e-2
        return (
            A
            * self.redshift_amp
            * (
                self.hydro_transition_temp.hubble_constant
                * self.length_scale
                / ((8 * np.pi) ** (1 / 3) * self.bubble_wall_velocity)
            )
            ** 2
            * K**2
        )
    
    @property
    def peak_amplitude_turb(self) -> float:
        A = 9.0
        return (
            A
            * self.redshift_amp
            * self.hydro_transition_temp.hubble_constant
            * self.length_scale
            * (self.kappa_turb * self.kinetic_energy_fraction) ** (3 / 2)
            * self._unnormalised_spectral_shape_turb(self.peak_frequency_turb)
        )

    def spectral_shape_sw(self, f: float) -> float:
        x = f / self.peak_frequency_sw_bubble_separation
        return x**3 * (7 / (4 + 3 * x**2)) ** 3.5

    def spectral_shape_sw_sound_shell(self, f: float, k3=False):
        """
        From https://arxiv.org/abs/2209.13551 Eq. 2.11. Originally from https://arxiv.org/abs/1909.10040 Eq. 5.7
        """
        b = 1
        m = (9 * self.rb**4 + b) / (self.rb**4 + 1)
        x = f / self.peak_frequency_sw_shell_thickness
        
        #IR power = 3.Modified according to https://arxiv.org/pdf/2308.12943
        if k3:
            return (
                x**3
                * ((1 + self.rb**4) / (self.rb**4 + x**4)) ** ((3 - b) / 4)
                * ((b + 4) / (b + 4 - m + m * x**2)) ** ((b + 4) / 2)
            )
        
        #IR power = 9
        return (
            x**9
            * ((1 + self.rb**4) / (self.rb**4 + x**4)) ** ((9 - b) / 4)
            * ((b + 4) / (b + 4 - m + m * x**2)) ** ((b + 4) / 2)
        )
        
        
    
    def spectral_shape_sw_semi_analytic_2022(self, f):
        """
        Based on https://arxiv.org/abs/2208.11697
        """
        a = 2.36
        b = 2.36
        c = 3.69
        x = f / self.peak_frequency_sw_semi_analytic_2022
        return self.spectral_shape_semi_analytic_2022_general(f, x, a, b, c)
    
    def _unnormalised_spectral_shape_turb(self, f: float) -> float:
        x = f / self.peak_frequency_turb
        return x**3 / (
            (1 + x) ** (11 / 3)
            * (
                1
                + 8
                * np.pi
                * f
                / (self.redshift_freq * self.hydro_transition_temp.hubble_constant)
            )
        )

    def spectral_shape_turb(self, f: float) -> float:
        return self._unnormalised_spectral_shape_turb(
            f
        ) / self._unnormalised_spectral_shape_turb(self.peak_frequency_turb)
    
    def spectral_shape_semi_analytic_2022_general(self, f, x, a, b, c):
        return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c
    
    def spectral_shape_coll_semi_analytic_2022(self, f):
        """
        Based on https://arxiv.org/abs/2208.11697
        """
        a = 2.41
        b = 2.42
        c = 4.08
        x = f / self.peak_frequency_coll_semi_analytic_2022
        return self.spectral_shape_semi_analytic_2022_general(f, x, a, b, c)

    def gw_total(self, f):
        return self.gw_sw(f) + self.gw_turb(f) + self.gw_coll(f)

    def gw_sw_sgbp_lattice_2017(self, f):
        return self.peak_amplitude_sw * self.spectral_shape_sw(f)

    def gw_sw_dbpl_sound_shell(self, f):
        return (
            self.peak_amplitude_sw_sound_shell * self.spectral_shape_sw_sound_shell(f)
        )
    def gw_sw_semi_analytic_2022(self, f):
        return self.peak_amplitude_sw_semi_analytic_2022 * self.spectral_shape_sw_semi_analytic_2022(f)
    
    def peak_amplitude_sw_higgsless_2024(self, OMEGA_SW, S, b) -> float:
        """
        Fit from https://arxiv.org/abs/2409.03651 Eq.5.8
        """
  
        _k = self.hydro_transition_temp.alpha / (1 + self.hydro_transition_temp.alpha)
        Ksw = _k * S * self.kappa_sw
        RH = self.hydro_transition_temp.hubble_constant * self.length_scale

        betaTf = self.transition_report.get('betaTf', None)
        betaTf = betaTf / self.hydro_transition_temp.hubble_constant

        dt0 = 11 / betaTf
        fluid_velocity = (self.kinetic_energy_fraction /
                          self.hydro_transition_temp.adiabatic_index(self.Pf))**0.5
        tau_sw = 1 / fluid_velocity
        dtfin = tau_sw 
        A_hyp = special.hyp2f1(2, 1 - 2*b, 2.0 - 2*b, (dt0 + dtfin) / (dt0 - 1))
        B_hyp = special.hyp2f1(2, 1 - 2*b, 2.0 - 2*b, dt0 / (dt0 - 1.0))
        
        # K2int
        factor1 = 1.0 / (1.0 - 2 * b)
        factor2 = (1 + dtfin / dt0) ** (1.0 - 2 * b) * A_hyp - B_hyp
        
        #expanding universe
        K2int = (Ksw**2 * dt0) * factor1 * factor2
        
        h2Omega = 3 * OMEGA_SW * self.redshift_amp * K2int * RH
        return h2Omega
    
    @property
    def peak_frequency_sw_higgsless_2024(self) -> float:
        """
        From https://arxiv.org/abs/2409.03651 eq.4.18
        """
        k2 = 0.45
        return k2 * self.redshift_freq / self.length_scale
    
    def spectral_shape_sw_higgsless_2024(self, f: float, k1, k2, n3) -> float:
        """
        From https://arxiv.org/abs/2409.03651
        """
        f1 = k1 * self.redshift_freq / self.length_scale
        f2 = k2 * self.redshift_freq / self.length_scale
        n1 = 3.0
        n2 = 1.0
        a1 = 3.6
        a2 = 2.4
        S = (f / f1)**n1 * (1 + (f / f1)**a1)**((n2 - n1) / a1) * (1 + (f / f2)**a2)**((n3 - n2) / a2)
        mu = self.safe_trapezoid(S, np.log(f), axis = -1)
        S2 = S / mu # both are normalized to the same arbitrary constant which drops out here
        return S2

    def gw_sw_higgsless_2024(self, f):
        """
        From https://arxiv.org/abs/2409.03651
        """
        OMEGA_SW = 3.11e-2
        S = 0.84
        b = 1.17
        
        k1 = 0.39
        k2 = 0.45
        n3 = -3.0
        return self.peak_amplitude_sw_higgsless_2024(OMEGA_SW, S, b) * self.spectral_shape_sw_higgsless_2024(f, k1, k2, n3)
    
    
    def gw_sw(self, f):
        if self.sound_wave_template is None:
            return np.zeros_like(f, dtype=float)

        sound_wave_functions = {
            "sgbp_lattice_2017": self.gw_sw_sgbp_lattice_2017,
            "dbpl_sound_shell": self.gw_sw_dbpl_sound_shell,
            "semi-analytic_2022": self.gw_sw_semi_analytic_2022,
            "higgsless_2024": self.gw_sw_higgsless_2024
        }

        if self.sound_wave_template not in sound_wave_functions:
            raise ValueError(
                f"Unknown sound-wave template: {self.sound_wave_template}. "
                f"Allowed values are: {sorted(sound_wave_functions)}."
            )

        return sound_wave_functions[self.sound_wave_template](f)

    def gw_turb_analytic_2009(self, f):
        return self.peak_amplitude_turb * self.spectral_shape_turb(f)

    def gw_turb(self, f):
        if self.turbulence_template is None:
            return np.zeros_like(f, dtype=float)

        turbulence_functions = {
            "analytic_2009": self.gw_turb_analytic_2009,
        }

        if self.turbulence_template not in turbulence_functions:
            raise ValueError(
                f"Unknown turbulence template: {self.turbulence_template}. "
                f"Allowed values are: {sorted(turbulence_functions)}."
            )

        return turbulence_functions[self.turbulence_template](f)

    def gw_coll_semi_analytic_2022(self, f):
        return self.peak_amplitude_coll_semi_analytic_2022 * self.spectral_shape_coll_semi_analytic_2022(f)

    def gw_coll(self, f):
        if self.collision_template is None:
            return np.zeros_like(f, dtype=float)

        collision_functions = {
            "semi-analytic_2022": self.gw_coll_semi_analytic_2022,
        }

        if self.collision_template not in collision_functions:
            raise ValueError(
                f"Unknown collision template: {self.collision_template}. "
                f"Allowed values are: {sorted(collision_functions)}."
            )

        return collision_functions[self.collision_template](f)

    @cached_property
    def kappa_sw(self) -> float:
        """
        @returns Efficiency of sound waves using the kappa-nu model
        """
        # adjust the bubble wall velocity value to avoid numerical instabilities

        bubble_wall_velocity = self.bubble_wall_velocity
        bubble_wall_velocity = max(bubble_wall_velocity, 1e-6)
        bubble_wall_velocity = min(bubble_wall_velocity, 0.999999)

        if bubble_wall_velocity != self.bubble_wall_velocity:
            logger.warning(
                "bubble wall velocity adjusted from %s to %s to avoid numerical instability",
                self.bubble_wall_velocity,
                bubble_wall_velocity,
            )

        kappa_sw = kappa_nu_model(
            self.hydro_transition_temp.soundSpeedSqTrue,
            self.hydro_transition_temp.alpha,
            bubble_wall_velocity,
            self.transition_report["use_cj_velocity"],
        )

        if kappa_sw > 1:
            raise RuntimeError(f"kappa_sw > 1: {kappa_sw}")

        if kappa_sw <= 0:
            raise RuntimeError(f"kappa_sw <= 0: {kappa_sw}")

        return kappa_sw

    @property
    def kinetic_energy_fraction(self) -> float:
        """
        @returns Kinetic energy fraction
        """
        if self.hydro_transition_temp.soundSpeedSqTrue <= 0:
            return 0.0

        K = self.kappa_sw * self.hydro_transition_temp.available_energy_fraction

        if K > 1:
            logger.warning("K > 1: %s", K)

        if K < 0:
            raise RuntimeError("K < 0: {K}")

        return K

    @property
    def scalar_field_energy_fraction(self) -> float:
        """
        @returns energy fraction in the scalar-field / bubble walls which
        sources GWs from bubble collisions
        """
        if self.collision_template is None:
            return 0.0

        if self.kappa_coll is None:
            raise ValueError(
                "`kappa_coll` must be set when `collision_template` is not None."
            )

        K = self.kappa_coll * self.hydro_transition_temp.available_energy_fraction

        if K > 1:
            logger.warning("K > 1: %s", K)

        if K < 0:
            raise RuntimeError("K < 0: {K}")

        return K

    @property
    def length_scale(self) -> float:
        """
        @returns Characteristic bubble length scale
        """
        key = "bubble_separation_p" if self.use_bubble_sep else "bubble_radius_p"
        return self.transition_report[key]

    def report(self, *detectors):
        report = {}

        if self.sound_wave_template is None:
            report["Peak amplitude (sound waves)"] = 0.0
            report["Peak frequency (sound waves)"] = 0.0
        elif self.sound_wave_template == "sgbp_lattice_2017":
            report["Peak amplitude (sound waves)"] = self.peak_amplitude_sw
            report["Peak frequency (sound waves)"] = (
                self.peak_frequency_sw_bubble_separation
            )
        elif self.sound_wave_template == "dbpl_sound_shell":
            report["Peak amplitude (sound waves)"] = self.peak_amplitude_sw_sound_shell
            report["Peak frequency (sound waves)"] = (
                self.peak_frequency_sw_shell_thickness
            )
        elif self.sound_wave_template == "semi-analytic_2022":
            report['Peak amplitude (sound waves)'] = self.peak_amplitude_sw_semi_analytic_2022
            report['Peak frequency (sound waves)'] = (
                self.peak_frequency_sw_semi_analytic_2022
            )
        elif self.sound_wave_template == "higgsless_2024":
            report['Peak amplitude (sound waves)'] = self.peak_amplitude_sw_higgsless_2024
            report['Peak frequency (sound waves)'] = (
                self.peak_frequency_sw_higgsless_2024
            )           
        else:
            raise ValueError(
                f"Unknown sound-wave template: {self.sound_wave_template}."
            )

        if self.turbulence_template is None:
            report["Peak amplitude (turbulence)"] = 0.0
            report["Peak frequency (turbulence)"] = 0.0
        else:
            report["Peak amplitude (turbulence)"] = self.peak_amplitude_turb
            report["Peak frequency (turbulence)"] = self.peak_frequency_turb

        if self.collision_template is None:
            report["Peak amplitude (collisions)"] = 0.0
            report["Peak frequency (collisions)"] = 0.0
        elif self.collision_template == "semi-analytic_2022":
            report["Peak amplitude (collisions)"] = self.peak_amplitude_coll_semi_analytic_2022
            report["Peak frequency (collisions)"] = self.peak_frequency_coll_semi_analytic_2022

        report["Signal-to-Noise Ratio"] = {
            d.label: d.SNR(self.gw_total) for d in detectors
        }
        report["Bubble wall velocity"] = self.bubble_wall_velocity
        report["Transition temperature"] = self.transition_temp
        report["Redshift temperature"] = self.redshift_temp
        report["Kinetic energy fraction"] = self.kinetic_energy_fraction
        report["Upsilon"] = self.upsilon
        report["Length scale"] = self.length_scale
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
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency (Hz)")


def extract_relevant_transitions(report: dict) -> dict:
    """
    @returns All transitions that are part of valid transition paths
    """
    relevant = []

    for path in report["paths"]:
        if path["valid"]:
            relevant += path["transitions"]

    return {k: report["transitions"][k] for k in relevant}


class GWAnalyser:
    """
    Analyze gravitational wave signals from every transition in cosmological history
    """

    def __init__(
        self,
        potential,
        phase_history,
        phase_structure=None,
        phase_tracer_file=None,
        force_relevant=False,
        **kwargs,
    ):
        if phase_tracer_file is not None:
            phase_structure = read_phase_tracer(phase_tracer_file=phase_tracer_file)

        relevant_transitions = (
            phase_history["transitions"]
            if force_relevant
            else extract_relevant_transitions(phase_history)
        )
        if not relevant_transitions:
            raise RuntimeError("No relevant transition detected in the phase history")

        gw_kwargs = read_gw_template_choices()
        gw_kwargs.update(kwargs)

        self.gws = {
            k: AnalyseIndividualTransition(phase_structure, v, potential, **gw_kwargs)
            for k, v in relevant_transitions.items()
        }

    def report_for_transition_ids(self, transition_ids, *detectors):
        transition_ids = {str(i) for i in transition_ids}

        gws = {k: v for k, v in self.gws.items() if k in transition_ids}
        reports = {k: v.report(*detectors) for k, v in gws.items()}

        if len(gws) > 0:
            reports["Combined"] = {}
            reports["Combined"]["Signal-to-Noise Ratio"] = {
                d.label: d.SNR(
                    lambda f: np.sum([g.gw_total(f) for g in gws.values()], axis=-1)
                )
                for d in detectors
            }

        return reports

    def report(self, *detectors):
        """
        @returns Data on GW spectrum
        """
        reports = {k: v.report(*detectors) for k, v in self.gws.items()}
        reports["Combined"] = {}
        reports["Combined"]["Signal-to-Noise Ratio"] = {
            d.label: d.SNR(self.gw_total) for d in detectors
        }
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
