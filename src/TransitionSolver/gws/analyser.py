"""
Analyse gravitational wave signals
==================================
"""

from __future__ import annotations

import json
import logging
import warnings
from functools import cached_property, lru_cache
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, special

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
ZP = 7.7  # Sound wave peak frequency from simulations

logger = logging.getLogger(__name__)

GW_TEMPLATE_CHOICES_FILE = "gw_template_choices.json"


def read_gw_template_choices() -> dict:
    template_file = files("TransitionSolver.settings").joinpath(
        GW_TEMPLATE_CHOICES_FILE
    )

    with template_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def interpolate_transition_report(transition_report: dict, key: str, temperature: float) -> float:
    """Linearly interpolate a transition-history quantity in temperature."""
    temperatures = np.asarray(transition_report['T'], dtype=float)
    values = np.asarray(transition_report[key], dtype=float)
    if not temperatures.min() <= temperature <= temperatures.max():
        raise ValueError(f'Cannot interpolate {key} at T={temperature}: the saved temperature range is [{temperatures.min()}, {temperatures.max()}]')
    order = np.argsort(temperatures)
    return float(np.interp(temperature, temperatures[order], values[order]))

def _temperature_at_false_vacuum_fraction(transition_report: dict, target_pf: float) -> float:
    """Interpolate the temperature at the first downward crossing of target Pf."""
    temperatures = np.asarray(transition_report['T'], dtype=float)
    false_vacuum_fractions = np.asarray(transition_report['Pf'], dtype=float)
    if temperatures.shape != false_vacuum_fractions.shape:
        raise ValueError('Saved T and Pf histories have different lengths')
    if len(temperatures) == 0:
        raise ValueError('Saved T and Pf histories are empty')
    if false_vacuum_fractions[0] == target_pf:
        return float(temperatures[0])
    for i in range(1, len(false_vacuum_fractions)):
        previous_pf = false_vacuum_fractions[i - 1]
        current_pf = false_vacuum_fractions[i]
        if previous_pf >= target_pf >= current_pf:
            if previous_pf == current_pf:
                return float(temperatures[i - 1])
            return float(np.interp(target_pf, [current_pf, previous_pf], [temperatures[i], temperatures[i - 1]]))
    raise ValueError(f'Cannot determine the temperature at Pf={target_pf}: the saved history does not contain a downward crossing of this value')


class AnalyseIndividualTransition:
    """
    Analyze gravitational wave signals from a single transition
    """

    def __init__(self, phase_structure, transition_report, potential,
                 use_bubble_sep=True, *, sound_wave_template, turbulence_template,
                 collision_template, kappa_coll, kappa_turb,
                 source_temperature=None, sound_wave_parameters=None):
        self.sound_wave_template = sound_wave_template
        self.turbulence_template = turbulence_template
        self.collision_template = collision_template
        self.kappa_turb = kappa_turb
        self.kappa_coll = kappa_coll
        self.use_bubble_sep = use_bubble_sep
        self.source_temperature = source_temperature
        self.sound_wave_parameters = dict(sound_wave_parameters or {})
        self.phase_structure = phase_structure
        self.transition_report = transition_report
        self.from_phase = phase_structure.phases[transition_report["false_phase"]]
        self.to_phase = phase_structure.phases[transition_report["true_phase"]]
        self.potential = potential
        self._ground_state_energy_density = phase_structure.ground_state_energy_density
        if not self.at_percolation:
            sampled_min = min(transition_report["T"])
            sampled_max = max(transition_report["T"])
            if not sampled_min <= source_temperature <= sampled_max:
                raise ValueError("Source temperature is outside the saved history")
        self.hydro_transition_temp = hydrodynamics.make_hydro_vars(
            self.from_phase, self.to_phase, potential, self.transition_temp,
            self._ground_state_energy_density)
        self.hydro_redshift_temp = hydrodynamics.make_hydro_vars(
            self.from_phase, self.to_phase, potential, self.redshift_temp,
            self._ground_state_energy_density)

    @cached_property
    def hydro_Tf(self):
        """Calculate completion hydrodynamics on first access."""
        return hydrodynamics.make_hydro_vars(
            self.from_phase,
            self.to_phase,
            self.potential,
            self.completion_temp_Tf,
            self._ground_state_energy_density,
        )

    @property
    def transition_temp(self) -> float:
        if not self.at_percolation:
            return self.source_temperature
        return self.transition_report['T_p']

    @property
    def completion_temp_Tf(self) -> float:
        return self.transition_report["T_f"]

    @cached_property
    def redshift_temp(self) -> float:
        if not self.at_percolation:
            T_min = max(self.from_phase.T[0], self.to_phase.T[0], self.potential.minimum_temperature)
            return hydrodynamics.reheat_temperature(self.from_phase, self.to_phase, self.potential, self.source_temperature, self.transition_report['T_c'], T_min)
        return self.transition_report['Treh_p']

    @property
    def bubble_wall_velocity(self) -> float:
        if not self.at_percolation:
            return interpolate_transition_report(self.transition_report, 'bubble_wall_velocity', self.source_temperature)
        return self.transition_report['bubble_wall_velocity_p']

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
        if not self.at_percolation:
            return interpolate_transition_report(self.transition_report, 'Pf', self.source_temperature)
        return self.transition_report['perc_threshold_pf']

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

    @property
    def length_scale_Tf(self) -> float:
        """
        @returns Characteristic bubble length scale
        """
        key = "bubble_separation_f" if self.use_bubble_sep else "bubble_radius_f"
        return self.transition_report[key]

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

    ####################################################################################
    # Sound waves: single broken power law fit from lattice simulations.               #
    # Paper: https://arxiv.org/abs/1704.05871                                          #
    # Select in gw_template_choices.json with:                                         #
    #     "sound_wave_template": "sgbp_lattice_2017"                                   #
    ####################################################################################

    @property
    def peak_frequency_sw_sgbp_lattice_2017(self):
        return self.redshift_freq / self.length_scale * 10 / (2 * np.pi)

    @property
    def peak_amplitude_sw_sgbp_lattice_2017(self) -> float:
        """
            Fit from https://arxiv.org/abs/1704.05871 taking account of erratum
            """
        OMEGA_SW = 0.012
        A = 2.061
        return A * OMEGA_SW * self.redshift_amp * self.kinetic_energy_fraction ** 2 * self.hydro_transition_temp.hubble_constant * self.length_scale / self.hydro_transition_temp.soundSpeedFalse * self.upsilon

    def spectral_shape_sw_sgbp_lattice_2017(self, f: float) -> float:
        x = f / self.peak_frequency_sw_sgbp_lattice_2017
        return x ** 3 * (7 / (4 + 3 * x ** 2)) ** 3.5

    def gw_sw_sgbp_lattice_2017(self, f):
        return (
            self.peak_amplitude_sw_sgbp_lattice_2017
            * self.spectral_shape_sw_sgbp_lattice_2017(f)
        )

    ####################################################################################
    # Sound waves: double broken power law based on the sound-shell model.             #
    # Papers: https://arxiv.org/abs/1909.10040,                                        #
    # https://arxiv.org/abs/2209.13551 and https://arxiv.org/abs/2308.12943            #
    # Select in gw_template_choices.json with:                                         #
    #     "sound_wave_template": "dbpl_sound_shell"                                    #
    ####################################################################################

    @property
    def rb_sw_dbpl_sound_shell(self):
        """
            @returns Ratio of shell thickness and bubble separation
            """
        v_w = self.bubble_wall_velocity
        c_s = self.hydro_transition_temp.soundSpeedFalse
        rb = abs(v_w - c_s) / v_w
        if not 0 <= rb <= 1:
            raise ValueError(
                "r_b is out of range for the Sound Shell Model template (expected 0 <= r_b <= 1): "
                f"r_b={rb}, v_w={v_w}, c_s={c_s}."
            )
        return rb

    @property
    def peak_frequency_sw_dbpl_sound_shell(self):
        return self.redshift_freq / self.length_scale * ZP / (2 * np.pi)

    @property
    def bubble_separation_break_frequency_sw_dbpl_sound_shell(self):
        """
        Lower break associated with the mean bubble separation. From
        https://arxiv.org/pdf/1909.10040 Eq. 6.1: z_b / z_p is approximately
        the relative sound-shell thickness, r_b.
        """
        return (
            self.rb_sw_dbpl_sound_shell
            * self.peak_frequency_sw_dbpl_sound_shell
        )

    def spectral_shape_sw_dbpl_sound_shell(self, f: float, A=9.0):
        """
            From https://arxiv.org/abs/2209.13551 (Eq. 2.11), originally from https://arxiv.org/abs/1909.10040 (Eq. 5.7)
            """
        b = 1
        m = (A * self.rb_sw_dbpl_sound_shell ** 4 + b) / (self.rb_sw_dbpl_sound_shell ** 4 + 1)
        x = f / self.peak_frequency_sw_dbpl_sound_shell
        return x ** A * ((1 + self.rb_sw_dbpl_sound_shell ** 4) / (self.rb_sw_dbpl_sound_shell ** 4 + x ** 4)) ** ((A - b) / 4) * ((b + 4) / (b + 4 - m + m * x ** 2)) ** ((b + 4) / 2)

    def mu_f_sw_dbpl_sound_shell(self, A=9.0):
        x = np.logspace(-8, 8, 6000)
        f = x * self.peak_frequency_sw_dbpl_sound_shell
        return self.safe_trapezoid(self.spectral_shape_sw_dbpl_sound_shell(f, A), np.log(x))

    def peak_amplitude_sw_dbpl_sound_shell(self, A=9.0) -> float:
        """
            Based on https://arxiv.org/abs/1909.10040
            """
        x = np.logspace(-8, 8, 6000)
        frequencies = x * self.peak_frequency_sw_dbpl_sound_shell
        shape = self.spectral_shape_sw_dbpl_sound_shell(frequencies, A)
        mu_f = self.safe_trapezoid(shape, np.log(x))
        normalization = 3.0 / mu_f / 2.061
        omega_ratio = 0.014 / 0.012
        return normalization * omega_ratio * self.peak_amplitude_sw_sgbp_lattice_2017

    def gw_sw_dbpl_sound_shell(self, f, A=9.0):
        return self.peak_amplitude_sw_dbpl_sound_shell(A) * self.spectral_shape_sw_dbpl_sound_shell(f, A)

    ####################################################################################
    # Shared helpers for the 2022 semi-analytic sound-wave and collision templates.    #
    # Paper: https://arxiv.org/abs/2208.11697                                          #
    ####################################################################################

    def _peak_frequency_general_semi_analytic_2022(self, A):
        """
        From https://arxiv.org/pdf/2208.11697
        """
        return self.redshift_freq * (
            A
            * (8 * np.pi) ** (1 / 3)
            * self.bubble_wall_velocity
            / (2 * np.pi * self.length_scale)
        )

    def _spectral_shape_general_semi_analytic_2022(self, f, x, a, b, c):
        return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c

    ####################################################################################
    # Sound waves: semi-analytic fit.                                                  #
    # Paper: https://arxiv.org/abs/2208.11697                                          #
    # Select in gw_template_choices.json with:                                         #
    #     "sound_wave_template": "semi-analytic_2022"                                  #
    ####################################################################################

    def peak_frequency_sw_semi_analytic_2022(self, B=0.66):
        return self.peak_frequency_sw_semi * B

    def peak_amplitude_sw_semi_analytic_2022(self, A=0.0514):
        return A * self.amplitude_scale_sw_semi_analytic_2022

    def spectral_shape_sw_semi_analytic_2022(self, f, A=0.0514, B=0.66, a=2.36, b=2.36, c=3.69):
        x = f / (self.peak_frequency_sw_semi * B)
        return A * (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c

    def gw_sw_semi_analytic_2022(self, f, A=0.0514, B=0.66, a=2.36, b=2.36, c=3.69):
        return self.amplitude_scale_sw_semi_analytic_2022 * self.spectral_shape_sw_semi_analytic_2022(f, A, B, a, b, c)
    
    ####################################################################################
    # Sound waves: double broken power law fit to Higgsless fluid simulations.         #
    # Paper: https://arxiv.org/abs/2409.03651                                          #
    # Select in gw_template_choices.json with:                                         #
    #     "sound_wave_template": "higgsless_2024"                                      #
    ####################################################################################

    def integrated_amplitude_sw_higgsless_2024(self, OMEGA_SW=0.032272833, S=1.012754507, b=1.267966476) -> float:
        """
            Fit from https://arxiv.org/abs/2409.03651 taking account of erratum
            """
        K2 = S * self.kinetic_energy_fraction
        RH = self.hydro_transition_temp.hubble_constant * self.length_scale
        factor = (8 * np.pi) ** (1 / 3)
        betaTf = factor * self.bubble_wall_velocity / self.length_scale_Tf
        betaTp = factor * self.bubble_wall_velocity / self.length_scale
        betaoverH_Tf = betaTf / self.hydro_Tf.hubble_constant
        betaoverH_Tp = betaTp / self.hydro_transition_temp.hubble_constant
        dt0 = 11 / betaoverH_Tf
        fluid_velocity = (self.kinetic_energy_fraction / self.hydro_transition_temp.adiabatic_index(self.Pf)) ** 0.5
        tau_sw = self.length_scale / fluid_velocity
        dtfin = tau_sw * self.hydro_transition_temp.hubble_constant
        A_hyp = special.hyp2f1(2, 1 - 2 * b, 2.0 - 2 * b, (dt0 + dtfin) / (dt0 - 1))
        B_hyp = special.hyp2f1(2, 1 - 2 * b, 2.0 - 2 * b, dt0 / (dt0 - 1.0))
        factor1 = 1.0 / (1.0 - 2 * b)
        factor2 = (1 + dtfin / dt0) ** (1.0 - 2 * b) * A_hyp - B_hyp
        K2int = K2 ** 2 * dt0 / (dt0 - 1.0) ** 2 * factor1 * factor2
        h2Omega = 3 * OMEGA_SW * self.redshift_amp * K2int * RH
        return h2Omega
    
    def peak_frequency_sw_higgsless_2024(self, k1R=0.55, k2R=0.552492698, n3=-2.668604333):
        x_peak = self._peak_frequency_ratio_sw_higgsless_2024(k1R, k2R, n3)
        return x_peak * self.redshift_freq / self.length_scale

    @staticmethod
    def _spectral_shape_raw_sw_higgsless_2024(x, k1, k2, n3):
        n1 = 3.0
        n2 = 1.0
        a1 = 3.6
        a2 = 2.4
        return (
            (x / k1) ** n1
            * (1 + (x / k1) ** a1) ** ((n2 - n1) / a1)
            * (1 + (x / k2) ** a2) ** ((n3 - n2) / a2)
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _peak_frequency_ratio_sw_higgsless_2024(k1, k2, n3):
        n1 = 3.0
        n2 = 1.0
        a1 = 3.6
        a2 = 2.4

        def log_derivative(log_x):
            x = np.exp(log_x)
            y1 = (x / k1) ** a1
            y2 = (x / k2) ** a2
            return (
                n1
                + (n2 - n1) * y1 / (1 + y1)
                + (n3 - n2) * y2 / (1 + y2)
            )

        return np.exp(optimize.brentq(log_derivative, -50.0, 50.0))

    # The user's Higgsless spectrum retains integration over the supplied frequency grid.
    
    def spectral_shape_sw_higgsless_2024(self, f: float, k1R=0.55, k2R=0.552492698, n3=-2.668604333) -> float:
        """
            From https://arxiv.org/abs/2409.03651 (Eq. 2.8)
            """
        f1 = k1R * self.redshift_freq / self.length_scale
        f2 = k2R * self.redshift_freq / self.length_scale
        n1 = 3.0
        n2 = 1.0
        a1 = 3.6
        a2 = 2.4
        S = (f / f1) ** n1 * (1 + (f / f1) ** a1) ** ((n2 - n1) / a1) * (1 + (f / f2) ** a2) ** ((n3 - n2) / a2)
        mu = self.safe_trapezoid(S, np.log(f), axis=-1)
        S2 = S / mu
        return S2
        
    def gw_sw_higgsless_2024(self, f, OMEGA_SW=0.032272833, S=1.012754507,
                           b=1.267966476, k1R=0.55, k2R=0.552492698, n3=-2.668604333):
        return self.integrated_amplitude_sw_higgsless_2024(OMEGA_SW, S, b) * self.spectral_shape_sw_higgsless_2024(f, k1R, k2R, n3)

    ####################################################################################
    # Turbulence: analytic broken-power-law template.                                  #
    # Paper: https://arxiv.org/abs/0909.0622                                           #
    # Select in gw_template_choices.json with:                                         #
    #     "turbulence_template": "analytic_2009"                                       #
    ####################################################################################

    @property
    def peak_frequency_turb_analytic_2009(self):
        return 3.5 * self.redshift_freq / self.length_scale

    @property
    def peak_amplitude_turb_analytic_2009(self) -> float:
        A = 9.0
        return A * self.redshift_amp * self.hydro_transition_temp.hubble_constant * self.length_scale * (self.kappa_turb * self.kinetic_energy_fraction) ** (3 / 2) * self._unnormalised_spectral_shape_turb_analytic_2009(self.peak_frequency_turb_analytic_2009)

    def _unnormalised_spectral_shape_turb_analytic_2009(self, f: float) -> float:
        x = f / self.peak_frequency_turb_analytic_2009
        return x ** 3 / ((1 + x) ** (11 / 3) * (1 + 8 * np.pi * f / (self.redshift_freq * self.hydro_transition_temp.hubble_constant)))

    def spectral_shape_turb_analytic_2009(self, f: float) -> float:
        return self._unnormalised_spectral_shape_turb_analytic_2009(f) / self._unnormalised_spectral_shape_turb_analytic_2009(self.peak_frequency_turb_analytic_2009)

    def gw_turb_analytic_2009(self, f):
        return (
            self.peak_amplitude_turb_analytic_2009
            * self.spectral_shape_turb_analytic_2009(f)
        )

    ####################################################################################
    # Bubble collisions: semi-analytic fit.                                            #
    # Paper: https://arxiv.org/abs/2208.11697                                          #
    # Select in gw_template_choices.json with:                                         #
    #     "collision_template": "semi-analytic_2022"                                   #
    ####################################################################################

    @property
    def peak_frequency_coll_semi_analytic_2022(self):
        if self.peak_amplitude_coll_semi_analytic_2022 == 0:
            return 0.0
        """
        From https://arxiv.org/pdf/2208.11697 table I third column
        """
        A = 0.77
        return self._peak_frequency_general_semi_analytic_2022(A)

    @property
    def peak_amplitude_coll_semi_analytic_2022(self) -> float:
        """
            Based on https://arxiv.org/abs/2208.11697
            """
        if self.kappa_coll is None:
            return 0.0
        A = 0.0531
        return A * self.redshift_amp * (self.hydro_transition_temp.hubble_constant * self.length_scale / ((8 * np.pi) ** (1 / 3) * self.bubble_wall_velocity)) ** 2 * self.kinetic_energy_fraction ** 2

    def spectral_shape_coll_semi_analytic_2022(self, f):
        """
        Based on https://arxiv.org/abs/2208.11697 table I third column
        """
        a = 2.41
        b = 2.42
        c = 4.08
        x = f / self.peak_frequency_coll_semi_analytic_2022
        return self._spectral_shape_general_semi_analytic_2022(f, x, a, b, c)

    def gw_coll_semi_analytic_2022(self, f):
        return (
            self.peak_amplitude_coll_semi_analytic_2022
            * self.spectral_shape_coll_semi_analytic_2022(f)
        )

    ####################################################################################
    # Template dispatch                                                                #
    ####################################################################################

    def gw_sw(self, f, **overrides):
        if self.sound_wave_template is None:
            return np.zeros_like(f, dtype=float)
        functions = {
            "sgbp_lattice_2017": self.gw_sw_sgbp_lattice_2017,
            "dbpl_sound_shell": self.gw_sw_dbpl_sound_shell,
            "semi-analytic_2022": self.gw_sw_semi_analytic_2022,
            "higgsless_2024": self.gw_sw_higgsless_2024,
            "higgsless_old": self.gw_sw_2,
        }
        if self.sound_wave_template not in functions:
            raise ValueError(f"Unknown sound-wave template: {self.sound_wave_template}")
        parameters = {**self.sound_wave_parameters, **overrides}
        return functions[self.sound_wave_template](f, **parameters)

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

    def gw_total(self, f):
        return self.gw_sw(f) + self.gw_turb(f) + self.gw_coll(f)

    def report(self, *detectors):
            report = {}
            parameters = self.sound_wave_parameters
            report["Sound-wave template"] = self.sound_wave_template
            report["Sound-wave parameter overrides"] = dict(parameters)

            if self.sound_wave_template is None:
                report["Peak amplitude (sound waves)"] = 0.0
                report["Peak frequency (sound waves)"] = 0.0
            elif self.sound_wave_template == "sgbp_lattice_2017":
                report["Peak amplitude (sound waves)"] = (
                    self.peak_amplitude_sw_sgbp_lattice_2017
                )
                report["Peak frequency (sound waves)"] = (
                    self.peak_frequency_sw_sgbp_lattice_2017
                )
            elif self.sound_wave_template == "dbpl_sound_shell":
                report["Peak amplitude (sound waves)"] = (
                    self.peak_amplitude_sw_dbpl_sound_shell(parameters.get("A", 9.0))
                )
                report["Peak frequency (sound waves)"] = (
                    self.peak_frequency_sw_dbpl_sound_shell
                )
                report["Bubble separation break frequency (sound waves)"] = (
                    self.bubble_separation_break_frequency_sw_dbpl_sound_shell
                )
            elif self.sound_wave_template == "semi-analytic_2022":
                report['Peak amplitude (sound waves)'] = self.peak_amplitude_sw_semi_analytic_2022(parameters.get("A", 0.0514))
                report['Peak frequency (sound waves)'] = (
                    self.peak_frequency_sw_semi_analytic_2022(parameters.get("B", 0.66))
                )
            elif self.sound_wave_template == "higgsless_2024":
                report['Integrated amplitude (sound waves)'] = self.integrated_amplitude_sw_higgsless_2024(
                    parameters.get("OMEGA_SW", 0.032272833), parameters.get("S", 1.012754507),
                    parameters.get("b", 1.267966476))
                report['Sound-wave normalisation'] = 'Integral over supplied frequency grid' 
                report['Peak frequency (sound waves)'] = (
                    self.peak_frequency_sw_higgsless_2024(parameters.get("k1R", 0.55), parameters.get("k2R", 0.552492698), parameters.get("n3", -2.668604333))
                )           
            elif self.sound_wave_template == "higgsless_old":
                report["Amplitude coefficient (sound waves)"] = self.peak_amplitude_sw_higgsless
            else:
                raise ValueError(
                    f"Unknown sound-wave template: {self.sound_wave_template}."
                )

            if self.turbulence_template is None:
                report["Peak amplitude (turbulence)"] = 0.0
                report["Peak frequency (turbulence)"] = 0.0
            else:
                report["Peak amplitude (turbulence)"] = (
                    self.peak_amplitude_turb_analytic_2009
                )
                report["Peak frequency (turbulence)"] = (
                    self.peak_frequency_turb_analytic_2009
                )

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


    @property
    def at_percolation(self):
        return self.source_temperature is None

    def safe_trapezoid(self, y, x, axis=-1):
        """
            Safely compute the trapezoidal integral of y with respect to x.

            Parameters
            ----------
            y : array_like
                Values to integrate.
            x : array_like
                Integration variable.
            axis : int, optional
                Axis along which to integrate (default: -1).

            Returns
            -------
            float or ndarray
                The integral result.
            """
        try:
            return np.trapezoid(y, x, axis=axis)
        except AttributeError:
            return np.trapz(y, x, axis=axis)

    @property
    def peak_frequency_sw_semi(self):
        return self.redshift_freq * ((8 * np.pi) ** (1 / 3) * self.bubble_wall_velocity / (2 * np.pi * self.length_scale))

    @property
    def amplitude_scale_sw_semi_analytic_2022(self) -> float:
        """
            Based on https://arxiv.org/abs/2208.11697
            """
        return self.redshift_amp * (self.hydro_transition_temp.hubble_constant * self.length_scale / ((8 * np.pi) ** (1 / 3) * self.bubble_wall_velocity)) ** 2 * self.kinetic_energy_fraction ** 2

    @property
    def peak_amplitude_sw_higgsless(self) -> float:
        """
            Fit from https://arxiv.org/abs/2502.19478 taking account of erratum
            """
        OMEGA_SW = 0.0311
        _k = self.hydro_transition_temp.alpha / (1 + self.hydro_transition_temp.alpha)
        Ksw = _k * 0.6 * self.kappa_sw
        f1 = 0.2 * self.redshift_freq / self.length_scale
        f2 = 0.5 * self.redshift_freq / self.length_scale / self.rb_sw_dbpl_sound_shell
        RH = self.hydro_transition_temp.hubble_constant * self.length_scale
        Htau_shell = RH / np.sqrt(Ksw / (4 / 3))
        Htau_sw = min(1, Htau_shell)
        h2Omega_int_sw = 3 * OMEGA_SW * self.redshift_amp * Ksw ** 2 * Htau_sw * RH
        return 1 / np.pi * (np.sqrt(2) + 2 * (f2 / f1) / (1 + f2 ** 2 / f1 ** 2)) * h2Omega_int_sw

    def spectral_shape_sw_double_broken_higgsless(self, f: float):
        """
            From https://arxiv.org/abs/2403.03723 (Eq. 2.8)
            """
        f1 = 0.2 * self.redshift_freq / self.length_scale
        f2 = 0.5 * self.redshift_freq / self.length_scale / self.rb_sw_dbpl_sound_shell
        n1 = 3.0
        n2 = 1.0
        n3 = -3.0
        a1 = 2
        a2 = 4
        S = (f / f1) ** n1 * (1 + (f / f1) ** a1) ** ((n2 - n1) / a1) * (1 + (f / f2) ** a2) ** ((n3 - n2) / a2)
        S_at_f2 = (f2 / f1) ** n1 * (1 + (f2 / f1) ** a1) ** ((n2 - n1) / a1) * 2 ** ((n3 - n2) / a2)
        S2 = S / S_at_f2
        return S2

    def gw_sw_1(self, f):
        return self.gw_sw_sgbp_lattice_2017(f)

    def gw_sw_2(self, f):
        return self.peak_amplitude_sw_higgsless * self.spectral_shape_sw_double_broken_higgsless(f)

    def gw_sw_3(self, f, OMEGA_SW, S, b, k1R, k2R, n3):
        return self.gw_sw_higgsless_2024(f, OMEGA_SW, S, b, k1R, k2R, n3)

    def gw_sw_4(self, f, A):
        return self.gw_sw_dbpl_sound_shell(f, A)

    def gw_semi(self, f, A, B, a, b, c):
        return self.gw_sw_semi_analytic_2022(f, A, B, a, b, c)

    def gw_sw_temp(self, f, T='Tp', *, OMEGA_SW=0.032272833, S=1.012754507, b=1.267966476, k1R=0.55, k2R=0.552492698, n3=-2.668604333):
        """Evaluate Higgsless new 2024 at a chosen source temperature.

            T may be "Tp", "Tf", "start" (Pf=0.9), or a number in GeV.
            None also selects the canonical percolation prediction. Pass the
            frequency array at once, so hydrodynamics is evaluated once per call.
            """
        report = self.transition_report
        if T is None:
            temperature = float(report['T_p'])
        elif isinstance(T, str):
            name = T.strip().lower().replace(' ', '').replace('_', '')
            if name == 'tp':
                temperature = float(report['T_p'])
            elif name == 'tf':
                if report.get('T_f') is None:
                    raise ValueError('Cannot evaluate Tf without a completion temperature')
                temperature = float(report['T_f'])
            elif name in ('start', 'ts', 't(pf=0.9)'):
                temperature = _temperature_at_false_vacuum_fraction(report, 0.9)
            else:
                raise ValueError("T must be 'Tp', 'Tf', 'start', or a value in GeV")
        else:
            temperature = float(T)
        if not np.isfinite(temperature):
            raise ValueError('Source temperature must be finite')
        at_percolation = temperature == float(report['T_p'])
        if at_percolation and self.at_percolation:
            gw = self
        else:
            if not at_percolation:
                key = 'bubble_separation' if self.use_bubble_sep else 'bubble_radius'
                length = interpolate_transition_report(report, key, temperature)
                if not np.isfinite(length) or length <= 0:
                    raise ValueError(f'No positive {key} at T={temperature:g} GeV')
            gw = AnalyseIndividualTransition(self.phase_structure, report, self.potential, use_bubble_sep=self.use_bubble_sep, sound_wave_template=self.sound_wave_template, turbulence_template=self.turbulence_template, collision_template=self.collision_template, kappa_coll=self.kappa_coll, kappa_turb=self.kappa_turb, source_temperature=None if at_percolation else temperature)
            if not at_percolation:
                if self.at_percolation:
                    gw.hydro_Tf = self.hydro_Tf
                else:
                    gw.hydro_Tf = hydrodynamics.make_hydro_vars(self.from_phase, self.to_phase, self.potential, float(report['T_f']), self.phase_structure.ground_state_energy_density)
        return gw.gw_sw_3(f, OMEGA_SW, S, b, k1R, k2R, n3)

    def plot_template_comparison(self, frequencies, detectors=None, ptas=None, ax=None):
            if ax is None:
                ax = plt.gca()

            if detectors is not None:
                for detector in detectors:
                    ax.loglog(frequencies, detector(frequencies), label=detector.label)

            if ptas is not None:
                for i, pta in enumerate(ptas):
                    pta.plot(ax, color=f"C{i}")

            #ax.loglog(frequencies, self.gw_total(frequencies), label="total")
            #ax.loglog(frequencies, self.gw_sw(frequencies), label="sw")
            #ax.loglog(frequencies, self.gw_sw_dbpl_sound_shell(frequencies), label="sw")
            #ax.loglog(frequencies, self.gw_sw_3(frequencies, OMEGA_SW=0.032272833, S=1.012754507, b=1.267966476, k1R=0.55, k2R=0.552492698, n3=-2.668604333), color='red', alpha=0.5, label="Higgsless new")
            #ax.loglog(frequencies, self.gw_sw_4(frequencies, A = 3), label="sound shell new", color='black')
            # Higgsless new 2024: same template parameters, three source temperatures.
            ax.loglog(frequencies, self.gw_sw_temp(frequencies, "Tf"),
                      label="Higgsless new 2024: Tf", color='red', linestyle=':')
            ax.loglog(frequencies, self.gw_sw_temp(frequencies, "Tp"),
                      label="Higgsless new 2024: Tp", color='black', linestyle='-')
            ax.loglog(frequencies, self.gw_sw_temp(frequencies, "start"),
                      label="Higgsless new 2024: T(Pf=0.9)", color='blue', linestyle='--')
            #ax.loglog(frequencies, self.gw_sw_4(frequencies, A = 9), label="sound shell old", color='navy', linestyle='--')
            #ax.loglog(frequencies, self.gw_sw_2(frequencies), label="higgsless old", color='cyan')
            #ax.loglog(frequencies, self.gw_sw_1(frequencies), label="single bronken", color='gold')
            #ax.loglog(frequencies, self.gw_turb(frequencies), label="turb")
            #ax.loglog(frequencies, self.gw_coll(frequencies), label="coll")
            ax.set_xlim(1e-10, 1e-5)  # 例如：频率范围 10^-4 到 10^-1 Hz
            ax.set_ylim(1e-14, 1e-7)  # 例如：振幅范围 10^-18 到 10^-4
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
                    lambda f: np.sum([g.gw_total(f) for g in gws.values()], axis=0)
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
        return np.sum([g.gw_total(f) for g in self.gws.values()], axis=0)

    def gw_sw(self, f):
        return np.sum([g.gw_sw(f) for g in self.gws.values()], axis=0)

    def gw_turb(self, f):
        return np.sum([g.gw_turb(f) for g in self.gws.values()], axis=0)

    def gw_coll(self, f):
        return np.sum([g.gw_coll(f) for g in self.gws.values()], axis=0)