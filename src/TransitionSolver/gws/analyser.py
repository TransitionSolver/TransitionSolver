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
OMEGA_SW = 0.012  # From erratum of https://arxiv.org/abs/1704.05871 TABLE IV.

logger = logging.getLogger(__name__)

GW_TEMPLATE_CHOICES_FILE = "gw_template_choices.json"


def read_gw_template_choices() -> dict:
    template_file = files("TransitionSolver.settings").joinpath(
        GW_TEMPLATE_CHOICES_FILE
    )

    with template_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def interpolate_transition_report(
    transition_report: dict, key: str, temperature: float
) -> float:
    """Linearly interpolate a transition-history quantity in temperature."""
    temperatures = np.asarray(transition_report["T"], dtype=float)
    values = np.asarray(transition_report[key], dtype=float)
    if not temperatures.min() <= temperature <= temperatures.max():
        raise ValueError(
            f"Cannot interpolate {key} at T={temperature}: the saved "
            f"temperature range is [{temperatures.min()}, {temperatures.max()}]"
        )
    order = np.argsort(temperatures)
    return float(np.interp(temperature, temperatures[order], values[order]))


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
        source_temperature=None,
    ):
        self.sound_wave_template = sound_wave_template
        self.turbulence_template = turbulence_template
        self.collision_template = collision_template
        self.kappa_turb = kappa_turb
        self.kappa_coll = kappa_coll
        self.use_bubble_sep = use_bubble_sep
        self.source_temperature = source_temperature

        self.transition_report = transition_report
        self.from_phase = phase_structure.phases[transition_report["false_phase"]]
        self.to_phase = phase_structure.phases[transition_report["true_phase"]]
        self.potential = potential

        if not self.at_percolation:
            sampled_min = min(transition_report["T"])
            sampled_max = max(transition_report["T"])
            if not sampled_min <= self.source_temperature <= sampled_max:
                raise ValueError(
                    "Source temperature must be within the saved transition-history "
                    f"range [{sampled_min}, {sampled_max}]"
                )

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
        if not self.at_percolation:
            return interpolate_transition_report(
                self.transition_report, "Pf", self.source_temperature
            )
        return self.transition_report["perc_threshold_pf"]

    @property
    def at_percolation(self):
        return (
            self.source_temperature is None
            or self.source_temperature == self.transition_report.get("T_p")
        )

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
        if not self.at_percolation:
            return self.source_temperature
        return self.transition_report["T_p"]

    @cached_property
    def redshift_temp(self) -> float:
        if not self.at_percolation:
            T_min = max(
                self.from_phase.T[0],
                self.to_phase.T[0],
                self.potential.minimum_temperature,
            )
            return hydrodynamics.reheat_temperature(
                self.from_phase,
                self.to_phase,
                self.potential,
                self.source_temperature,
                self.transition_report["T_c"],
                T_min,
            )
        return self.transition_report["Treh_p"]

    @property
    def bubble_wall_velocity(self) -> float:
        if not self.at_percolation:
            return interpolate_transition_report(
                self.transition_report,
                "bubble_wall_velocity",
                self.source_temperature,
            )
        return self.transition_report["bubble_wall_velocity_p"]

    @property
    def peak_frequency_coll(self):
        if self.peak_amplitude_coll == 0:
            return 0.0
        return self.redshift_freq * (
            0.77
            * (8 * np.pi) ** (1 / 3)
            * self.bubble_wall_velocity
            / (2 * np.pi * self.length_scale)
        )

    @property
    def peak_frequency_sw_bubble_separation(self):
        return 1.58 * self.redshift_freq / self.length_scale * ZP / 10

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
    def peak_amplitude_coll(self) -> float:
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

    def spectral_shape_sw_double_broken(self, f: float):
        """
        From https://arxiv.org/abs/2209.13551 (Eq. 2.11), originally from https://arxiv.org/abs/1909.10040 (Eq. 5.7)
        """
        b = 1
        m = (9 * self.rb**4 + b) / (self.rb**4 + 1)
        x = f / self.peak_frequency_sw_shell_thickness
        return (
            x**9
            * ((1 + self.rb**4) / (self.rb**4 + x**4)) ** ((9 - b) / 4)
            * ((b + 4) / (b + 4 - m + m * x**2)) ** ((b + 4) / 2)
        )

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

    def spectral_shape_coll(self, f):
        a = 2.41
        b = 2.42
        c = 4.08
        x = f / self.peak_frequency_coll
        # Using normalised spectral shape, so A = 5.13e-2 is moved to the
        # amplitude calculation.
        return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c

    def gw_total(self, f):
        return self.gw_sw(f) + self.gw_turb(f) + self.gw_coll(f)

    def gw_sw_sgbp_lattice_2017(self, f):
        return self.peak_amplitude_sw * self.spectral_shape_sw(f)

    def gw_sw_dbpl_sound_shell(self, f):
        return (
            self.peak_amplitude_sw_sound_shell * self.spectral_shape_sw_double_broken(f)
        )

    def gw_sw(self, f):
        if self.sound_wave_template is None:
            return np.zeros_like(f, dtype=float)

        sound_wave_functions = {
            "sgbp_lattice_2017": self.gw_sw_sgbp_lattice_2017,
            "dbpl_sound_shell": self.gw_sw_dbpl_sound_shell,
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
        return self.peak_amplitude_coll * self.spectral_shape_coll(f)

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
        if not self.at_percolation:
            key = "bubble_separation" if self.use_bubble_sep else "bubble_radius"
            return interpolate_transition_report(
                self.transition_report, key, self.source_temperature
            )

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
        else:
            report["Peak amplitude (collisions)"] = self.peak_amplitude_coll
            report["Peak frequency (collisions)"] = self.peak_frequency_coll

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
        transition_ids=None,
        source_temperatures=None,
        **kwargs,
    ):
        if phase_tracer_file is not None:
            phase_structure = read_phase_tracer(phase_tracer_file=phase_tracer_file)

        if transition_ids is not None:
            transition_ids = [str(transition_id) for transition_id in transition_ids]
            relevant_transitions = {
                transition_id: phase_history["transitions"][transition_id]
                for transition_id in transition_ids
            }
        elif force_relevant:
            relevant_transitions = phase_history["transitions"]
        else:
            relevant_transitions = extract_relevant_transitions(phase_history)

        if not relevant_transitions:
            raise RuntimeError("No relevant transition detected in the phase history")

        gw_kwargs = read_gw_template_choices()
        gw_kwargs.update(kwargs)
        source_temperatures = {
            str(k): v for k, v in (source_temperatures or {}).items()
        }

        self.phase_structure = phase_structure
        self.potential = potential
        self.transition_reports = relevant_transitions
        self.gw_kwargs = gw_kwargs

        self.gws = {
            k: AnalyseIndividualTransition(
                phase_structure,
                v,
                potential,
                source_temperature=source_temperatures.get(k),
                **gw_kwargs,
            )
            for k, v in relevant_transitions.items()
        }

    def transition_at_temperature(self, transition_id, temperature):
        """Analyse one transition at a chosen physical temperature."""
        transition_id = str(transition_id)
        return AnalyseIndividualTransition(
            self.phase_structure,
            self.transition_reports[transition_id],
            self.potential,
            source_temperature=temperature,
            **self.gw_kwargs,
        )

    def _report_at_temperature(self, transition_id, temperature, *detectors):
        """Report all scan quantities for one transition temperature."""
        transition_id = str(transition_id)
        transition_report = self.transition_reports[transition_id]
        analysis = self.transition_at_temperature(transition_id, temperature)
        result = analysis.report(*detectors)
        result["False vacuum fraction"] = analysis.Pf
        result["Mean bubble separation"] = interpolate_transition_report(
            transition_report, "bubble_separation", temperature
        )
        result["Mean bubble radius"] = interpolate_transition_report(
            transition_report, "bubble_radius", temperature
        )
        result["Beta"] = interpolate_transition_report(
            transition_report, "beta", temperature
        )
        result["Hubble constant"] = interpolate_transition_report(
            transition_report, "H", temperature
        )
        result["Beta/H"] = result["Beta"] / result["Hubble constant"]

        propagation_velocity = max(
            analysis.bubble_wall_velocity,
            analysis.hydro_transition_temp.soundSpeedFalse,
        )
        conversion = (8 * np.pi) ** (1 / 3) * propagation_velocity
        result["Mean bubble separation from beta"] = (
            conversion / result["Beta"]
            if np.isfinite(result["Beta"]) and result["Beta"] > 0
            else np.nan
        )
        result["Beta/H from mean bubble separation"] = (
            conversion
            / result["Mean bubble separation"]
            / result["Hubble constant"]
            if np.isfinite(result["Mean bubble separation"])
            and result["Mean bubble separation"] > 0
            and np.isfinite(result["Hubble constant"])
            and result["Hubble constant"] > 0
            else np.nan
        )
        return result

    def temperature_scan_report(self, transition_id, *detectors):
        """Evaluate GWs over the full valid sampled temperature history."""
        transition_id = str(transition_id)
        transition_report = self.transition_reports[transition_id]

        # Keep all temperatures with a finite, positive mean bubble separation.
        temperatures = []
        for temperature, separation in zip(
            transition_report["T"],
            transition_report["bubble_separation"],
        ):
            if np.isfinite(separation) and separation > 0:
                temperatures.append(temperature)

        temperatures.sort(reverse=True)
        if not temperatures:
            raise RuntimeError(
                "No finite, positive mean bubble separation was found for "
                f"transition {transition_id}"
            )

        return {
            "Highest sampled temperature": temperatures[0],
            "Lowest sampled temperature": temperatures[-1],
            "Milestone temperatures": {
                key: transition_report.get(key)
                for key in ("T_gamma", "T_n", "T_p", "T_e", "T_f")
            },
            "Results": [
                self._report_at_temperature(transition_id, temperature, *detectors)
                for temperature in temperatures
            ],
        }

    def temperature_scan_report_for_transition_ids(self, transition_ids, *detectors):
        """Return full temperature scans for selected transitions."""
        return {
            str(transition_id): self.temperature_scan_report(
                transition_id, *detectors
            )
            for transition_id in transition_ids
        }

    @staticmethod
    def _sampled_range(results, key):
        """Return sampled extrema and the temperatures where they occur."""
        finite_results = [result for result in results if np.isfinite(result[key])]
        minimum = min(finite_results, key=lambda result: result[key])
        maximum = max(finite_results, key=lambda result: result[key])
        return {
            "Minimum": minimum[key],
            "Maximum": maximum[key],
            "Temperature at minimum": minimum["Transition temperature"],
            "Temperature at maximum": maximum["Transition temperature"],
        }

    def temperature_uncertainty_report(
        self, transition_id, *detectors, scan_report=None
    ):
        """Return sampled prediction ranges from near percolation to completion."""
        transition_id = str(transition_id)
        transition_report = self.transition_reports[transition_id]
        T_p = transition_report.get("T_p")
        T_f = transition_report.get("T_f")
        if T_p is None or T_f is None:
            raise ValueError(
                "Cannot calculate temperature uncertainty without percolation "
                "and completion temperatures"
            )

        requested_start = transition_report["T_c"] + 0.8 * (
            T_p - transition_report["T_c"]
        )
        if scan_report is None:
            temperatures = [
                temperature
                for temperature, separation in zip(
                    transition_report["T"],
                    transition_report["bubble_separation"],
                )
                if T_f <= temperature <= requested_start
                and np.isfinite(separation)
                and separation > 0
            ]
            results = [
                self._report_at_temperature(
                    transition_id, temperature, *detectors
                )
                for temperature in temperatures
            ]
        else:
            results = [
                result
                for result in scan_report["Results"]
                if T_f <= result["Transition temperature"] <= requested_start
            ]

        for boundary in (requested_start, T_f):
            if not any(
                np.isclose(result["Transition temperature"], boundary)
                for result in results
            ):
                separation = interpolate_transition_report(
                    transition_report, "bubble_separation", boundary
                )
                if np.isfinite(separation) and separation > 0:
                    results.append(
                        self._report_at_temperature(
                            transition_id, boundary, *detectors
                        )
                    )

        results.sort(key=lambda result: result["Transition temperature"], reverse=True)
        if not results:
            raise RuntimeError(
                "No valid temperatures in the uncertainty interval "
                f"[{T_f}, {requested_start}] for transition {transition_id}"
            )

        actual_start = results[0]["Transition temperature"]
        start_adjusted = not np.isclose(actual_start, requested_start)
        if start_adjusted:
            logger.warning(
                "Requested temperature-uncertainty start T=%s has no valid "
                "mean bubble separation. Starting transition %s at the first "
                "valid sampled temperature T=%s.",
                requested_start,
                transition_id,
                actual_start,
            )

        scalar_keys = [
            key
            for key, value in results[0].items()
            if key not in ("Transition temperature", "Signal-to-Noise Ratio")
            and np.isscalar(value)
        ]
        ranges = {
            key: self._sampled_range(results, key)
            for key in scalar_keys
            if any(np.isfinite(result[key]) for result in results)
        }
        snr_ranges = {
            detector: self._sampled_range(
                [
                    {
                        "Transition temperature": result["Transition temperature"],
                        detector: result["Signal-to-Noise Ratio"][detector],
                    }
                    for result in results
                ],
                detector,
            )
            for detector in results[0]["Signal-to-Noise Ratio"]
        }
        if snr_ranges:
            ranges["Signal-to-Noise Ratio"] = snr_ranges

        return {
            "Requested start temperature": requested_start,
            "Actual start temperature": actual_start,
            "Start temperature adjusted": start_adjusted,
            "Completion temperature": T_f,
            "Highest evaluated temperature": results[0]["Transition temperature"],
            "Lowest evaluated temperature": results[-1]["Transition temperature"],
            "Extrema are over sampled and interpolated endpoint evaluations": True,
            "Ranges": ranges,
        }

    def temperature_uncertainty_report_for_transition_ids(
        self, transition_ids, *detectors, scan_reports=None
    ):
        """Return temperature-uncertainty ranges for selected transitions."""
        scan_reports = scan_reports or {}
        return {
            str(transition_id): self.temperature_uncertainty_report(
                transition_id,
                *detectors,
                scan_report=scan_reports.get(str(transition_id)),
            )
            for transition_id in transition_ids
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
        return self.plot_for_transition_ids(
            self.gws, frequencies, detectors, ptas, show
        )

    def plot_for_transition_ids(
        self,
        transition_ids,
        frequencies=None,
        detectors=None,
        ptas=None,
        show=False,
    ):
        """Plot GW spectra for selected transitions."""
        if frequencies is None:
            frequencies = np.logspace(-11, 3, 1000)

        selected = [self.gws[str(i)] for i in transition_ids]
        n = len(selected)
        fig, ax = plt.subplots(n)

        if n <= 1:
            ax = [ax]

        for a, gw in zip(ax, selected):
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
