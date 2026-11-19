"""
Test gravitational waves
========================
"""

import json
import os
from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from TransitionSolver.gws import GWAnalyser, lisa
from TransitionSolver.gws.analyser import (
    AnalyseIndividualTransition,
    interpolate_transition_report,
)
from TransitionSolver import gws, benchmarks
from dictcmp import assert_deep_equal


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"

phase_tracer_file = BASELINE / "rss_bp1_phase_structure.dat"


# use a function here else failure to decode json brings down whole test suite
def get_phase_history(name="RSS_BP1"):
    with open(BASELINE / f"{name.lower()}_phase_structure.json", "r") as f:
        return json.load(f)


NAMES = [f"RSS_BP{k}" for k in range(1, 14)]


PYTEST_MPL_KWARGS = {
    "remove_text": True,
    "deterministic": True,
    "savefig_kwargs": {"format": "pdf"},
    "tolerance": 25,
}


@pytest.mark.parametrize("name", NAMES)
def test_report(generate_baseline, name):
    if name in ["RSS_BP12"]:
        pytest.xfail(f"{name} is expected to fail")

    analyser = GWAnalyser(
        getattr(benchmarks, name),
        get_phase_history(name),
        phase_tracer_file=BASELINE / f"{name.lower()}_phase_structure.dat",
    )
    report = analyser.report(lisa)
    assert_deep_equal(
        report,
        BASELINE / f"{name.lower()}_gw.json",
        generate_baseline=generate_baseline,
    )


@pytest.mark.mpl_image_compare(**PYTEST_MPL_KWARGS)
def test_plot_gw():
    analyser = GWAnalyser(
        benchmarks.RSS_BP1, get_phase_history(), phase_tracer_file=phase_tracer_file
    )
    return analyser.plot(detectors=[lisa], ptas=[gws.nanograv_15])


def test_snr():
    analyser = GWAnalyser(
        benchmarks.RSS_BP1, get_phase_history(), phase_tracer_file=phase_tracer_file
    )
    snr = lisa.SNR(analyser.gw_total)
    assert np.isclose(snr, 8.909189289876617)


def test_higgsless_2024_peak_and_normalisation():
    k1 = 0.39
    k2 = 0.45
    n3 = -3.0

    x_peak = (
        AnalyseIndividualTransition._peak_frequency_ratio_sw_higgsless_2024(
            k1, k2, n3
        )
    )
    shape_at_peak = (
        AnalyseIndividualTransition._spectral_shape_raw_sw_higgsless_2024(
            x_peak, k1, k2, n3
        )
    )
    shape_integral = (
        AnalyseIndividualTransition._spectral_shape_integral_sw_higgsless_2024(
            k1, k2, n3
        )
    )

    assert np.isclose(x_peak, 0.4233116849)
    assert k1 < x_peak < k2
    assert shape_at_peak > (
        AnalyseIndividualTransition._spectral_shape_raw_sw_higgsless_2024(
            k1, k1, k2, n3
        )
    )
    assert shape_at_peak > (
        AnalyseIndividualTransition._spectral_shape_raw_sw_higgsless_2024(
            k2, k1, k2, n3
        )
    )
    assert np.isclose(shape_at_peak / shape_integral, 0.7226121793)


def test_higgsless_2024_expanding_time_integral():
    with open(BASELINE / "rss_bp1_phase_structure_pt_action.json", "r") as f:
        phase_history = json.load(f)

    analyser = GWAnalyser(
        benchmarks.RSS_BP1,
        phase_history,
        phase_tracer_file=phase_tracer_file,
        sound_wave_template="higgsless_2024",
    )
    transition = analyser.gws["1"]
    hubble_p = transition.hydro_transition_temp.hubble_constant
    hubble_f = transition.hydro_Tf.hubble_constant
    beta_f = (
        (8 * np.pi) ** (1 / 3)
        * transition.bubble_wall_velocity
        / transition.length_scale_Tf
    )
    dt0 = 11 * hubble_f / beta_f
    fluid_velocity = np.sqrt(
        transition.kinetic_energy_fraction
        / transition.hydro_transition_temp.adiabatic_index(transition.Pf)
    )
    duration = hubble_p * transition.length_scale / fluid_velocity

    time_integral = quad(
        lambda delta_eta: (dt0 / (dt0 + delta_eta)) ** (2 * 1.17)
        / (1 + delta_eta) ** 2,
        0,
        duration,
    )[0]
    shape_peak = AnalyseIndividualTransition._spectral_shape_raw_sw_higgsless_2024(
        0.4233116849, 0.39, 0.45, -3.0
    )
    shape_integral = (
        AnalyseIndividualTransition._spectral_shape_integral_sw_higgsless_2024(
            0.39, 0.45, -3.0
        )
    )
    expected_amplitude = (
        3
        * 3.11e-2
        * transition.redshift_amp
        * (0.84 * transition.kinetic_energy_fraction) ** 2
        * time_integral
        * hubble_p
        * transition.length_scale
        * shape_peak
        / shape_integral
    )

    assert np.isclose(
        transition.peak_amplitude_sw_higgsless_2024,
        expected_amplitude,
        rtol=1e-7,
        atol=0,
    )


def test_source_temperature_report(generate_baseline):
    analyser = GWAnalyser(
        benchmarks.RSS_BP4,
        get_phase_history("RSS_BP4"),
        phase_tracer_file=BASELINE / "rss_bp4_phase_structure.dat",
    )
    analysis = analyser.transition_at_temperature("0", 63.5)

    assert_deep_equal(
        analysis.report(lisa),
        BASELINE / "rss_bp4_gw_at_temperature.json",
        generate_baseline=generate_baseline,
    )


def test_temperature_scan_uses_all_valid_samples(monkeypatch):
    phase_history = get_phase_history("RSS_BP4")
    phase_history["transitions"]["0"]["T_f"] = None
    analyser = GWAnalyser(
        benchmarks.RSS_BP4,
        phase_history,
        phase_tracer_file=BASELINE / "rss_bp4_phase_structure.dat",
    )
    monkeypatch.setattr(
        analyser,
        "_report_at_temperature",
        lambda _, temperature, *detectors: {
            "Transition temperature": temperature,
        },
    )

    report = analyser.temperature_scan_report("0")
    transition = phase_history["transitions"]["0"]
    expected = [
        temperature
        for temperature, separation in zip(
            transition["T"], transition["bubble_separation"]
        )
        if np.isfinite(separation) and separation > 0
    ]

    assert [result["Transition temperature"] for result in report["Results"]] == sorted(
        expected, reverse=True
    )


def test_temperature_uncertainty_reports_sampled_ranges(monkeypatch):
    phase_history = get_phase_history("RSS_BP4")
    analyser = GWAnalyser(
        benchmarks.RSS_BP4,
        phase_history,
        phase_tracer_file=BASELINE / "rss_bp4_phase_structure.dat",
    )
    transition = phase_history["transitions"]["0"]

    def fake_report(_, temperature, *detectors):
        return {
            "Transition temperature": temperature,
            "Test quantity": (temperature - transition["T_p"]) ** 2,
            "Signal-to-Noise Ratio": {"Test detector": temperature},
        }

    monkeypatch.setattr(analyser, "_report_at_temperature", fake_report)
    report = analyser.temperature_uncertainty_report("0")
    start = report["Highest evaluated temperature"]
    value_range = report["Ranges"]["Test quantity"]

    assert interpolate_transition_report(transition, "Pf", start) == pytest.approx(
        0.9
    )
    assert report["Lowest evaluated temperature"] == transition["T_f"]
    assert value_range["Minimum"] >= 0
    assert value_range["Temperature at minimum"] <= start
    assert "Test detector" in report["Ranges"]["Signal-to-Noise Ratio"]


def test_interpolation_rejects_temperature_outside_saved_history():
    transition = {"T": [10.0, 5.0], "value": [1.0, 2.0]}

    with pytest.raises(ValueError, match="saved temperature range"):
        interpolate_transition_report(transition, "value", 11.0)


@pytest.mark.mpl_image_compare(**PYTEST_MPL_KWARGS)
def test_plot_pta():
    fig, ax = plt.subplots()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")

    gws.nanograv_15.plot(ax, color="red")
    gws.ppta_dr3.plot(ax, color="green")
    gws.epta_dr2_full.plot(ax, color="blue")

    ax.legend(scatterpoints=1)

    return fig


@pytest.mark.mpl_image_compare(**PYTEST_MPL_KWARGS)
def test_plot_lisa():
    f = np.logspace(-4, -1, 400)
    fig, ax = plt.subplots()

    ax.loglog(f, gws.lisa(f), label=gws.lisa.label)
    ax.loglog(f, gws.lisa_thrane(f), label=gws.lisa_thrane.label)
    ax.loglog(f, gws.lisa_thrane_1_yr(f), label=gws.lisa_thrane_1_yr.label)
    ax.loglog(f, gws.lisa_thrane_2019(f), label=gws.lisa_thrane_2019.label)
    ax.loglog(f, gws.lisa_thrane_2019_snr_1(f), label=gws.lisa_thrane_2019_snr_1.label)
    ax.loglog(
        f, gws.lisa_thrane_2019_snr_10(f), label=gws.lisa_thrane_2019_snr_10.label
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Frequency")

    return fig
