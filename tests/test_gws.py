"""
Test gravitational waves
========================
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import matplotlib.pyplot as plt
import numpy as np

from TransitionSolver.gws import GWAnalyser, lisa
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
    assert np.isclose(snr, 59.706589252791396)


def test_source_temperature_at_percolation_uses_existing_values():
    phase_history = get_phase_history("RSS_BP4")
    analyser = GWAnalyser(
        benchmarks.RSS_BP4,
        phase_history,
        phase_tracer_file=BASELINE / "rss_bp4_phase_structure.dat",
    )
    transition_id = next(iter(analyser.gws))
    transition = phase_history["transitions"][transition_id]

    at_percolation = analyser.transition_at_temperature(
        transition_id, transition["T_p"]
    )

    assert at_percolation.transition_temp == transition["T_p"]
    assert at_percolation.redshift_temp == transition["Treh_p"]
    assert at_percolation.Pf == transition["perc_threshold_pf"]
    assert at_percolation.length_scale == transition["bubble_separation_p"]
    assert (
        at_percolation.bubble_wall_velocity
        == transition["bubble_wall_velocity_p"]
    )


def test_temperature_uncertainty_scan_uses_first_valid_sample(monkeypatch, caplog):
    phase_history = get_phase_history("RSS_BP4")
    analyser = GWAnalyser(
        benchmarks.RSS_BP4,
        phase_history,
        phase_tracer_file=BASELINE / "rss_bp4_phase_structure.dat",
    )

    def fake_analysis(_, temperature):
        return SimpleNamespace(
            Pf=0.5,
            report=lambda *detectors: {"Transition temperature": temperature},
        )

    monkeypatch.setattr(analyser, "transition_at_temperature", fake_analysis)
    report = analyser.temperature_uncertainty_report("0")

    transition = phase_history["transitions"]["0"]
    expected_start = transition["T_c"] + 0.8 * (
        transition["T_p"] - transition["T_c"]
    )
    assert report["Requested start temperature"] == expected_start
    assert report["Actual start temperature"] <= expected_start
    assert report["Start temperature adjusted"]
    assert report["Results"][0]["Transition temperature"] == report[
        "Actual start temperature"
    ]
    assert report["Results"][-1]["Transition temperature"] == transition["T_f"]
    assert "first valid sampled temperature" in caplog.text


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
