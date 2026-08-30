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

from TransitionSolver.gws import GWAnalyser, lisa
from TransitionSolver.gws.analyser import AnalyseIndividualTransition
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
    assert np.isclose(snr, 36.08117516093357)


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
        AnalyseIndividualTransition._spectral_shape_sw_higgsless_2024_raw(
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
        AnalyseIndividualTransition._spectral_shape_sw_higgsless_2024_raw(
            k1, k1, k2, n3
        )
    )
    assert shape_at_peak > (
        AnalyseIndividualTransition._spectral_shape_sw_higgsless_2024_raw(
            k2, k1, k2, n3
        )
    )
    assert np.isclose(shape_at_peak / shape_integral, 0.7226121793)


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
