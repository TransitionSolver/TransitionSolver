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
from TransitionSolver import gws, RSS_BP1
from dictcmp import assert_deep_equal


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"

phase_structure_file = THIS / "rss_bp1_phase_structure.dat"


with open(BASELINE / "rss_bp1_phase_structure.json", 'r') as f:
    phase_history = json.load(f)


def test_report(generate_baseline):
    analyser = GWAnalyser(RSS_BP1, phase_structure_file, phase_history)
    report = analyser.report(lisa)
    assert_deep_equal(report, BASELINE / "rss_bp1_gw.json", generate_baseline=generate_baseline)


@pytest.mark.mpl_image_compare
def test_plot_gw():
    analyser = GWAnalyser(RSS_BP1, phase_structure_file, phase_history)
    return analyser.plot(detectors=[lisa], ptas=[gws.nanograv_15])


def test_snr():
    analyser = GWAnalyser(RSS_BP1, phase_structure_file, phase_history)
    snr = lisa.SNR(analyser.gw_total)
    assert np.isclose(snr, 61.573514537762286)
    
    
@pytest.mark.mpl_image_compare
def test_plot_pta():
    f = np.logspace(-4, -1, 400)
    fig, ax = plt.subplots()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Frequency")

    gws.nanograv_15.plot(ax, color="red")
    gws.ppta_dr3.plot(ax, color="green")
    gws.epta_dr2_full.plot(ax, color="blue")

    ax.legend(scatterpoints=1)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_lisa():
    f = np.logspace(-4, -1, 400)
    fig, ax = plt.subplots()

    ax.loglog(f, gws.lisa(f), label=gws.lisa.label)
    ax.loglog(f, gws.lisa_thrane(f), label=gws.lisa_thrane.label)
    ax.loglog(f, gws.lisa_thrane_1_yr(f), label=gws.lisa_thrane_1_yr.label)
    ax.loglog(f, gws.lisa_thrane_2019(f), label=gws.lisa_thrane_2019.label)
    ax.loglog(f, gws.lisa_thrane_2019_snr_1(f), label=gws.lisa_thrane_2019_snr_1.label)
    ax.loglog(f, gws.lisa_thrane_2019_snr_10(f), label=gws.lisa_thrane_2019_snr_10.label)
    ax.legend(loc="upper left")
    ax.set_xlabel("Frequency")

    return fig
