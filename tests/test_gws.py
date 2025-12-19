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
from dictcmp import allclose


THIS = Path(os.path.dirname(os.path.abspath(__file__)))

phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
phase_history_file = THIS / "rss_bp1_phase_structure.json"

with open(THIS / phase_history_file, 'r') as f:
    phase_history = json.load(f)

analyser = GWAnalyser(RSS_BP1, phase_structure_file, phase_history)


def test_report():
    report = analyser.report(lisa)
    assert allclose(report, THIS / "rss_bp1_gw.json")


@pytest.mark.mpl_image_compare
def test_plot_gw():
    return analyser.plot(detectors=[lisa], ptas=[gws.nanograv_15])


def test_snr():
    snr = lisa.SNR(analyser.gw_total)
    assert np.isclose(snr, 302.98903223016237)
    
    
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
