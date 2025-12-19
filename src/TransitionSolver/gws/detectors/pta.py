"""
PTA data from disk and frequencies based on period
==================================================
"""


import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .detector import SECONDS_PER_YEAR


SECONDS_PER_DAY = 24 * 60 * 60
THIS = Path(os.path.dirname(os.path.abspath(__file__)))


class PTAFromDisk:
    """
    Plot PTA data from disk
    """

    def __init__(self, f, file_name, label=None):
        """
        @param f Frequency data
        @param file_name File containing log_{10} \Omega h^2 data
        """
        self.f = f
        log10_omega_h2 = np.loadtxt(file_name)
        self.omega_h2 = 10.**log10_omega_h2
        self.label = label

    def plot(self, ax=None, rw=0.08, color="red", alpha=0.8, label=None):
        """
        @returns PTA data plotted on axis object
        """
        if ax is None:
            ax = plt.gca()

        if label is None:
            label = self.label

        box = ax.boxplot(self.omega_h2,
                         positions=self.f,
                         widths=rw * self.f,
                         whis=(2.5, 97.5),
                         showfliers=False,
                         medianprops={'lw': 0},
                         whiskerprops={'lw': 1.5},
                         showcaps=False,
                         patch_artist=True,
                         manage_ticks=False)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color=color)

        for patch in box['boxes']:
            patch.set(facecolor=color, alpha=alpha)

        ax.scatter(np.nan, np.nan, color=color, alpha=alpha,
                   marker="s", s=75, label=label)

        return ax

# NANOGrav 15yr


f = np.arange(1, 9) / (16.03 * SECONDS_PER_YEAR)
nanograv_15 = PTAFromDisk(
    f, THIS / "logOmegah2_NANO15_0708.csv", label="NANOGrav 15 yr")

# PPTA DR3

f = np.arange(1, 9) / (6605 * SECONDS_PER_DAY)
ppta_dr3 = PTAFromDisk(
    f, THIS / "logOmegah2_PPTADR3_0708.csv", label="PPTA DR3")

# EPTA DR2 Full

f = np.arange(1, 11) / (24.7 * SECONDS_PER_YEAR)
epta_dr2_full = PTAFromDisk(
    f, THIS / "logOmegah2_EPTADR2FULL_0708.csv", label="EPTA DR2 Full")
