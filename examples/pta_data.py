"""
PTA data from disk and frequencies based on period
==================================================
"""

import numpy as np


year_in_seconds = 365 * 24 * 60 * 60
day_in_seconds = 24 * 60 * 60

# NANOGrav 15yr

log10_omega_h2 = np.loadtxt("./data/logOmegah2_NANO15_0708.csv")
f = np.arange(1, 9) / (16.03 * year_in_seconds)
nanograv_15 = [f, 10.**log10_omega_h2]

# PPTA DR3

log10_omega_h2 = np.loadtxt("./data/logOmegah2_PPTADR3_0708.csv")
f = np.arange(1, 9) / (6605 * day_in_seconds)
ppta_dr3 = [f, 10.**log10_omega_h2]

# EPTA DR2 Full

log10_omega_h2 = np.loadtxt("./data/logOmegah2_EPTADR2FULL_0708.csv")
f = np.arange(1,11) / (24.7 * year_in_seconds)
epta_dr2_full = [f, 10.**log10_omega_h2]

