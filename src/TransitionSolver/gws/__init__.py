"""
Gravitational wave calculations
===============================
"""

from .giese_kappa import kappa_nu_model
from .detectors.lisa import lisa, lisa_thrane, lisa_thrane_1_yr, lisa_thrane_2019, lisa_thrane_2019_snr_1, lisa_thrane_2019_snr_10
from .detectors.pta import nanograv_15, ppta_dr3, epta_dr2_full
from .detectors.detector import Detector
from .analyser import GWAnalyser
from .hydrodynamics import HydroVars
from . import hydrodynamics
