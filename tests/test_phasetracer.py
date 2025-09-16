"""
Check PhaseTracer installation
==============================
"""

import filecmp
import os
import subprocess
from pathlib import Path

import numpy as np

from TransitionSolver import phasetracer


THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def test_pt_home():
    assert phasetracer.PT_HOME.is_dir()


def test_pt_lib():
    assert phasetracer.PT_LIB.is_file()


def test_pt_unit_test():
    subprocess.check_call(phasetracer.PT_UNIT_TEST)


def test_pt_run():
    point = np.loadtxt(THIS / 'rss_bp1.txt')
    result = phasetracer.pt_run('run_RSS', point)
    assert filecmp.cmp(result, THIS / "rss_bp1_phase_structure.dat")
