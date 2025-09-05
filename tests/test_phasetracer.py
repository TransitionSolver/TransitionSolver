"""
Check PhaseTracer installation
==============================
"""

import subprocess

from TransitionSolver import phasetracer


def test_pt_home():
    assert phasetracer.PT_HOME.is_dir()

def test_pt_lib():
    assert phasetracer.PT_LIB.is_file()

def test_unit_pt_unit_test():
    subprocess.check_call(phasetracer.PT_UNIT_TEST)
