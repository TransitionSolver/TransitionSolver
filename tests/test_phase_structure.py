""""
Test phase structure report
===========================
"""

import os
from pathlib import Path

from TransitionSolver import phasehistory, RSS_BP1, read_phase_tracer
from dictcmp import assert_deep_equal

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline" 


def test_phase_structure(generate_baseline):
    phase_tracer_file = THIS / "rss_bp1_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    result = phasehistory.find_phase_history(
        RSS_BP1, phase_structure, vw=1)
    # TODO ignoring T and SonT as the size of them changes between local & github
    # but good to reinclude them later
    assert_deep_equal(result, BASELINE / "rss_bp1_phase_structure.json", exclude_paths=['analysisTime', 'T', 'SonT'], generate_baseline=generate_baseline)
