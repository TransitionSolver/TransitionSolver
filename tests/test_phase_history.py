""" "
Test phase history report
===========================
"""

import os
from pathlib import Path

import pytest
import numpy as np

from TransitionSolver import phasehistory, benchmarks, read_phase_tracer
from dictcmp import assert_deep_equal

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


NAMES = [f"RSS_BP{k}" for k in range(1, 14)]

@pytest.mark.parametrize("name", NAMES)
def test_phase_history(generate_baseline, name):
    phase_tracer_file = BASELINE / f"{name.lower()}_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    model = getattr(benchmarks, name)
    result = phasehistory.find_phase_history(
        model, phase_structure, bubble_wall_velocity=1
    )
    assert_deep_equal(
        result,
        BASELINE / f"{name.lower()}_phase_structure.json",
        exclude_types=[list],
        significant_digits=2,
        generate_baseline=generate_baseline,
    )

def test_phase_analyser():
    phase_tracer_file = BASELINE / "rss_bp1_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    analyser = phasehistory.PhaseHistoryAnalyser(benchmarks.RSS_BP1, phase_structure)

    nodes = analyser.phase_nodes()
    node_t = [n.temperature for sub in nodes for n in sub]
    assert np.allclose(node_t, [216.1466902, 216.1466902, 101.2334823, 101.2334823])

    trans = analyser.phase_indexed_trans(nodes)
    idx = [i.index for sub in trans for i in sub]
    assert idx == [0, 0]

    assert analyser.unique_transition_idx == [0, 1]

    assert np.allclose(analyser.unique_transition_temperatures, [216.1466902, 101.2334823])

    assert analyser.is_high_temperature_phase == [True, True, False]

    assert analyser.is_low_temperature_phase == [False, False, True]

    assert analyser.trivial_paths() == []

    paths, frontier = analyser.init_frontier(trans, nodes)
    id = [t.id for p in paths for t in p.transitions]
    assert id == []

    idx = [i.index for i in frontier]
    assert idx == [0, 0]

    transition_edge = frontier.pop(0)
    transition = transition_edge.transition
    path = transition_edge.path

    analyser.analyse_transition(trans, transition_edge, path, transition)
    assert np.isclose(transition.properties.T_p, 215.59452845391613)
