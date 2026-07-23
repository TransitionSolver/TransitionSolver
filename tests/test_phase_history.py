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


def make_analyser():
    phase_tracer_file = BASELINE / "rss_bp1_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    return phasehistory.PhaseHistoryAnalyser(benchmarks.RSS_BP1, phase_structure)


@pytest.mark.skip(reason="we found this to be platform dependent and recommend PT by default")
@pytest.mark.parametrize("name", NAMES)
def test_phase_history(generate_baseline, name):
    phase_tracer_file = BASELINE / f"{name.lower()}_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    model = getattr(benchmarks, name)
    result = phasehistory.find_phase_history(
        model, phase_structure, bubble_wall_velocity=1, action_ct=True
    )

    ignore_entries = [
        "bubble_radius_p",
        "bubble_separation_p",
        "idx_e",
        "idx_f",
        "idx_gamma",
        "idx_n",
        "idx_nbar",
        "idx_n",
        "idx_p",
        "idx_decreasing_v_phys",
        "action_3d_nbar",
        "action_3d_n",
        "action_3d_e",
        "action_3d_p",
        "action_3d_f",
        "Treh_e",
        "Treh_p",
        "T_p",
        "alpha_p",
        "T_e",
        "T_gamma",
        "Tmin",
        "size",
    ]

    exclude_paths = [f"root['transitions'][*]['{e}']" for e in ignore_entries]

    assert_deep_equal(
        result,
        BASELINE / f"{name.lower()}_phase_structure.json",
        exclude_types=[list],
        significant_digits=3,
        generate_baseline=generate_baseline,
        exclude_paths=exclude_paths,
    )


@pytest.mark.parametrize("name", NAMES)
def test_phase_history_pt_action(generate_baseline, name):
    phase_tracer_file = BASELINE / f"{name.lower()}_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    model = getattr(benchmarks, name)
    result = phasehistory.find_phase_history(
        model, phase_structure, bubble_wall_velocity=1, action_ct=False
    )

    for transition in result["transitions"].values():
        if transition["T_p"] is not None:
            assert np.isfinite(transition["alpha_p"])

    assert_deep_equal(
        result,
        BASELINE / f"{name.lower()}_phase_structure_pt_action.json",
        exclude_types=[list],
        significant_digits=3,
        generate_baseline=generate_baseline,
        exclude_paths=["root['transitions'][*]['alpha_p']"],
    )


def test_phase_nodes():
    analyser = make_analyser()
    nodes = analyser.phase_nodes()
    node_t = [n.temperature for sub in nodes for n in sub]
    assert np.allclose(node_t, [216.1466902, 216.1466902, 101.2334823, 101.2334823])


def test_phase_indexed_trans():
    analyser = make_analyser()
    nodes = analyser.phase_nodes()
    trans = analyser.phase_indexed_trans(nodes)
    idx = [i.index for sub in trans for i in sub]
    assert idx == [0, 0]


def test_unique_transition_idx():
    analyser = make_analyser()
    assert analyser.unique_transition_idx == [0, 1]


def test_unique_transition_temperatures():
    analyser = make_analyser()
    assert np.allclose(
        analyser.unique_transition_temperatures, [216.1466902, 101.2334823]
    )


def test_is_high_temperature_phase():
    analyser = make_analyser()
    assert analyser.is_high_temperature_phase == [True, True, False]


def test_is_low_temperature_phase():
    analyser = make_analyser()
    assert analyser.is_low_temperature_phase == [False, False, True]


def test_trivial_paths():
    analyser = make_analyser()
    assert analyser.trivial_paths() == []


def test_init_frontier():
    analyser = make_analyser()
    nodes = analyser.phase_nodes()
    trans = analyser.phase_indexed_trans(nodes)
    paths, frontier = analyser.init_frontier(trans, nodes)
    id = [t.id for p in paths for t in p.transitions]
    assert id == []

    idx = [i.index for i in frontier]
    assert idx == [0, 0]


def test_analyse_transition(generate_baseline):
    analyser = make_analyser()
    nodes = analyser.phase_nodes()
    trans = analyser.phase_indexed_trans(nodes)
    _, frontier = analyser.init_frontier(trans, nodes)
    transition_edge = frontier.pop(0)
    transition = transition_edge.transition
    path = transition_edge.path

    analyser.analyse_transition(
        trans, transition_edge, path, transition, action_ct=False
    )

    assert_deep_equal(
        transition.report(),
        BASELINE / "transition_phase_structure.json",
        exclude_types=[list],
        significant_digits=3,
        generate_baseline=generate_baseline,
    )
