"""Test result-file output."""

from types import SimpleNamespace

from TransitionSolver import report


def test_transition_outputs_copy_parameter_point(tmp_path, monkeypatch):
    point_file = tmp_path / "input.txt"
    point_file.write_text("1.0 2.0\n", encoding="utf-8")
    output = tmp_path / "results"

    monkeypatch.setattr(report, "phase_tracer_info", lambda: {})
    figure = SimpleNamespace(savefig=lambda path: path.touch())
    context = SimpleNamespace(params={})

    report.save_transition_outputs(
        {"paths": [], "transitions": {}},
        figure,
        "phase tracer output",
        context,
        output,
        point_file,
    )

    assert (output / "parameter_point.txt").read_text(encoding="utf-8") == (
        point_file.read_text(encoding="utf-8")
    )
