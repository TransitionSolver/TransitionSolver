import os
import subprocess
import sys


def test_action_module_import_does_not_load_phase_tracer_action_header(tmp_path):
    env = os.environ.copy()
    env.setdefault("PHASETRACER", "/Users/zy/work/TransitionSolver/PhaseTracer")
    env["TRANSITIONSOLVER_COSMOTRANSITIONS_CACHE"] = str(tmp_path / "cosmotransitions")
    env["TRANSITIONSOLVER_CPPYY_BACKEND_CACHE"] = str(tmp_path / "cppyy-backend")
    env["TRANSITIONSOLVER_CPPYY_PCH_CACHE"] = str(tmp_path / "cppyy-pch")

    result = subprocess.run(
        [sys.executable, "-c", "import TransitionSolver.action; print('import ok')"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "import ok" in result.stdout
