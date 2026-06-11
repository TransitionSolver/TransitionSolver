"""
TransitionSolver
================
"""

from importlib import import_module


_EXPORTS = {
    "load_potential": ("effective_potential", "load_potential"),
    "RSS_BP1": ("benchmarks", "RSS_BP1"),
    "RSS_BP2": ("benchmarks", "RSS_BP2"),
    "RSS_BP3": ("benchmarks", "RSS_BP3"),
    "read_phase_tracer": ("phasetracer", "read_phase_tracer"),
    "run_phase_tracer": ("phasetracer", "run_phase_tracer"),
    "build_phase_tracer": ("phasetracer", "build_phase_tracer"),
    "phase_tracer_info": ("phasetracer", "phase_tracer_info"),
    "find_phase_history": ("phasehistory", "find_phase_history"),
    "plot_summary": ("plot", "plot_summary"),
    "saveall": ("report", "saveall"),
    "PhaseHistoryAnalyser": ("analysis.phase_history_analysis", "PhaseHistoryAnalyser"),
    "GWAnalyser": ("gws", "GWAnalyser"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
