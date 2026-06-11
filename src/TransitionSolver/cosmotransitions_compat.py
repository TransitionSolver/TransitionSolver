"""
Compatibility helpers for importing cosmoTransitions
====================================================
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import sys
import sysconfig


FINITE_T_DATA_FILES = ("finiteT_f.dat.txt", "finiteT_b.dat.txt")


def ensure_cosmotransitions_writable() -> None:
    """
    Make cosmoTransitions importable when its installation is read-only.

    cosmoTransitions.finiteT writes spline data files next to finiteT.py at
    import time. If that package directory is not writable, use a TransitionSolver
    cache copy of cosmoTransitions instead of asking users to change permissions
    in site-packages.
    """
    package_dir = _find_cosmotransitions_package()
    if package_dir is None:
        return

    if _finite_t_data_ready(package_dir) or _directory_writable(package_dir):
        return

    cache_parent = _cosmotransitions_cache_dir()
    package_copy = cache_parent / "cosmoTransitions"

    try:
        shutil.copytree(package_dir, package_copy, dirs_exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "cosmoTransitions needs to write finiteT spline data during import, "
            f"but its installation directory is not writable: {package_dir}. "
            "TransitionSolver also could not prepare a writable package copy at "
            f"{package_copy}. Set TRANSITIONSOLVER_COSMOTRANSITIONS_CACHE to a "
            "writable directory and retry."
        ) from exc

    cache_parent_str = str(cache_parent)
    if cache_parent_str not in sys.path:
        sys.path.insert(0, cache_parent_str)

    _clear_cosmotransitions_imports()


def _find_cosmotransitions_package() -> pathlib.Path | None:
    spec = importlib.util.find_spec("cosmoTransitions")
    if spec is None or spec.origin is None:
        return None

    return pathlib.Path(spec.origin).parent


def _finite_t_data_ready(package_dir: pathlib.Path) -> bool:
    return all((package_dir / name).exists() for name in FINITE_T_DATA_FILES)


def _directory_writable(path: pathlib.Path) -> bool:
    return os.access(path, os.W_OK)


def _cosmotransitions_cache_dir() -> pathlib.Path:
    override = os.getenv("TRANSITIONSOLVER_COSMOTRANSITIONS_CACHE")
    if override:
        return pathlib.Path(override)

    tag = sysconfig.get_config_var("SOABI") or f"py{sys.version_info.major}{sys.version_info.minor}"
    return pathlib.Path.home() / ".cache" / "TransitionSolver" / "cosmoTransitions" / tag


def _clear_cosmotransitions_imports() -> None:
    for name in list(sys.modules):
        if name == "cosmoTransitions" or name.startswith("cosmoTransitions."):
            sys.modules.pop(name, None)
