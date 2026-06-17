"""
Compatibility helpers for importing cppyy on macOS.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile


MACPORTS_ZSTD = "/opt/local/lib/libzstd.1.dylib"
HOMEBREW_ZSTD = pathlib.Path("/opt/homebrew/lib/libzstd.1.dylib")
MACOS_SDK_SEARCH_DIRS = (
    pathlib.Path("/Library/Developer/CommandLineTools/SDKs"),
    pathlib.Path(
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform"
        "/Developer/SDKs"
    ),
)


def import_cppyy():
    """Import cppyy after applying macOS fixes needed by current wheels."""
    _configure_macos_sdk()
    _patch_macos_zstd()

    cppyy = importlib.import_module("cppyy")
    if sys.platform == "darwin":
        for include_path in ("/opt/homebrew/include", "/usr/local/include"):
            if pathlib.Path(include_path).exists():
                cppyy.add_include_path(include_path)
    return cppyy


def _configure_macos_sdk() -> None:
    if sys.platform != "darwin":
        return

    sdk_version = _macos_sdk_version()
    cling_version = _cppyy_cling_version()
    if sdk_version is None or cling_version is None:
        return

    if (
        _version_tuple(sdk_version) < (26, 0)
        or _version_tuple(cling_version) > (6, 32, 8)
    ):
        return

    sdk = _find_macos_15_sdk()
    if sdk is None:
        return

    os.environ["SDKROOT"] = str(sdk)
    args = os.environ.get("EXTRA_CLING_ARGS", "")
    for arg in (f"-isysroot {sdk}", f"-target {_macos_target()}", "-std=c++20"):
        if arg not in args:
            args = f"{args} {arg}".strip()
    os.environ["EXTRA_CLING_ARGS"] = args

    if "CLING_STANDARD_PCH" not in os.environ:
        pch = _cache_dir("cppyy_pch") / "allDict.cxx.pch"
        pch.parent.mkdir(parents=True, exist_ok=True)
        os.environ["CLING_STANDARD_PCH"] = str(pch)


def _patch_macos_zstd() -> None:
    if sys.platform != "darwin" or os.environ.get("CPPYY_BACKEND_LIBRARY"):
        return

    backend = _cppyy_backend_dir()
    if backend is None:
        return

    lib_cling = backend / "lib" / "libCling.so"
    if not _linked_to(lib_cling, MACPORTS_ZSTD):
        return

    if not HOMEBREW_ZSTD.exists():
        return

    copied_backend = _copy_cppyy_backend(backend)
    copied_lib_cling = copied_backend / "lib" / "libCling.so"
    backend_library = copied_backend / "lib" / "libcppyy_backend.so"
    if _linked_to(copied_lib_cling, MACPORTS_ZSTD):
        subprocess.run(
            [
                "install_name_tool",
                "-change",
                MACPORTS_ZSTD,
                str(HOMEBREW_ZSTD),
                str(copied_lib_cling),
            ],
            check=True,
        )
    backend_parent = str(copied_backend.parent)
    if backend_parent not in sys.path:
        sys.path.insert(0, backend_parent)
    _clear_cppyy_backend_imports()
    os.environ["CPPYY_BACKEND_LIBRARY"] = str(backend_library)


def _cppyy_backend_dir() -> pathlib.Path | None:
    spec = importlib.util.find_spec("cppyy_backend")
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).parent


def _copy_cppyy_backend(src: pathlib.Path) -> pathlib.Path:
    dst = _cache_dir("cppyy_backend") / "cppyy_backend"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def _clear_cppyy_backend_imports() -> None:
    for name in list(sys.modules):
        if name == "cppyy_backend" or name.startswith("cppyy_backend."):
            sys.modules.pop(name, None)


def _linked_to(library: pathlib.Path, install_name: str) -> bool:
    if not library.exists():
        return False
    result = subprocess.run(
        ["otool", "-L", str(library)], capture_output=True, text=True, check=False
    )
    return result.returncode == 0 and install_name in result.stdout


def _macos_sdk_version() -> str | None:
    result = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _cppyy_cling_version() -> str | None:
    try:
        return importlib.metadata.version("cppyy-cling")
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_macos_15_sdk() -> pathlib.Path | None:
    candidates = []
    for root in MACOS_SDK_SEARCH_DIRS:
        candidates.extend(root.glob("MacOSX15*.sdk"))
    return sorted(candidates)[-1] if candidates else None


def _macos_target() -> str:
    if platform.machine() == "x86_64":
        return "x86_64-apple-macos15.0"
    return "arm64-apple-macos15.0"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def _cache_dir(name: str) -> pathlib.Path:
    tag = (
        sysconfig.get_config_var("SOABI")
        or f"py{sys.version_info.major}{sys.version_info.minor}"
    )
    cache = pathlib.Path.home() / ".cache" / "TransitionSolver" / name / tag
    try:
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    except OSError:
        return (
            pathlib.Path(tempfile.gettempdir()).resolve()
            / "TransitionSolver"
            / name
            / tag
        )
