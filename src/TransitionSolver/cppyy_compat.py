"""
Compatibility helpers for importing cppyy
=========================================
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform as platform_module
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from typing import Optional


MACPORTS_ZSTD = "/opt/local/lib/libzstd.1.dylib"
HOMEBREW_ZSTD_CANDIDATES = (
    pathlib.Path("/opt/homebrew/lib/libzstd.1.dylib"),
    pathlib.Path("/usr/local/lib/libzstd.1.dylib"),
)
DEFAULT_INCLUDE_PATHS = (
    pathlib.Path("/opt/homebrew/include"),
    pathlib.Path("/usr/local/include"),
)
MACOS_SDK_SEARCH_DIRS = (
    pathlib.Path("/Library/Developer/CommandLineTools/SDKs"),
    pathlib.Path("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"),
)


def import_cppyy():
    """
    Import cppyy, repairing a known macOS wheel dependency mismatch if possible.

    Some cppyy-cling macOS wheels reference MacPorts' zstd path even on
    Homebrew systems. If Homebrew's zstd is present, patch the dependency in a
    TransitionSolver-owned copy of cppyy_backend before importing cppyy.
    """
    _raise_for_unsupported_macos_sdk()
    _prepare_macos_zstd_dependency()

    try:
        cppyy = importlib.import_module("cppyy")
    except RuntimeError as exc:
        if _repair_macos_zstd_dependency(exc):
            _clear_cppyy_imports()
            cppyy = importlib.import_module("cppyy")
        else:
            raise _cppyy_import_error(exc) from exc
    _add_default_include_paths(cppyy)
    return cppyy


def _repair_macos_zstd_dependency(error: RuntimeError, platform: str = sys.platform) -> bool:
    if platform != "darwin" or MACPORTS_ZSTD not in str(error):
        return False

    return _prepare_macos_zstd_dependency(platform=platform, force=True, error=error)


def _raise_for_unsupported_macos_sdk(platform: str = sys.platform) -> None:
    if platform != "darwin" or os.getenv("TRANSITIONSOLVER_CPPYY_ALLOW_UNSUPPORTED_SDK"):
        return

    sdk_version = _macos_sdk_version()
    cling_version = _cppyy_cling_version()
    if sdk_version is None or cling_version is None:
        return

    if _version_at_least(sdk_version, (26, 0)) and _version_at_most(cling_version, (6, 32, 8)):
        compatible_sdk = _find_compatible_macos_sdk()
        if compatible_sdk is not None:
            _configure_macos_cppyy_sdk(compatible_sdk)
            return

        raise RuntimeError(
            "Detected macOS SDK "
            f"{sdk_version} with cppyy-cling {cling_version}. This combination can crash "
            "Cling while parsing the current macOS libc++ headers before Python can "
            "recover. Install Xcode/Command Line Tools with a macOS 15 SDK, or set "
            "TRANSITIONSOLVER_CPPYY_SDKROOT to a compatible SDK path. You may also set "
            "TRANSITIONSOLVER_CPPYY_ALLOW_UNSUPPORTED_SDK=1 "
            "to bypass this guard if you have verified cppyy works in your environment."
        )


def _macos_sdk_version() -> Optional[str]:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None

    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "--show-sdk-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _cppyy_cling_version() -> Optional[str]:
    try:
        return importlib.metadata.version("cppyy-cling")
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_compatible_macos_sdk() -> pathlib.Path | None:
    override = os.getenv("TRANSITIONSOLVER_CPPYY_SDKROOT")
    if override:
        path = pathlib.Path(override)
        return path if path.exists() else None

    sdkroot = os.getenv("SDKROOT")
    if sdkroot:
        path = pathlib.Path(sdkroot)
        version = _sdk_version_from_path(path)
        if path.exists() and version is not None and _version_at_least(version, (15, 0)) and _version_at_most(version, (15, 99)):
            return path

    candidates = []
    for root in MACOS_SDK_SEARCH_DIRS:
        for sdk in root.glob("MacOSX*.sdk"):
            version = _sdk_version_from_path(sdk)
            if version is not None and _version_at_least(version, (15, 0)) and _version_at_most(version, (15, 99)):
                candidates.append((version, sdk))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _configure_macos_cppyy_sdk(sdk: pathlib.Path) -> None:
    sdk = sdk.resolve()
    os.environ["SDKROOT"] = str(sdk)

    flags = os.environ.get("EXTRA_CLING_ARGS", "")
    additions = [
        f"-isysroot {sdk}",
        f"-target {_macos_target_triple()}",
        "-std=c++20",
    ]
    for addition in additions:
        if addition not in flags:
            flags = f"{flags} {addition}".strip()
    os.environ["EXTRA_CLING_ARGS"] = flags

    if not os.getenv("CLING_STANDARD_PCH"):
        pch = _cppyy_pch_cache_path()
        try:
            pch.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pch = _cppyy_pch_temp_path()
            pch.parent.mkdir(parents=True, exist_ok=True)
        os.environ["CLING_STANDARD_PCH"] = str(pch)


def _macos_target_triple() -> str:
    machine = platform_module.machine()
    if machine == "x86_64":
        return "x86_64-apple-macos15.0"
    return "arm64-apple-macos15.0"


def _cppyy_pch_cache_path() -> pathlib.Path:
    override = os.getenv("TRANSITIONSOLVER_CPPYY_PCH_CACHE")
    if override:
        return pathlib.Path(override) / "allDict.cxx.pch"

    tag = sysconfig.get_config_var("SOABI") or f"py{sys.version_info.major}{sys.version_info.minor}"
    return pathlib.Path.home() / ".cache" / "TransitionSolver" / "cppyy_pch" / tag / "allDict.cxx.pch"


def _cppyy_pch_temp_path() -> pathlib.Path:
    tag = sysconfig.get_config_var("SOABI") or f"py{sys.version_info.major}{sys.version_info.minor}"
    return pathlib.Path(tempfile.gettempdir()) / "TransitionSolver" / "cppyy_pch" / tag / "allDict.cxx.pch"


def _prepare_macos_zstd_dependency(
    platform: str = sys.platform, force: bool = False, error: Optional[RuntimeError] = None
) -> bool:
    if platform != "darwin" or os.getenv("CPPYY_BACKEND_LIBRARY"):
        return False

    backend_src = _find_cppyy_backend()
    if backend_src is None:
        return False

    src_lib_cling = backend_src / "lib" / "libCling.so"
    if not force and not _linked_to_macports_zstd(src_lib_cling):
        return False

    backend_copy = _copy_cppyy_backend(backend_src)
    if backend_copy is None:
        return False

    lib_cling = backend_copy / "lib" / "libCling.so"
    backend_library = backend_copy / "lib" / "libcppyy_backend.so"
    zstd = _find_homebrew_zstd()
    install_name_tool = shutil.which("install_name_tool")

    if not lib_cling.exists() or not backend_library.exists() or zstd is None or install_name_tool is None:
        return False

    result = subprocess.run(
        [install_name_tool, "-change", MACPORTS_ZSTD, str(zstd), str(lib_cling)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "cppyy could not load because its bundled libCling references "
            f"{MACPORTS_ZSTD}, and TransitionSolver could not patch it to use "
            f"{zstd} in its local cppyy backend copy. install_name_tool said:\n{result.stderr}"
        ) from error

    backend_parent = str(backend_copy.parent)
    if backend_parent not in sys.path:
        sys.path.insert(0, backend_parent)

    os.environ["CPPYY_BACKEND_LIBRARY"] = str(backend_library)
    _clear_cppyy_imports()
    return True


def _find_cppyy_backend() -> pathlib.Path | None:
    spec = importlib.util.find_spec("cppyy_backend")
    if spec is None or spec.origin is None:
        return None

    return pathlib.Path(spec.origin).parent


def _copy_cppyy_backend(src: pathlib.Path | None = None) -> pathlib.Path | None:
    if src is None:
        src = _find_cppyy_backend()
    if src is None:
        return None

    dst = _cppyy_backend_cache_package_dir()
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def _cppyy_backend_cache_package_dir() -> pathlib.Path:
    override = os.getenv("TRANSITIONSOLVER_CPPYY_BACKEND_CACHE")
    if override:
        return pathlib.Path(override) / "cppyy_backend"

    tag = sysconfig.get_config_var("SOABI") or f"py{sys.version_info.major}{sys.version_info.minor}"
    return pathlib.Path.home() / ".cache" / "TransitionSolver" / "cppyy_backend" / tag / "cppyy_backend"


def _find_homebrew_zstd() -> pathlib.Path | None:
    for candidate in HOMEBREW_ZSTD_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _add_default_include_paths(cppyy) -> None:
    for include_path in DEFAULT_INCLUDE_PATHS:
        if include_path.exists():
            cppyy.add_include_path(str(include_path))


def _linked_to_macports_zstd(lib_cling: pathlib.Path) -> bool:
    if not lib_cling.exists():
        return False

    otool = shutil.which("otool")
    if otool is None:
        return False

    result = subprocess.run([otool, "-L", str(lib_cling)], capture_output=True, text=True, check=False)
    return result.returncode == 0 and MACPORTS_ZSTD in result.stdout


def _version_at_least(version: str, minimum: tuple) -> bool:
    return _version_tuple(version) >= minimum


def _version_at_most(version: str, maximum: tuple) -> bool:
    return _version_tuple(version) <= maximum


def _version_tuple(version: str) -> tuple:
    if isinstance(version, tuple):
        return version

    parts = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _sdk_version_from_path(path: pathlib.Path) -> tuple | None:
    name = path.name
    if not name.startswith("MacOSX") or not name.endswith(".sdk"):
        return None

    version = name[len("MacOSX"):-len(".sdk")]
    if not version or version == ".sdk":
        return None
    return _version_tuple(version)


def _clear_cppyy_imports() -> None:
    for name in list(sys.modules):
        if (
            name == "cppyy"
            or name.startswith("cppyy.")
            or name == "cppyy_backend"
            or name.startswith("cppyy_backend.")
        ):
            sys.modules.pop(name, None)


def _cppyy_import_error(error: RuntimeError) -> RuntimeError:
    if sys.platform == "darwin" and MACPORTS_ZSTD in str(error):
        return RuntimeError(
            "cppyy could not load because its bundled libCling references "
            f"{MACPORTS_ZSTD}. Install zstd with Homebrew, then retry so "
            "TransitionSolver can patch a local copy of cppyy's backend, or "
            "set up a cppyy installation whose libCling links to an existing "
            "zstd library."
        )
    return error
