"""
Test cppyy compatibility helpers
================================
"""

import importlib.util
import os
import pathlib
import subprocess
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "TransitionSolver" / "cppyy_compat.py"

spec = importlib.util.spec_from_file_location("cppyy_compat_under_test", MODULE)
cppyy_compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cppyy_compat)


class CppyyCompatibilityTest(unittest.TestCase):
    def test_adds_default_include_paths(self):
        cppyy = mock.Mock()

        with mock.patch.object(pathlib.Path, "exists", return_value=True):
            cppyy_compat._add_default_include_paths(cppyy)

        cppyy.add_include_path.assert_any_call("/opt/homebrew/include")
        cppyy.add_include_path.assert_any_call("/usr/local/include")

    def test_repair_is_noop_off_macos(self):
        error = RuntimeError(cppyy_compat.MACPORTS_ZSTD)

        repaired = cppyy_compat._repair_macos_zstd_dependency(error, platform="linux")

        self.assertFalse(repaired)

    def test_unsupported_macos_sdk_is_reported_before_importing_cppyy(self):
        with mock.patch.object(cppyy_compat, "_macos_sdk_version", return_value="26.4"), \
                mock.patch.object(cppyy_compat, "_cppyy_cling_version", return_value="6.32.8"), \
                mock.patch.object(cppyy_compat, "_find_compatible_macos_sdk", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                cppyy_compat._raise_for_unsupported_macos_sdk(platform="darwin")

        self.assertIn("macOS SDK 26.4", str(ctx.exception))
        self.assertIn("cppyy-cling 6.32.8", str(ctx.exception))

    def test_unsupported_macos_sdk_uses_compatible_sdk_when_available(self):
        sdk = pathlib.Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")

        with mock.patch.object(cppyy_compat, "_macos_sdk_version", return_value="26.4"), \
                mock.patch.object(cppyy_compat, "_cppyy_cling_version", return_value="6.32.8"), \
                mock.patch.object(cppyy_compat, "_find_compatible_macos_sdk", return_value=sdk), \
                mock.patch.dict(os.environ, {}, clear=True):
            cppyy_compat._raise_for_unsupported_macos_sdk(platform="darwin")
            self.assertEqual(str(sdk), os.environ["SDKROOT"])
            self.assertIn(f"-isysroot {sdk}", os.environ["EXTRA_CLING_ARGS"])
            self.assertIn("-target arm64-apple-macos15.0", os.environ["EXTRA_CLING_ARGS"])
            self.assertTrue(os.environ["CLING_STANDARD_PCH"].endswith("allDict.cxx.pch"))

    def test_repair_patches_macos_zstd_dependency(self):
        error = RuntimeError(cppyy_compat.MACPORTS_ZSTD)
        backend_copy = pathlib.Path("/tmp/cppyy_backend")
        lib_cling = backend_copy / "lib" / "libCling.so"
        backend_library = backend_copy / "lib" / "libcppyy_backend.so"
        zstd = pathlib.Path("/opt/homebrew/lib/libzstd.1.dylib")
        completed = subprocess.CompletedProcess(args=[], returncode=0)

        with mock.patch.object(cppyy_compat, "_copy_cppyy_backend", return_value=backend_copy), \
                mock.patch.object(pathlib.Path, "exists", return_value=True), \
                mock.patch.object(cppyy_compat, "_find_homebrew_zstd", return_value=zstd), \
                mock.patch.object(cppyy_compat.shutil, "which", return_value="/usr/bin/install_name_tool"), \
                mock.patch.object(cppyy_compat.subprocess, "run", return_value=completed) as run, \
                mock.patch.dict(os.environ, {}, clear=True):
            repaired = cppyy_compat._repair_macos_zstd_dependency(error, platform="darwin")
            self.assertEqual(str(backend_library), os.environ["CPPYY_BACKEND_LIBRARY"])

        self.assertTrue(repaired)
        run.assert_called_once_with(
            [
                "/usr/bin/install_name_tool",
                "-change",
                cppyy_compat.MACPORTS_ZSTD,
                str(zstd),
                str(lib_cling),
            ],
            capture_output=True,
            text=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
