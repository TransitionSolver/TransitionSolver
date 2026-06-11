"""
Test platform-specific PhaseTracer paths
=======================================
"""

import unittest
import tempfile
from pathlib import Path

from TransitionSolver.phasetracer import compiler_search_args, rpath, shared_library


class PhaseTracerPlatformTest(unittest.TestCase):
    def test_shared_library_uses_dylib_on_macos(self):
        result = shared_library(Path("/pt/lib"), "phasetracer", platform="darwin")

        self.assertEqual(Path("/pt/lib/libphasetracer.dylib"), result)

    def test_shared_library_uses_so_off_macos(self):
        result = shared_library(Path("/pt/lib"), "phasetracer", platform="linux")

        self.assertEqual(Path("/pt/lib/libphasetracer.so"), result)

    def test_rpath_uses_darwin_linker_syntax_on_macos(self):
        result = rpath(Path("/pt/lib"), platform="darwin")

        self.assertEqual("-Wl,-rpath,/pt/lib", result)

    def test_rpath_uses_gnu_linker_syntax_off_macos(self):
        result = rpath(Path("/pt/lib"), platform="linux")

        self.assertEqual("-Wl,-rpath=/pt/lib", result)

    def test_compiler_search_args_uses_existing_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "prefix"
            include = prefix / "include"
            lib = prefix / "lib"
            include.mkdir(parents=True)
            lib.mkdir()

            result = compiler_search_args(prefixes=(prefix, Path(tmp) / "missing"))

        self.assertEqual(["-I", include, f"-L{lib}", rpath(lib)], result)


if __name__ == "__main__":
    unittest.main()
