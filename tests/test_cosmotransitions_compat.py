"""
Test cosmoTransitions compatibility helpers
===========================================
"""

import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "TransitionSolver" / "cosmotransitions_compat.py"

spec = importlib.util.spec_from_file_location("cosmotransitions_compat_under_test", MODULE)
cosmotransitions_compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cosmotransitions_compat)


class CosmoTransitionsCompatibilityTest(unittest.TestCase):
    def test_uses_package_in_place_when_finite_t_data_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = pathlib.Path(tmp) / "cosmoTransitions"
            package_dir.mkdir()
            for name in cosmotransitions_compat.FINITE_T_DATA_FILES:
                (package_dir / name).write_text("0 0\n", encoding="utf-8")

            with mock.patch.object(
                cosmotransitions_compat, "_find_cosmotransitions_package", return_value=package_dir
            ), mock.patch.object(cosmotransitions_compat, "_directory_writable", return_value=False), \
                    mock.patch.dict(os.environ, {}, clear=True):
                cosmotransitions_compat.ensure_cosmotransitions_writable()

            self.assertNotEqual(str(package_dir.parent), sys.path[0])

    def test_copies_package_to_cache_when_installation_is_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package_dir = root / "site-packages" / "cosmoTransitions"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "finiteT.py").write_text("value = 1\n", encoding="utf-8")

            cache_parent = root / "cache"

            with mock.patch.object(
                cosmotransitions_compat, "_find_cosmotransitions_package", return_value=package_dir
            ), mock.patch.object(cosmotransitions_compat, "_directory_writable", return_value=False), \
                    mock.patch.dict(os.environ, {"TRANSITIONSOLVER_COSMOTRANSITIONS_CACHE": str(cache_parent)}):
                original_path = list(sys.path)
                try:
                    cosmotransitions_compat.ensure_cosmotransitions_writable()
                    self.assertEqual(str(cache_parent), sys.path[0])
                finally:
                    sys.path[:] = original_path

            self.assertTrue((cache_parent / "cosmoTransitions" / "__init__.py").exists())
            self.assertTrue((cache_parent / "cosmoTransitions" / "finiteT.py").exists())


if __name__ == "__main__":
    unittest.main()
