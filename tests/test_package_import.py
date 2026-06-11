"""
Test package import behavior
============================
"""

import subprocess
import sys
import unittest


class PackageImportTest(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_effective_potential(self):
        code = (
            "import sys; "
            "import TransitionSolver; "
            "print('TransitionSolver.effective_potential' in sys.modules)"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual("", result.stderr)
        self.assertEqual("False", result.stdout.strip())
        self.assertEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
