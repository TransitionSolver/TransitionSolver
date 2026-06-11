"""
Test CLI import behavior
========================
"""

import subprocess
import sys
import unittest


class CliImportTest(unittest.TestCase):
    def test_cli_import_does_not_import_cosmotransitions(self):
        code = (
            "import sys; "
            "from TransitionSolver.cli import cli; "
            "print('cosmoTransitions.generic_potential' in sys.modules)"
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
