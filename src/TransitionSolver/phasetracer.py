"""
Find PhaseTracer paths
======================
"""

import os
from pathlib import Path

PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_UNIT_TEST = PT_HOME / "bin" / "unit_tests"
