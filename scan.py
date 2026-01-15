"""
Example of performing a scan over parameters
============================================

Perturb the RSS benchmark point 1 and re-compute the
nucleation temperature.
"""

import numpy as np

from TransitionSolver import load_potential
from TransitionSolver import build_phase_tracer, read_phase_tracer, run_phase_tracer, find_phase_history
from TransitionSolver.benchmarks import RSS_BP1_POINT


# Compile PhaseTracer model
model_header = "RSS.hpp"
exe_name = build_phase_tracer(model_header)

# Load C++ potential class
RSS = load_potential(model_header)

# Highest temperature to consider in PhaseTracer
t_high = 250.

# Scan 10 points

for i in range(10):

    # Select point in RSS parameter space by perturbing BP1

    point = np.array(RSS_BP1_POINT)
    point[0] += 0.01 * np.random.randn()

    print(f"point {i} = {point}")

    # Run PhaseTracer and parse output

    phase_structure_raw = run_phase_tracer(exe_name, point=point, t_high=t_high)
    phase_structure = read_phase_tracer(phase_structure_raw)

    # Make potential for this point

    potential = RSS(point)
    potential.set_daisy_method(2)  # TODO remove all these settings
    potential.set_bUseBoltzmannSuppression(True)
    potential.set_useGSResummation(True)

    # Run TransitionSolver

    try:
        tr_report = find_phase_history(potential, phase_structure)
    except Exception as e:
        print(f"error = {e}")
        continue

    # Inspect the nucleation temperature in the first transition

    Tn = tr_report['transitions'][0]['Tn']
    print(f"Nucleation temperature = {Tn}")
