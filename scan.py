"""
Example of performing a scan over parameters
============================================
"""

import numpy as np

from TransitionSolver import load_potential
from TransitionSolver import build_phase_tracer, read_phase_tracer, run_phase_tracer, find_phase_history
from TransitionSolver.benchmarks import RSS_BP1_POINT


model_header = "RSS.hpp"
potential = load_potential(model_header)
exe_name = build_phase_tracer(model_header)
t_high = 250.


for i in range(10):

    # Select point
    
    point = np.array(RSS_BP1_POINT)
    point[0] += 0.01 * np.random.randn()
    
    print(f"point {i} = {point}")
    
    # Run PhaseTracer
    
    phase_structure_raw = run_phase_tracer(exe_name, point=point, t_high=t_high)
    phase_structure = read_phase_tracer(phase_structure_raw)
    
    # Make potential
    
    p = potential(point)
    p.set_daisy_method(2)
    p.set_bUseBoltzmannSuppression(True)
    p.set_useGSResummation(True)

    # Run TransitionSolver

    try:
        tr_report = find_phase_history(p, phase_structure)
    except Exception as e:
        print(f"error = {e}")
        continue
        
    # Inspect quantity of interest

    Tn = tr_report['transitions'][0]['Tn']
    print(f"Nucleation temperature = {Tn}")
