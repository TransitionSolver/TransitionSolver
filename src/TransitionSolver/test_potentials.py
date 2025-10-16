"""
Compare potential from 2 models
===============================
"""

import numpy as np

from .effective_potential import load_potential
from .models.real_scalar_singlet_model import RealScalarSingletModel
from .models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz


point = np.loadtxt('rss_bp1.txt')
boltz = False


def reorder(point):
    python_order = ["c1", "b3", "theta", "vs", "ms", "muh", "mus", "lh", "ls", "c2", "muh0", "mus0"]
    cpp_order = ["muh", "mus", "lh", "ls", "c1", "c2", "b3", "muh0", "mus0", "vs", "ms", "theta"]
    return [point[python_order.index(e)] for e in cpp_order]

python_potential = RealScalarSingletModel_Boltz(*point) if boltz else RealScalarSingletModel(*point)
cpp_potential = load_potential(reorder(point), "RSS.hpp")
cpp_potential.set_bUseBoltzmannSuppression(boltz)

temperatures = [0, 100, 500, 1000]
field_values = [-1000, -500, -100, 0, 100, 500, 1000]
max_rel_diff = 0.

for a in field_values:
    for b in field_values:
        for T in temperatures:
        
            phi = np.array([a, b])
            python = python_potential(phi, T)
            cpp = cpp_potential(phi, T)
            
            rel_diff = (python - cpp) / (python + cpp)
            max_rel_diff = max(max_rel_diff, abs(rel_diff))

            print(f"phi = {phi}. T = {T}")
            print(f"python = {python}\ncpp = {cpp}\nDifference = {rel_diff * 100} %")

print(f"Maximum observed difference = {max_rel_diff * 100} %")



