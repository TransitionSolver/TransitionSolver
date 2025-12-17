"""
Algebraic simplification of CJ velocity + kappa nu model
========================================================
"""

from sympy.abc import a, c, v
from sympy import sqrt, simplify


v = (1 + sqrt(3 * a * (1 - c**2 + 3 * c**2 * a))) / (1 / c + 3 * c * a)
v = simplify(v)

nu = 1 / c**2 + 1
tmp = 1 - 3 * a + v**2 * (1 / c**2 + 3 * a)
disc = 4 * v**2 * (1 - nu) + tmp**2
disc = simplify(disc)
vm = (tmp + sqrt(disc)) / 2 / (nu - 1.) / v
vm = simplify(vm)

print(disc)
print(vm)
