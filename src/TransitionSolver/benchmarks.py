""""
Benchmark models for testing & examples
=======================================
"""

from . import load_potential


RSS = load_potential("RSS.hpp")


def make_RSS_potential(point):
    potential = RSS(point)
    potential.set_daisy_method(2)
    potential.set_bUseBoltzmannSuppression(True)
    return potential


RSS_BP1_POINT = [-7.253805920400513969e+02, -1.142030590385311370e+02, 2.987030761010101010e-01, 6.999301941045084732e+02, 4.256276196807469319e+02, 2.780020068332878291e+05, 4.049553405540934909e+04, 5.828128166279837596e-02, 8.498960515918183023e-02, 4.516261190913655743e-01, 2.791372362500696909e+05, 4.081096031831684377e+04]
RSS_BP1 = make_RSS_potential(RSS_BP1_POINT)
