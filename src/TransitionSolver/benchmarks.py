""""
Benchmark models for testing & examples
=======================================
"""

from . import load_potential


def make_rss_bp(point):
    """
    @returns RSS model with settings to match benchmark points
    """
    potential = load_potential(point, "RSS.hpp")
    potential.set_daisy_method(2)
    potential.set_bUseBoltzmannSuppression(True)
    return potential


RSS_BP1_POINT = [278002.0068332878, 40495.53405540935, 0.058281281662798376, 0.08498960515918183, -725.3805920400514, 0.4516261190913656, -114.20305903853114, 279137.2362500697, 40810.960318316844, 699.9301941045085, 425.62761968074693, 0.2987030761010101]
RSS_BP1 = make_rss_bp(RSS_BP1_POINT)
