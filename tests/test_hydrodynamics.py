"""
Test thermodynamic normalisation used for the Hubble rate and GW energy.
"""

import numpy as np

from TransitionSolver.gws import hydrodynamics


class FixedPhase:
    def __init__(self, field_value):
        self.field_value = field_value
        self.T = np.array([1.0, 200.0])

    def find_phase_at_t(self, _temperature, _potential):
        return np.array([self.field_value])


class QuarticPotential:
    raddof = 20.0

    def __init__(self, offset=0.0):
        self.offset = offset

    def get_temperature_scale(self):
        return 100.0

    def free_energy_density(self, field, temperature):
        if field[0] == 0:
            vacuum_energy, quartic = 10.0, 2.0e-4
        else:
            vacuum_energy, quartic = -40.0, 1.0e-4
        return self.offset + vacuum_energy - quartic * temperature**4

FROM_PHASE = FixedPhase(0)
TO_PHASE = FixedPhase(1)
TEMPERATURE = 100.0

# test that for phases extending to T=0 the code works as expected:
# - The original (T=0) subtraction is still calculated correctly.
# - The radiation-dominated fallback is not used when (F_{\rm gs}(0)) is known.
# - A consistent common shift of the potential and ground-state energy has no physical effect.
def test_known_ground_state_preserves_existing_normalisation():
    potential = QuarticPotential(offset=100.0)
    ground_state_energy = 60.0
    hydro = hydrodynamics.make_hydro_vars(
        FROM_PHASE, TO_PHASE, potential, TEMPERATURE, ground_state_energy
    )

    expected_false_energy = 50.0 + 3 * 2.0e-4 * TEMPERATURE**4
    assert np.isclose(hydro.energyDensityFalse, expected_false_energy)
    assert np.isclose(
        hydro.cosmologicalEnergyDensityFalse, hydro.energyDensityFalse
    )
    assert not hydro.assumesRadiationDomination

    pf = 0.71
    expected_gamma = 1 + hydro.average_pressure_density(pf) \
        / hydro.energyDensityFalse
    assert np.isclose(hydro.adiabatic_index(pf), expected_gamma)

    shifted_potential = QuarticPotential(offset=1123.0)
    shifted = hydrodynamics.make_hydro_vars(
        FROM_PHASE, TO_PHASE, shifted_potential, TEMPERATURE,
        ground_state_energy + 1023.0
    )
    assert np.isclose(
        shifted.cosmologicalEnergyDensityFalse,
        hydro.cosmologicalEnergyDensityFalse,
    )
    assert np.isclose(shifted.hubble_constant, hydro.hubble_constant)

# check that when phases do not extedn to zero
# radiaton domination expressions are used
def test_radiation_domination_fallback_is_normalisation_independent():
    potential = QuarticPotential(offset=100.0)
    hydro = hydrodynamics.make_hydro_vars(
        FROM_PHASE, TO_PHASE, potential, TEMPERATURE, None
    )

    delta_rho = 50.0 + 3 * 1.0e-4 * TEMPERATURE**4
    radiation = np.pi**2 / 30 * potential.raddof * TEMPERATURE**4
    assert np.isclose(
        hydro.cosmologicalEnergyDensityFalse, radiation + delta_rho
    )
    assert hydro.assumesRadiationDomination
    assert hydro.adiabatic_index(0.71) == 4 / 3

    shifted = hydrodynamics.make_hydro_vars(
        FROM_PHASE, TO_PHASE, QuarticPotential(offset=1123.0),
        TEMPERATURE, None
    )
    assert np.isclose(
        shifted.cosmologicalEnergyDensityFalse,
        hydro.cosmologicalEnergyDensityFalse,
    )
    assert np.isclose(shifted.hubble_constant, hydro.hubble_constant)
    assert np.isclose(
        shifted.available_energy_fraction,
        hydro.available_energy_fraction,
    )

    direct = hydrodynamics.cosmological_energy_density(
        FROM_PHASE, TO_PHASE, potential, TEMPERATURE, None
    )
    assert np.isclose(direct, hydro.cosmologicalEnergyDensityFalse)


def test_radiation_energy_density_uses_raddof():
    potential = QuarticPotential()
    expected = np.pi**2 / 30 * potential.raddof * TEMPERATURE**4
    assert np.isclose(
        hydrodynamics.radiation_energy_density(potential, TEMPERATURE),
        expected,
    )
