"""
Represent phase structure
=========================
"""

import warnings

import numpy as np
from scipy.optimize import fmin_powell


class Phase:
    """
    Represent a phase from PT data
    """
    def __init__(self, key: float, phase: np.ndarray):
        self.key = int(key)
        self.T = phase[:, 0].T
        self.V = phase[:, 1].T
        self.phi = phase[:, 2:].T

    def find_phase_at_t(self, T: float, potential, rel_tol=10) -> np.ndarray:
        """
        @returns Location of the phase at the given temperature by using interpolation between the stored data points,
        and local optimisation from that interpolated point
        """
        if T < self.T[0] or T > self.T[-1]:
            raise ValueError(f'Attempted to find phase {self.key} at T={T}, while defined only for'
                f'[{self.T[0]}, {self.T[-1]}]')

        idx = np.absolute(self.T - T).argmin()

        if self.T[idx] == T:
            return self.phi[..., idx]

        nxt_idx = idx - 1 if self.T[idx] > T else idx + 1
        dphi = self.phi[..., idx] - self.phi[..., nxt_idx]
        dt = self.T[idx] - self.T[nxt_idx]
        linear_interp = self.phi[..., idx] + (T - self.T[idx]) * dphi / dt
        direc = 0.5 * np.diag(dphi)

        optimised_point = fmin_powell(lambda X: potential(X, T), linear_interp, disp=False, direc=direc)

        if np.linalg.norm(optimised_point - linear_interp) > rel_tol * np.linalg.norm(dphi):
            warnings.warn("Using linear interpolation as the optimiser probably found a different phase")
            return linear_interp

        return np.atleast_1d(optimised_point)


class TransitionProperties(dict):
    """
    Dict-like object of transition properties
    """
    DEFAULT = None

    def __getattr__(self, key):
        return self.get(key, self.DEFAULT)

    def __setattr__(self, key, value):
        self[key] = value

    @property
    def completed(self):
        return self.Tf is not None and self.Tf >= 0.

    @property
    def meanBubbleSeparationArray(self):
        return [d**(-1/3) if d != 0 else 0 for d in self.bubbleNumberDensity]



class Transition:
    """
    Represent a transition from PT data and store data about that
    transition
    """
    def __init__(self, transition: np.ndarray):
        self.key = transition[-3]
        # The ID is used to match up the indices in a transition path (which is a list of transition ids)
        self.ID = int(transition[-2])
        self.false_phase = int(transition[0])
        self.true_phase = int(transition[1])

        self.properties = TransitionProperties()
        self.properties.subcritical = transition[-1] > 0
        self.properties.Tc = transition[2]
        self.properties.analysed = False

    def report(self) -> dict:
        """
        @returns Data about transition collated into a dictionary
        """
        report = self.properties.copy()
        report['id'] = self.ID
        report['falsePhase'] = self.false_phase
        report['truePhase'] = self.true_phase
        report['completed'] = self.properties.completed
        return report

    def __str__(self) -> str:
        return str(self.ID)

    def __repr__(self) -> str:
        return str(self)


class PhaseStructure:

    def __init__(self, phases=None, transitions=None, paths=None):
        self.phases = [] if not phases else phases
        self.transitions = [] if not transitions else transitions
        self.paths = [] if not paths else paths
        self.transitions.sort(key=lambda x: x.ID)

    @property
    def groud_state_energy_density(self):
        """
        @returns The lowest energy of any phase at T = 0
        """
        groud_state_energy_density = np.inf

        for phase in self.phases:
            if phase.T[0] == 0 and phase.V[0] < groud_state_energy_density:
                groud_state_energy_density = phase.V[0]

        if np.isinf(groud_state_energy_density):
            warnings.warn("Could not determine ground state energy density; assuming 0")
            groud_state_energy_density = 0.

        return groud_state_energy_density
