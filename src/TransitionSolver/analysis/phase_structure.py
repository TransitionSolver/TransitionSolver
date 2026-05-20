"""
Represent phase structure
=========================
"""

import warnings

import numpy as np
from typing import Union
from scipy import optimize

class Phase:
    """
    Represent a phase from PT data
    """
    def __init__(self, key: float, phase: np.ndarray):
        self.key = int(key)
        self.T = phase[:, 0].T
        self.V = phase[:, 1].T
        self.phi = phase[:, 2:].T

    def find_phase_at_t(self, T: float, potential) -> np.ndarray:
        if T < self.T[0] or T > self.T[-1]:
            raise RuntimeError(f'Attempted to find phase {self.key} at T={T}, while defined only for'
                f'[{self.T[0]}, {self.T[-1]}]')

        minIndex = 0
        maxIndex = len(self.T)

        # Binary search to find the boundary temperature data (specifically, the indices in the T array).
        while maxIndex - minIndex > 1:
            midpoint = (minIndex + maxIndex) // 2

            if self.T[midpoint] == T:
                return self.phi[..., midpoint]
            elif self.T[midpoint] < T:
                minIndex = midpoint
            else:
                maxIndex = midpoint

        def V(X) -> Union[float, np.ndarray]: return potential(X, T)
        offset = 0.001*potential.get_field_scale()

        # Interpolate between the most relevant data points.
        if minIndex == maxIndex:
            interpPoint = self.phi[..., minIndex]
        else:
            interpFactor = (T - self.T[minIndex]) / (self.T[maxIndex] - self.T[minIndex])
            interpPoint = self.phi[..., minIndex] + interpFactor*(self.phi[..., maxIndex] - self.phi[..., minIndex])

        direction = np.diag(np.ones(potential.get_n_scalars())*offset)
        optimisedPoint = optimize.fmin_powell(V, interpPoint, disp=False, direc=direction)

        if len(optimisedPoint.shape) == 0:
            optimisedPoint = optimisedPoint.ravel()
        
        if abs(T - self.T[minIndex]) < 0.2*potential.get_temperature_scale() and \
                np.linalg.norm(optimisedPoint - interpPoint) > 0.2*potential.get_field_scale():
            # The minimum shifted too far, so the optimiser probably found a different phase.
            # Return the previous point.
            return interpPoint

        return optimisedPoint


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
        return self.T_f is not None and self.T_f >= 0.

    def report(self):
        # add data, but in order that makes it most readable
        report = {k: v for k, v in self.items() if np.isscalar(v)}
        report['size'] = len(self.T) if self.analysed else 0
        report = report | {k: v for k, v in self.items() if not np.isscalar(v)}    
        return report

class Transition:
    """
    Represent a transition from PT data and store data about that
    transition
    """
    def __init__(self, transition: np.ndarray):
        self.key = transition[-3]
        # the ID matches up the IDs in a transition path
        # use integer for numeric sorting, stringify when outputting json Reports to avoid ambiguity in dict keys when serialized 
        self.ID = int(transition[-2])
        self.false_phase = int(transition[0])
        self.true_phase = int(transition[1])

        self.properties = TransitionProperties()
        self.properties.analysed = False
        self.properties.T_c = transition[2]
        self.properties.subcritical = bool(transition[-1] > 0)  # don't want a numpy.bool

    def report(self) -> dict:
        """
        @returns Data about transition collated into a dictionary
        """
        report = {}
        report['false_phase'] = self.false_phase
        report['true_phase'] = self.true_phase
        report['completed'] = bool(self.properties.completed)  # don't want np.bool
        report = report | self.properties.report()
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
