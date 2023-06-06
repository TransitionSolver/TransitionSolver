from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from scipy import optimize
import traceback
if TYPE_CHECKING:
    from models import AnalysablePotential
    import TransitionAnalysis as TA


class Phase:
    key: int
    T: np.ndarray
    V: np.ndarray
    phi: np.ndarray
    n_field: int

    def __init__(self, key: float, phase: np.ndarray):
        self.key = int(key)
        self.T = phase[:, 0].T
        self.V = phase[:, 1].T
        self.phi = phase[:, 2:].T
        self.n_field = phase.shape[1] - 2

    # Returns the location of the phase at the given temperature by using interpolation between the stored data points,
    # and local optimisation from that interpolated point.
    def findPhaseAtT(self, T: float, potential: AnalysablePotential) -> np.ndarray:
        if T < self.T[0] or T > self.T[-1]:
            raise TA.InvalidTemperatureException(f'Attempted to find phase {self.key} at T={T}, while defined only for'
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

        def V(X) -> Union[float, np.ndarray]: return potential.Vtot(X, T)
        offset = 0.001*potential.fieldScale

        # Interpolate between the most relevant data points.
        if minIndex == maxIndex:
            interpPoint = self.phi[..., minIndex]
        else:
            interpFactor = (T - self.T[minIndex]) / (self.T[maxIndex] - self.T[minIndex])
            interpPoint = self.phi[..., minIndex] + interpFactor*(self.phi[..., maxIndex] - self.phi[..., minIndex])

        direction = np.diag(np.ones(potential.Ndim)*offset)
        optimisedPoint = optimize.fmin_powell(V, interpPoint, disp=False, direc=direction)

        if len(optimisedPoint.shape) == 0:
            optimisedPoint = optimisedPoint.ravel()

        if abs(T - self.T[minIndex]) < 0.2*potential.temperatureScale and \
                np.linalg.norm(optimisedPoint - interpPoint) > 0.2*potential.fieldScale:
            # The minimum shifted too far, so the optimiser probably found a different phase.
            # Return the previous point.
            return interpPoint

        return optimisedPoint


class Transition:
    analysis: Optional[TA.AnalysedTransition]
    n_field: int
    Tc: float
    Tn: float
    # The nucleation temperature calculated using Nbar, the actual number of bubbles nucleated, taking into account
    # diminishing false vacuum volume within which to nucleate.
    Tnbar: float
    Tp: float
    Te: float
    Tf: float
    # The temperature at which the physical volume of the false vacuum starts shrinking.
    TVphysDecr_high: float
    # The temperature at which the physical volume of the false vacuum stops shrinking.
    TVphysDecr_low: float
    Treh_p: float
    Treh_e: float
    Treh_f: float

    Tmin: float
    TGammaMax: float
    Teq: float
    SonTmin: float
    SonTGammaMax: float
    SonTeq: float

    decreasingVphysAtTp: bool
    decreasingVphysAtTf: bool

    key: int
    ID: int
    vw: float
    gammaTn: float
    meanBubbleRadius: float
    meanBubbleSeparation: float
    energyWeightedBubbleRadius: float
    volumeWeightedBubbleRadius: float

    transitionStrength: float

    totalNumBubbles: float
    totalNumBubblesCorrected: float

    bFoundTeq: bool
    bFoundNucleationWindow: bool

    def __init__(self, transition: np.ndarray):
        self.n_field = (len(transition)+1 - 7) // 2
        self.Tc = transition[2]
        self.true_vacuum = transition[3:3+self.n_field]
        self.false_vacuum = transition[3+self.n_field:3+self.n_field*2]
        self.key = transition[-3]
        # The ID is used to match up the indices in a transition path (which is a list of transition ids).
        self.ID = int(transition[-2])
        self.subcritical = transition[-1] > 0
        self.false_phase = int(transition[0])
        self.true_phase = int(transition[1])

        self.gammaTc = np.linalg.norm(self.true_vacuum - self.false_vacuum) / self.Tc

        self.Tn = -1.
        self.Tnbar = -1.
        self.Tp = -1.
        self.Te = -1.
        self.Tf = -1.
        self.TVphysDecr_high = -1.
        self.TVphysDecr_low = -1.
        self.Treh_p = -1
        self.Treh_e = -1
        self.Treh_f = -1

        self.Tmin = -1.
        self.TGammaMax = -1.
        self.Teq = -1.
        self.SonTmin = -1.
        self.SonTGammaMax = -1.
        self.SonTeq = -1.

        self.decreasingVphysAtTp = False
        self.decreasingVphysAtTf = False

        self.vw = -1.
        self.gammaTn = -1.
        self.analysis = None

        self.meanBubbleRadius = -1.
        self.meanBubbleSeparation = -1.
        self.energyWeightedBubbleRadius = -1.
        self.volumeWeightedBubbleRadius = -1.

        self.transitionStrength = -1.

        self.totalNumBubbles = 0.
        self.totalNumBubblesCorrected = 0.

        self.bFoundTeq = False
        self.bFoundNucleationWindow = False

    def starts(self) -> bool:
        # Changed from Tn.
        return self.Tp > 0

    def completes(self) -> bool:
        return self.Tf > 0

    def successful(self) -> bool:
        return self.Tf > 0 and self.decreasingVphysAtTf

    def halts(self) -> bool:
        return self.starts() and not self.completes()

    # Whether the completed transition leaves pockets of false vacuum.
    def leavesRemnants(self) -> bool:
        return self.completes() and not self.decreasingVphysAtTf

    # TODO: [2023] shouldn't all of these properties be in AnalysedTransition?
    def getOutputList(self) -> list[float]:
        return [self.Tn, self.Tp, self.Te, self.Tf, self.meanBubbleRadius, self.meanBubbleSeparation,
                self.energyWeightedBubbleRadius, self.volumeWeightedBubbleRadius, self.transitionStrength]

    def getReport(self, reportFileName) -> dict:
        report = {}

        report['id'] = self.ID
        report['falsePhase'] = self.false_phase
        report['truePhase'] = self.true_phase
        report['analysed'] = self.analysis is not None
        report['subcritical'] = bool(self.subcritical)

        if self.analysis is not None:
            report['completed'] = bool(self.Tf >= 0.0)
            if len(self.analysis.error) > 0: report['error'] = self.analysis.error
            # If we are updating the report file and reused its action data (i.e. if actionCurveFile == reportFileName),
            # make sure to keep that data in the report rather than create a circular reference.
            if self.analysis.actionCurveFile == '' or self.analysis.actionCurveFile == reportFileName:
                report['T'] = self.analysis.T
                report['SonT'] = self.analysis.SonT
            else:
                report['actionCurveFile'] = self.analysis.actionCurveFile

            report['vw'] = self.vw
            report['N'] = self.totalNumBubbles
            report['Nbar'] = self.totalNumBubblesCorrected

        report['Tc'] = self.Tc
        if self.Tn > 0:
            report['Tn'] = self.Tn
            report['SonTn'] = self.analysis.SonTn
        if self.Tnbar > 0:
            report['Tnbar'] = self.Tnbar
            report['SonTnbar'] = self.analysis.SonTnbar

        if self.Tp > 0: report['Tp'] = self.Tp
        if self.Te > 0: report['Te'] = self.Te
        if self.Tf > 0: report['Tf'] = self.Tf

        if self.TVphysDecr_high > 0: report['TVphysDecr_high'] = self.TVphysDecr_high
        if self.TVphysDecr_low > 0: report['TVphysDecr_low'] = self.TVphysDecr_low

        if self.Treh_p > 0: report['Treh_p'] = self.Treh_p
        if self.Treh_e > 0: report['Treh_e'] = self.Treh_e
        if self.Treh_f > 0: report['Treh_f'] = self.Treh_f

        if self.Tn > 0: report['betaTn'] = self.analysis.betaTn
        if self.Tnbar > 0: report['betaTnbar'] = self.analysis.betaTnbar
        if self.Tp > 0: report['betaTp'] = self.analysis.betaTp
        if self.Te > 0: report['betaTe'] = self.analysis.betaTe
        if self.Tf > 0: report['betaTf'] = self.analysis.betaTf

        if self.Tmin > 0:
            report['Tmin'] = self.Tmin
            report['SonTmin'] = self.SonTmin
        if self.TGammaMax > 0:
            report['TGammaMax'] = self.TGammaMax
            report['SonTGammaMax'] = self.SonTGammaMax
        #if self.Teq > 0:
        report['Teq'] = self.Teq
        #report['SonTeq'] = self.SonTeq

        if self.Tp > 0: report['decreasingVphysAtTp'] = self.decreasingVphysAtTp
        if self.Tf > 0: report['decreasingVphysAtTf'] = self.decreasingVphysAtTf

        report['foundTeq'] = self.bFoundTeq
        report['foundNucleationWindow'] = self.bFoundNucleationWindow

        if self.Tp > 0:
            report['meanBubbleSeparation'] = self.meanBubbleSeparation
            report['meanBubbleRadius'] = self.meanBubbleRadius
            report['energyWeightedBubbleRadius'] = self.energyWeightedBubbleRadius
            report['volumeWeightedBubbleRadius'] = self.volumeWeightedBubbleRadius
            report['transitionStrength'] = self.transitionStrength

        return report


class TransitionPath:
    path: list[int]

    def __init__(self, path: list[int]):
        self.path = path


class PhaseStructure:
    phases: list[Phase]
    transitions: list[Transition]
    transitionPaths: list[TransitionPath]
    reportMessage: str
    groundStateEnergyDensity: float

    def __init__(self):
        self.phases = []
        self.transitions = []
        self.transitionPaths = []
        self.reportMessage = ''
        self.groundStateEnergyDensity = 0.

    def determineGroundStateEnergyDensity(self):
        self.groundStateEnergyDensity = np.inf

        for phase in self.phases:
            if phase.T[0] == 0 and phase.V[0] < self.groundStateEnergyDensity:
                self.groundStateEnergyDensity = phase.V[0]

        if self.groundStateEnergyDensity == np.inf:
            self.groundStateEnergyDensity = 0.


def load_data(dat_name, bExpectFile=True) -> tuple[bool, Optional[PhaseStructure]]:
    try:
        with open(dat_name) as d:
            data = d.read()
    except FileNotFoundError:
        if bExpectFile:
            traceback.print_exc()
        # TODO: should probably return an empty PhaseStructure object with just its reportMessage set.
        return False, None

    parts = data.split("\n\n")
    parts = [part.split("\n") for part in parts]

    phaseStructure = PhaseStructure()

    for part in parts:
        if len(part) < 2:
            continue

        label = part[0].split(" ")
        if label[1] == "phase":
            phaseStructure.phases.append(constructPhase(part))
        elif label[1] == "transition":
            phaseStructure.transitions.append(constructTransition(part))
        elif label[1] == "transition-path":
            phaseStructure.transitionPaths.append(constructTransitionPath(part))
        else:
            raise(Exception("Invalid header for phase, transition or transition path: " + part[0]))

    phaseStructure.transitions.sort(key=lambda x: x.ID)

    phaseStructure.determineGroundStateEnergyDensity()

    return True, phaseStructure


def constructPhase(text) -> Phase:
    return Phase(text[0].split()[2], np.array([[float(string) for string in text[i].split()]
        for i in range(1, len(text))]))


def constructTransition(text) -> Transition:
    return Transition(np.array([float(string) for string in text[1].split()]))


def constructTransitionPath(text) -> TransitionPath:
    pathElements = text[1].split()
    path = np.zeros(len(pathElements), dtype='int')
    for i in range(len(path)):
        path[i] = int(pathElements[i]) if pathElements[i][0] != '-' else int(pathElements[i][1:])-1
    return TransitionPath(list(path))
    #return TransitionPath(np.array([int(string) for string in text[1].split()]))
