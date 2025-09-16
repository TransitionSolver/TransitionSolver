"""
Analyse transition graph
========================
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import phase_structure as PS

class Path:
    phases: list[PS.Phase]
    transitions: list[PS.Transition]
    pathPrefixLinks: list['Path']
    pathSuffixLinks: list['Path']

    def __init__(self, phase: PS.Phase):
        self.phases = [phase]
        self.transitions = []
        self.pathPrefixLinks = []
        self.pathSuffixLinks = []

    def customPrint(self, bPrintPrefix: bool = True, bPrintSuffix: bool = True) -> str:
        transitionIndex = 0
        outputString = ""

        if len(self.pathPrefixLinks) > 0:
            if bPrintPrefix:
                outputString += str([prefix.customPrint(bPrintPrefix=True, bPrintSuffix=False) for prefix in
                    self.pathPrefixLinks])

            # Might not have a transition yet if the path has just been created.
            if len(self.transitions) > 0:
                outputString += f" --({self.transitions[0]})--> "

            transitionIndex += 1

        for i in range(len(self.phases) - 1):
            outputString += f"{self.phases[i]} --({self.transitions[transitionIndex]})--> "
            transitionIndex += 1

        if len(self.phases) > 0:
            outputString += f"{self.phases[-1]}"

        if len(self.pathSuffixLinks) > 0:
            outputString += f" --({self.transitions[-1]})--> "

            if bPrintSuffix:
                outputString += str([suffix.customPrint(bPrintPrefix=False, bPrintSuffix=True) for suffix in
                    self.pathSuffixLinks])

        return outputString

    def __str__(self) -> str:
        """if len(self.transitions) == 0 and self.pathLink is not None:
            return str(self.pathLink)

        outputString = str(self.phases[0])

        maxIndex = len(self.transitions) - (0 if self.pathLink is None else 1)

        for i in range(maxIndex):
            outputString += " --(" + str(self.transitions[i]) + ")--> " + str(self.phases[i+1])

        if self.pathLink is not None:
            outputString += str(self.pathLink)"""

        return self.customPrint(bPrintPrefix=True, bPrintSuffix=True)

    def __repr__(self) -> str:
        return str(self)


# TODO: [2023] figure out if we really need Node, ProperPhaseNode and FrontierNode. If so, document their uses and
#  differences.
class Node:
    transitions: list[PS.Transition]

    def __init__(self):
        self.transitions = []

    def sortTransitions(self, transitions):
        self.transitions.sort(key=lambda index: transitions[index].Tc, reverse=True)


class OptimalFrontierNode:
    transition: PS.Transition
    path: Path

    def __init__(self, transition: PS.Transition, path: Path):
        self.transition = transition
        self.path = path

    def __str__(self) -> str:
        return f'{self.transition}({self.path})'

    def __repr__(self) -> str:
        return str(self)


class ProperPhaseNode:
    phase: int  # index in phases list (not necessarily phase id?) TODO: [2023]
    temperature: float
    paths: list['ProperPath']

    def __init__(self, phase: int, temperature: float):
        self.phase = phase
        self.temperature = temperature
        self.paths = []

    def __str__(self) -> str:
        return f'{self.phase}({self.temperature})'

    def __repr__(self) -> str:
        return str(self)


class ProperPath:
    phases: list[ProperPhaseNode]
    transitions: list[PS.Transition]
    suffixLinks: list['ProperPath']
    prefixLinks: list['ProperPath']
    bValid: bool

    def __init__(self, phase: ProperPhaseNode):
        self.phases = [phase]
        self.transitions = []
        self.suffixLinks = []
        self.prefixLinks = []
        self.bValid = False

    def __str__(self) -> str:
        return self.customPrint(bPrefix=True, bSuffix=True)

    def __repr__(self) -> str:
        return str(self)

    # Truncates this path so it excludes the subpath prior to index, and constructs and returns a path consisting of the
    # truncated region prior to index. This paths remains as the suffix, while a new path is created for the prefix.
    def splitAt(self, index: int) -> tuple['ProperPath', 'ProperPath']:
        prefixPhases = self.phases[:index]
        prefixTransitions = self.transitions[:index - (0 if len(self.suffixLinks) > 0 else 1)]

        del self.phases[:index]
        del self.transitions[:index - (1 if len(self.suffixLinks) > 0 else 0)]

        prefixPath = ProperPath(prefixPhases[0])
        prefixPath.phases = prefixPhases
        prefixPath.transitions = prefixTransitions

        prefixPath.prefixLinks = self.prefixLinks
        self.prefixLinks = [prefixPath]
        prefixPath.suffixLinks.append(self)

        # Any path that was a prefix of this total path should now point to the prefix, not the suffix.
        #for prefixLink in prefixPath.prefixLinks:
        #    prefixLink.suffixLinks.remove(self)
        #    prefixLink.suffixLinks.append(prefixPath)
        for prefixLink in self.prefixLinks:
            prefixLink.suffixLinks.remove(self)

        for prefixLink in self.prefixLinks:
            prefixLink.suffixLinks.append(prefixPath)

        return prefixPath, self

    def splitAtNode(self, node: ProperPhaseNode) -> tuple['ProperPath', 'ProperPath']:
        for i in range(len(self.phases)):
            if self.phases[i] == node:
                return self.splitAt(i)

        # TODO: [2023] make this a custom exception.
        raise Exception("<ProperPath.splitAtNode> Didn't find node!")

    def customPrint(self, bPrefix: bool = True, bSuffix: bool = True) -> str:
        outputString = ''

        if bPrefix and len(self.prefixLinks) > 0:
            outputString += '[' + ', '.join([prefix.customPrint(bPrefix=True, bSuffix=False) for prefix in
                self.prefixLinks]) + '] '

        for i in range(len(self.transitions)):
            outputString += f'{self.phases[i]} --({self.transitions[i].ID}) --> '

        if len(self.suffixLinks) == 0:
            outputString += f'{self.phases[-1]}'
        elif bSuffix:
            outputString += '[' + ', '.join([suffix.customPrint(bPrefix=False, bSuffix=True) for suffix in
                self.suffixLinks]) + ']'

        return outputString

    def getAllPaths(self, suffix: str = '', paths: Optional[list[str]] = None) -> list[str]:
        if paths is None:
            paths = []

        suffix = self.customPrint(bPrefix=False, bSuffix=False) + suffix

        if len(self.prefixLinks) == 0:
            paths.append(suffix)
            return paths

        for prefix in self.prefixLinks:
            paths = prefix.getAllPaths(suffix, paths)

        return paths

    def getReport(self) -> dict:
        report = {}
        report['valid'] = self.bValid
        report['phases'] = [p.phase for p in self.phases]
        if len(self.transitions) > 0: report['transitions'] = [t.ID for t in self.transitions]
        return report


class ProperTransitionEdge:
    falsePhaseNode: ProperPhaseNode
    truePhaseNode: ProperPhaseNode
    transition: PS.Transition
    index: int
    path: Optional[ProperPath]

    def __init__(self, falsePhaseNode: ProperPhaseNode, truePhaseNode: ProperPhaseNode, transition: PS.Transition,
            index: int):
        self.falsePhaseNode = falsePhaseNode
        self.truePhaseNode = truePhaseNode
        self.transition = transition
        self.index = index
        self.path = None

    def __str__(self) -> str:
        return f'{self.falsePhaseNode} --({self.transition.ID})--> {self.truePhaseNode}'

    def __repr__(self) -> str:
        return str(self)


class FrontierNode:
    transitionPathIndex: int
    pathIndex: int
    Tc: float

    def __init__(self, transitionPathIndex: int, pathIndex: int, Tc: float):
        self.transitionPathIndex = transitionPathIndex
        self.pathIndex = pathIndex
        self.Tc = Tc

    def __lt__(self, other) -> bool:
        return self.Tc > other.Tc
