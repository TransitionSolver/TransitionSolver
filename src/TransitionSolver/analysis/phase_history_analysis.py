"""
Analyse phase history
=====================
"""

from __future__ import annotations

from typing import Optional

from ..models.analysable_potential import AnalysablePotential
from .phase_structure import PhaseStructure, Phase
from .transition_analysis import TransitionAnalyser, Timer
from .transition_graph import TransitionEdge, Path, PhaseNode


class PhaseHistoryAnalyser:
    bDebug: bool = False
    bPlot: bool = False
    bReportAnalysis: bool = False
    bReportPaths: bool = False
    bCheckPossibleCompletion: bool = True
    bAllowErrorsForTn: bool = True
    bAnalyseTransitionPastCompletion: bool = False
    bForcePhaseOnAxis: bool = False
    time_limit: float = 200.

    # Second return value is whether we timed out.
    def analysePhaseHistory_supplied(self, potential: AnalysablePotential, phaseStructure: PhaseStructure, vw=None, action_ct=True) -> tuple[list[Path], bool, Optional[Timer]]:  # TODO make false

        timer = Timer(self.time_limit)
        if self.bDebug:
            print('Parameter point:', potential.getParameterPoint())

        # Extract high and low temperature phases.
        phases = phaseStructure.phases
        transitions = phaseStructure.transitions
        paths = phaseStructure.paths

        # TODO: added on 23/06/2022 to handle the case where PhaseTracer reports no possible transition paths. Need to
        #  make sure PhaseTracer would have handled the case where we could stay in the same phase.
        if not paths:
            return [], False, None
       
        # Find the highest temperature.
        high_t = [phase.T[-1] for phase in phases]
        highTemp = max(high_t)
        bHighTemperaturePhase = [t == highTemp for t in high_t]

        # Find all low temperature phases. These are needed to check if a path is valid, i.e. if it terminates at a low
        # temperature phase.
        # TODO: why was it necessary to do it this way?
        bLowTemperaturePhase = [False]*len(phases)
        for i in range(len(paths)):
            if paths[i][-1] < 0:
                phase = -(paths[i][-1]+1)
            else:
                phase = transitions[paths[i][-1]].true_phase
            bLowTemperaturePhase[phase] = True

        if self.bDebug:
            print('Low temperature phases:', [i for i in range(len(phases)) if bLowTemperaturePhase[i]])

        # If there are no transitions, check if any phase is both a high and low temperature phase. This constitutes a
        # valid path, but with no transitions between phases.
        if not transitions:
            validPaths = []

            for i, p in enumerate(phases):
                if bLowTemperaturePhase[i] and bHighTemperaturePhase[i]:
                    # The temperature of this phase might as well be the highest sampled temperature.
                    validPaths.append(Path(PhaseNode(i, p.T[-1])))

            if self.bReportPaths:
                print('No transitions.')
                print('Valid paths:', validPaths)
            return validPaths, False, None

        unique = {t.properties.Tc: i for i, t in enumerate(transitions)}
        uniqueTransitionTemperatures = list(unique.keys())
        uniqueTransitionTemperatureIndices = list(unique.values())

        transitionLists = [[]] * len(uniqueTransitionTemperatures)
        # The first dimension is the phase index.
        phaseIndexedTransitions: list[list[TransitionEdge]] = [[] for _ in range(len(phases))]

        phaseNodes: list[list[PhaseNode]] = [[] for _ in range(len(phases))]

        # Construct phase nodes segmenting each phase at each unique transition temperature.
        for i in range(len(phases)):
            for j in range(len(uniqueTransitionTemperatures)):
                if phases[i].T[-1] >= uniqueTransitionTemperatures[j] >= phases[i].T[0]:
                    phaseNodes[i].append(PhaseNode(i, uniqueTransitionTemperatures[j]))

        for i in range(len(uniqueTransitionTemperatures)):
            if i == len(uniqueTransitionTemperatures)-1:
                maxIndex = len(transitions)
            else:
                maxIndex = uniqueTransitionTemperatureIndices[i+1]

            for j in range(uniqueTransitionTemperatureIndices[i], maxIndex):
                falsePhase = transitions[j].false_phase
                truePhase = transitions[j].true_phase

                falsePhaseNodeIndex = -1
                truePhaseNodeIndex = -1

                for k in range(len(phaseNodes[falsePhase])):
                    if phaseNodes[falsePhase][k].temperature == uniqueTransitionTemperatures[i]:
                        falsePhaseNodeIndex = k
                        break

                for k in range(len(phaseNodes[truePhase])):
                    if phaseNodes[truePhase][k].temperature == uniqueTransitionTemperatures[i]:
                        truePhaseNodeIndex = k
                        break

                transitionEdge = TransitionEdge(phaseNodes[falsePhase][falsePhaseNodeIndex],
                    phaseNodes[truePhase][truePhaseNodeIndex], transitions[j], len(phaseIndexedTransitions[falsePhase]))

                transitionLists[i].append(transitionEdge)

                phaseIndexedTransitions[falsePhase].append(transitionEdge)

            # Need to fill in with other possible transitions. This is what the reverse transition thing was all about!
            # TODO: do this later.

        paths = []
        frontier = []

        for i in range(len(phases)):
            if bHighTemperaturePhase[i]:
                paths.append(Path(phaseNodes[i][0]))
                phaseNodes[i][0].paths.append(paths[-1])

                # Find the highest temperature transition from this phase and add it to the frontier.
                if len(phaseIndexedTransitions[i]) > 0:
                    frontier.append(phaseIndexedTransitions[i][0])
                    phaseIndexedTransitions[i][0].path = paths[-1]

                    j = 1
                    while j < len(phaseIndexedTransitions[i]) and phaseIndexedTransitions[i][j].transition.properties.Tc ==\
                            highTemp:
                        paths.append(Path(phaseNodes[i][0]))
                        phaseNodes[i][0].paths.append(paths[-1])

                        frontier.append(phaseIndexedTransitions[i][j])
                        phaseIndexedTransitions[i][j].path = paths[-1]
                        j += 1

        frontier.sort(key=lambda tr: tr.transition.properties.Tc, reverse=True)

        while frontier:
            if timer.timeout():
                return [], True, timer

            if self.bDebug:
                print('Frontier:')
                for i in range(len(frontier)):
                    print(f'{i}: {frontier[i]}')

                print('Paths:')
                for i in range(len(paths)):
                    print(f'{i}: {paths[i]}')
            transitionEdge = frontier.pop(0)
            transition = transitionEdge.transition
            path = transitionEdge.path
            if not transition.properties.analysed:
                transition.properties.analysed = True
                if self.bReportAnalysis:
                    print(f'Analysing transition {transition.ID} ({transition.false_phase} --(T={transition.properties.Tc})-->'
                          f' {transition.true_phase})')

                Tmin = self.getMinTransitionTemperature_indexed(phases, phaseIndexedTransitions, transitionEdge)

                if path.transitions:
                    # TODO: changed on 16/11/2021. It is not correct to assume that the last transition along this path
                    #  gets us to the false phase of the current transition. If the path has been extended from the
                    #  current false vacuum by an alternative transition, then the end of the path does not correspond
                    #  to the current false vacuum.
                    #Tmax = min(transition.Tc, path.transitions[-1].Tn)

                    # Find the most recent transition that has a true phase equal to the current false phase (i.e. which
                    # transition got us to this point).
                    Tmax = transition.properties.Tc
                    for tr in path.transitions:
                        if tr.true_phase == transition.false_phase:
                            Tmax = min(Tmax, tr.properties.Tn)
                            break
                else:
                    Tmax = transition.properties.Tc

                if Tmin >= Tmax:
                    if self.bDebug:
                        print(f'Transition not possible since Tmin > Tmax ({Tmin} > {Tmax}).')
                else:
                    actionFileName = ''

                    transitionAnalyser: TransitionAnalyser = TransitionAnalyser(potential, transition.properties,
                        phases[transition.false_phase], phases[transition.true_phase],
                        phaseStructure.groud_state_energy_density, Tmin=Tmin, Tmax=Tmax, vw=vw, action_ct=action_ct)

                    transitionAnalyser.bDebug = self.bDebug
                    transitionAnalyser.bPlot = self.bPlot
                    transitionAnalyser.bReportAnalysis = self.bReportAnalysis

                    transitionAnalyser.analyseTransition(startTime=timer.start_time)

                    if timer.timeout():
                        return [], True, timer
            else:
                if self.bDebug:
                    print(f'Already analysed transition {transition.ID}')

            # If the transition begins.
            if transition.properties.Tp and transition.properties.Tf:
                frontier = [f for f in frontier if f.false_phase_node.phase != transitionEdge.false_phase_node.phase and f.transition.properties.Tc < transition.properties.Tf]

                # First we handle the false phase side. The fact that the transition happened may cause a path splitting
                # that couldn't be guaranteed before the transition was analysed.
                # If the end of the current path is not where this transition starts, then the path has been continued
                # in another direction due to a previous element in the frontier. One or two new paths need to be added
                # to account for this divergence, depending on whether the divergence occurs at the start of the path or
                # midway through, respectively.
                if transitionEdge.false_phase_node.phase != path.phases[-1].phase:
                    # Don't split the path at the first phase along the path.
                    bSplit = len(path.phases) > 2
                    if bSplit:
                        # TODO: shouldn't we search for the last occurrence of this false phase along the path? Can we
                        #  guarantee that only one phase could have been added?
                        prefix, suffix = path.splitAt(len(path.phases)-1)
                        paths.append(prefix)

                    # Create a new path for this recently handled transition.
                    prevPath = path
                    path = Path(transitionEdge.false_phase_node)
                    paths.append(path)
                    # Add this path to the phase node where this divergence occurs.
                    # TODO: changed on 15/11/2021 because pipeline/noNucleation_msScan-BP4_1/24 was breaking here.
                    #prevPath.phases[-2].paths.append(path)
                    transitionEdge.false_phase_node.paths.append(path)  # is it this simple? (added 15/11/2021)
                    #if bSplit:
                    #    paths[-2].phases[-1].paths.append(path)
                    #else:
                    #    transitionEdge.false_phase_node.paths.append(path)???

                    # If we don't split the path, we still need to handle links between the prefixes of the previous
                    # path and the new path. Whatever path is a prefix of the previous path is also a previous of the
                    # new path.
                    if not bSplit:
                        path.prefix_links = prevPath.prefix_links

                        # This new path is also the suffix of all its prefixes.
                        for prefix in path.prefix_links:
                            prefix.suffix_links.append(path)

                # Now we need to handle the true phase side. If the transition takes the path to a phase node that is
                # not currently occupied by another path, then the path can be freely extended there. If there are
                # several paths from the true phase node, then we can set this path's suffixes to all of those paths and
                # no further work is required. If there is only one path through the true phase node, we need to split
                # it at the intersection point, setting the suffix of that path as the suffix of this path.

                # We can immediately add this transition to the current path, but the true phase will only be added to
                # this path if the true phase doesn't already have a path through it. Otherwise, we can simply point to
                # the other path as a suffix, implicitly adding the true phase to this path.
                path.transitions.append(transition)

                truePhase = transitionEdge.true_phase_node

                # If the true phase has no paths through it currently, extend the path as usual and mark that this path
                # goes through the true phase. This is the only case where we check for new transitions to add to the
                # frontier. In all other cases, a suffix will handle that instead, and this path can be ignored.
                if len(truePhase.paths) == 0:
                    # Extend the path.
                    path.phases.append(transitionEdge.true_phase_node)
                    # Mark that this path goes through the true phase node.
                    truePhase.paths.append(path)
                    # Find transitions from this point to add to the frontier.
                    newFrontier = self.getNewFrontier(transitionEdge, phaseIndexedTransitions, path, True, phases)

                    # Insert the new frontier nodes into the frontier, maintaining the sorted order (i.e. decreasing
                    # critical temperature).
                    for i in range(len(newFrontier)):
                        frontier.append(newFrontier[i])

                        j = len(frontier)-1
                        while j > 0 and frontier[j].transition.properties.Tc > frontier[j-1].transition.properties.Tc:
                            temp = frontier[j]
                            frontier[j] = frontier[j-1]
                            frontier[j-1] = temp
                # If there is only one path going through the true phase node, we need to split it at the intersection
                # point unless the intersection point is the start of the other path.
                elif len(truePhase.paths) == 1:
                    # If the intersection point is the start of the other path, simply set the other path as the suffix
                    # of this path.
                    if truePhase.paths[0].phases[0] == truePhase:
                        path.suffix_links.append(truePhase.paths[0])
                        truePhase.paths[0].prefix_links.append(path)
                    # Otherwise, the other path needs to be split at the intersection point.
                    else:
                        prefix, suffix = truePhase.paths[0].split_at_node(truePhase)
                        # Add this path as a prefix of the suffix, and the suffix as a suffix of this path.
                        suffix.prefix_links.append(path)
                        path.suffix_links.append(suffix)
                # If there are already multiple paths going through the true phase node, we are guaranteed that they all
                # begin at that node (i.e. they have been split already as necessary). Simply add those paths as
                # suffixes of this path.
                else:
                    path.suffix_links += truePhase.paths

                    for suffix in truePhase.paths:
                        suffix.prefix_links.append(path)
            # TODO: I only just added this part (as of 20/01/2021). It was completely missing before. Need to make sure
            #  this does all that it needs to, and that a suffix wouldn't have already handled this frontier extension.
            # If the transition doesn't begin.
            else:
                # We only need to check the false phase side. Add the next highest temperature transition along this
                # phase to the frontier.
                newFrontier = self.getNewFrontier(transitionEdge, phaseIndexedTransitions, path, False, phases)

                # Insert the new frontier nodes into the frontier, maintaining the sorted order (i.e. decreasing
                # critical temperature).
                for i in range(len(newFrontier)):
                    frontier.append(newFrontier[i])

                    j = len(frontier) - 1
                    while j > 0 and frontier[j].transition.properties.Tc > frontier[j-1].transition.properties.Tc:
                        temp = frontier[j]
                        frontier[j] = frontier[j-1]
                        frontier[j-1] = temp

        validPaths = []

        for p in paths:
            if len(p.suffix_links) == 0 and bLowTemperaturePhase[p.phases[-1].phase]:
                p.is_valid = True
                validPaths.append(p)

        if self.bReportPaths:
            print('\nAll paths:')
            for i in range(len(paths)):
                print(f'{i}: {paths[i]}')

            print('\nValid paths:')
            for i in range(len(validPaths)):
                print(f'{i}: {validPaths[i]}')

        if self.bReportAnalysis:
            print('\nAnalysed transitions:', [transitions[i].ID for i in range(len(transitions))
                                              if transitions[i].properties.analysed is True])

        timer.save()
        return paths, False, timer

    def getNewFrontier(self, transitionEdge: TransitionEdge, phaseIndexedTransitions:
            list[list[TransitionEdge]], path: Path, bTransitionStarts: bool, phases: list[Phase])\
            -> list[TransitionEdge]:
        newFrontier: list[TransitionEdge] = []
        falsePhase: int = transitionEdge.false_phase_node.phase
        truePhase: int = transitionEdge.true_phase_node.phase

        # Find if there is a transition further down the false phase, and add it to the frontier if it could start
        # before the path transitions to a new phase either through the current transition or another transition along
        # the path.
        if transitionEdge.index < len(phaseIndexedTransitions[falsePhase])-1:
            Tc = phaseIndexedTransitions[falsePhase][transitionEdge.index+1].transition.properties.Tc
            bAlreadyTransitioned = False
            # Check the transitions already in this path, which may include a recent transition from the current false
            # phase with a pending split based on the current transition.
            for transition in path.transitions:
                if transition.properties.Tc > transitionEdge.transition.properties.Tc and transition.properties.Tf > Tc:
                    bAlreadyTransitioned = True
                    break

            # Check the current transition.
            bAlreadyTransitioned = bAlreadyTransitioned or transitionEdge.transition.properties.Tf is not None

            if not bAlreadyTransitioned:
                newFrontier.append(phaseIndexedTransitions[falsePhase][transitionEdge.index+1])
                phaseIndexedTransitions[falsePhase][transitionEdge.index+1].path = path

        # Only add transitions from the true phase if the current transition got us to the true phase.
        if bTransitionStarts:
            # Find if there is a transition further down the true phase.
            # Add all transitions from this phase above the current temperature that aren't reversed before the current
            # temperature. This represents all transitions that can occur as soon as we transition to this true phase.
            for i in range(len(phaseIndexedTransitions[truePhase])):
                newTransition = phaseIndexedTransitions[truePhase][i].transition

                # If the transition is no longer possible by the time we get to its false phase, don't consider it.
                if phases[newTransition.true_phase].T[0] > transitionEdge.transition.properties.Tn:
                    continue

                # If the new transition could have started above the temperature we get to the true phase, then we may
                # be able to add it as a subcritical transition.
                if transitionEdge.transition.properties.Tn < newTransition.properties.Tc:
                    # Make sure the transition isn't reversed, removing it as a possibility.
                    newTruePhase = newTransition.true_phase
                    bReversed = False

                    for j in range(len(phaseIndexedTransitions[newTruePhase])):
                        reverseTransition = phaseIndexedTransitions[newTruePhase][j].transition
                        # The transition must occur between the current transition's temperature (Tn) and the forward
                        # transition's temperature (Tc).
                        if reverseTransition.properties.Tc >= newTransition.properties.Tc:
                            continue
                        if reverseTransition.properties.Tc <= transitionEdge.transition.properties.Tn:
                            break
                        if reverseTransition.true_phase == truePhase:
                            bReversed = True
                            break

                    # If the transition is not reversed, then we can still follow this transition.
                    if not bReversed:
                        # Add it to the frontier.
                        newFrontier.append(phaseIndexedTransitions[truePhase][i])
                        phaseIndexedTransitions[truePhase][i].path = path
                        if self.bDebug:
                            print('Adding transition subcritically to the frontier!',
                                phaseIndexedTransitions[truePhase][i])
                #if phaseIndexedTransitions[truePhase][i].transition.Tc <= transitionEdge.transition.Tn:
                # Otherwise, we can add it as a regular transition.
                else:
                    newFrontier.append(phaseIndexedTransitions[truePhase][i])
                    phaseIndexedTransitions[truePhase][i].path = path
                    break

        return newFrontier

    def getMinTransitionTemperature_indexed(self, phases: list[Phase], phaseIndexedTransitions:
            list[list[TransitionEdge]], transitionEdge: TransitionEdge) -> float:
        Tmin = 0
        transition = transitionEdge.transition

        # Find if there is a reverse transition. This tells us that the forwards transition cannot occur below the
        # critical temperature of the reverse transition.
        for p in phaseIndexedTransitions[transition.true_phase][transitionEdge.index+1:]:
            reverse_transition = p.transition
            if reverse_transition.properties.Tc < transition.properties.Tc and reverse_transition.true_phase == transition.false_phase:
                Tmin = reverse_transition.properties.Tc
                break

        return max(Tmin, phases[transition.false_phase].T.min(), phases[transition.true_phase].T.min())

