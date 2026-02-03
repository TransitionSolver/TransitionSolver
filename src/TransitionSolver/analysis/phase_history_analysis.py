"""
Analyse phase history
=====================
"""
import math
from .transition_analysis import TransitionAnalyser, Timer
from .transition_graph import TransitionEdge, Path, PhaseNode


class PhaseHistoryAnalyser:

    def __init__(self, potential, phase_structure, time_limit=200.):
        self.potential = potential
        self.phase_structure = phase_structure
        self.time_limit = time_limit
        self.paths = []

    def report(self):
        """
        @returns Report phase history from TransitionSolver objects
        """
        report = {}
        report['valid'] = any(p.is_valid for p in self.paths)
        report['paths'] = [p.report() for p in self.paths]
        report['transitions'] = {t.ID: t.report() for t in self.phase_structure.transitions}
        return report

    @property
    def unique_transition_temperatures(self):
        temps = []
        seen = set()
        for t in self.phase_structure.transitions:
            Tc = t.properties.T_c
            if Tc not in seen:
                seen.add(Tc)
                temps.append(Tc)
        return temps

    @property
    def unique_transition_idx(self):
        idx = []
        seen = set()
        for i, t in enumerate(self.phase_structure.transitions):
            Tc = t.properties.T_c
            if Tc not in seen:
                seen.add(Tc)
                idx.append(i)
        return idx


    @property
    def is_high_temperature_phase(self):
        """
        @returns Find all high temperature phases
        """
        highest_temperatures = [phase.T[-1] for phase in self.phase_structure.phases]
        max_temperature = max(highest_temperatures)
        return [
            temp == max_temperature for temp in highest_temperatures
        ]

    @property
    def is_low_temperature_phase(self):
        """
        Find all low temperature phases. These are needed to check if a path is valid, i.e. if it terminates at a low
        temperature phase.
        """
        is_low_temperature_phase = [False] * len(self.phase_structure.phases)

        for path in self.phase_structure.paths:
            last_step = path[-1]
            if last_step < 0:
                phase_index = -(last_step + 1)
            else:
                phase_index = self.phase_structure.transitions[last_step].true_phase
            is_low_temperature_phase[phase_index] = True

        return is_low_temperature_phase

    def phase_indexed_trans(self, phase_nodes):
        # The first dimension is the phase index.
        phase_indexed_trans: list[list[TransitionEdge]] = [[] for _ in range(len(self.phase_structure.phases))]

        for i, ut in enumerate(self.unique_transition_temperatures):
            if i == len(self.unique_transition_temperatures) - 1:
                max_idx = len(self.phase_structure.transitions)
            else:
                max_idx = self.unique_transition_idx[i + 1]

            for j in range(self.unique_transition_idx[i], max_idx):
                false_phase = self.phase_structure.transitions[j].false_phase
                true_phase = self.phase_structure.transitions[j].true_phase

                false_phase_node_index = next((k for k, node in enumerate(phase_nodes[false_phase]) if node.temperature == ut), None)
                true_phase_node_index = next((k for k, node in enumerate(phase_nodes[true_phase]) if node.temperature == ut), None)

                if false_phase_node_index is None or true_phase_node_index is None:
                    raise KeyError(f"Missing PhaseNode at T_c={ut} for false={false_phase}, true={true_phase}")

                transition_edge = TransitionEdge(phase_nodes[false_phase][false_phase_node_index],
                                                 phase_nodes[true_phase][true_phase_node_index], self.phase_structure.transitions[j], len(phase_indexed_trans[false_phase]))

                phase_indexed_trans[false_phase].append(transition_edge)

        return phase_indexed_trans
    
    def phase_nodes(self):
        nodes = []
        for i, p in enumerate(self.phase_structure.phases):
            Tmin, Tmax = p.T[0], p.T[-1]   # assumes sorted
            nodes_i = [PhaseNode(i, t) for t in self.unique_transition_temperatures if Tmax >= t >= Tmin]
            nodes.append(nodes_i)
        return nodes

    # find "trivial paths" with no transition that *may* exist in the cosmological phase history 
    def trivial_paths(self):
        paths = []
        for i, phase in enumerate(self.phase_structure.phases):
            if not (self.is_high_temperature_phase[i] and self.is_low_temperature_phase[i]):
                continue # ignore phases not present at highest and lowest temperatures

            Tmax = phase.T[-1]  # assumes phase.T is sorted ascending
            start_node = PhaseNode(i, Tmax)
            paths.append(Path(start_node))

        return paths

    # finds starting point for phase history analysis
    # Initializes the search for phase transition paths:
    # - Creates an initial path for each phase existing at the highest temperature
    # - For each such phase, identifies its highest-temperature transition(s)
    # - Degenerate case: multiple transitions at the same temperature create separate branching paths
    # - Builds a frontier queue of transitions to explore, sorted by temperature (highest first)
    # Returns: (initial_paths, frontier_queue)
    def init_frontier(self, phase_indexed_trans, phase_nodes):
        paths = []
        frontier = []

        for i in range(len(self.phase_structure.phases)):
            # phases that exist at the highest temperature are valid starting points
            if self.is_high_temperature_phase[i]:
                paths.append(Path(phase_nodes[i][0]))
                phase_nodes[i][0].paths.append(paths[-1])

                # Find the highest temperature transition from this phase and add it to the frontier.
                if len(phase_indexed_trans[i]) > 0:
                    frontier.append(phase_indexed_trans[i][0])
                    phase_indexed_trans[i][0].path = paths[-1]

                    j = 1
                    phase_max_Tc = phase_indexed_trans[i][0].transition.properties.T_c
                    while j < len(phase_indexed_trans[i]) and math.isclose(
                            phase_indexed_trans[i][j].transition.properties.T_c,
                            phase_max_Tc, rel_tol=0.0, abs_tol=1e-12):

                        paths.append(Path(phase_nodes[i][0]))
                        phase_nodes[i][0].paths.append(paths[-1])

                        frontier.append(phase_indexed_trans[i][j])
                        phase_indexed_trans[i][j].path = paths[-1]
                        j += 1

        frontier.sort(key=lambda tr: tr.transition.properties.T_c, reverse=True)
        return paths, frontier

    def analyse_transition(self, phase_indexed_trans, transition_edge, path, transition, **kwargs):
        if transition.properties.analysed:
            return

        Tmin = self.min_trans_temperature_idxed(phase_indexed_trans, transition_edge)
        Tmax = self.max_trans_temperature(path, transition)

        if Tmin < Tmax:
            transition.properties.analysed = True
            ta = TransitionAnalyser(self.potential, transition.properties,
                                                    self.phase_structure.phases[transition.false_phase], self.phase_structure.phases[transition.true_phase],
                                                    self.phase_structure.groud_state_energy_density, Tmin=Tmin, Tmax=Tmax, **kwargs)
            ta.analyse()
        else:
            transition.properties.analysed = False
            transition.properties.error = "T_MAX < T_MIN"

    def analyse(self, bubble_wall_velocity=None, action_ct=True):  # TODO make false

        timer = Timer(self.time_limit)
        # if there are no paths in phase_structure, set paths empty and return 
        if not self.phase_structure.paths:
            self.paths = []
            return

        # If there are no transitions, check if any phase is both a high and low temperature phase.
        # This constitutes a valid path, but with no transitions between phases.
        if not self.phase_structure.transitions:
            self.paths = self.trivial_paths()
            return

        # Construct phase nodes segmenting each phase at each unique transition temperature.
        phase_nodes = self.phase_nodes()
        phase_indexed_trans = self.phase_indexed_trans(phase_nodes)
        paths, frontier = self.init_frontier(phase_indexed_trans, phase_nodes)

        while frontier:
            if timer.timeout():
                self.paths = []
                return

            transition_edge = frontier.pop(0)
            transition = transition_edge.transition
            path = transition_edge.path

            self.analyse_transition(phase_indexed_trans, transition_edge, path, transition, bubble_wall_velocity=bubble_wall_velocity, action_ct=action_ct)

            if timer.timeout():
                self.paths = []
                return

            # If the transition completes.
            if transition.properties.T_p is not None  and transition.properties.T_f is not None:
                # Guard against NaN silently passing through.
                if math.isnan(transition.properties.T_p) or math.isnan(transition.properties.T_f):
                    raise ValueError(f"Invalid milestone temperature(s): T_p={Tp}, T_f={Tf} for transition ID={transition.ID}")
                Tf = transition.properties.T_f
                # discard histories involving transitions out of the phase which sis the false vacuum in the transition that completed
                # whenever the critical temperatrure is below the completion temperature of the transition that completed.
                # such histories cannot be in the full cosmoplgical history.
                # note this pruning assumes we never return to this phase with a later transition, so we
                # we don't handle such cases which is reasonable but we should keep this in mind
                frontier = [
                    f for f in frontier
                    if not (f.false_phase_node.phase == transition_edge.false_phase_node.phase
                            and f.transition.properties.T_c < Tf)
                ]



                # First we handle the false phase side. The fact that the transition happened may cause a path splitting
                # that couldn't be guaranteed before the transition was analysed.
                # If the end of the current path is not where this transition starts, then the path has been continued
                # in another direction due to a previous element in the frontier. One or two new paths need to be added
                # to account for this divergence, depending on whether the divergence occurs at the start of the path or
                # midway through, respectively.
                if transition_edge.false_phase_node.phase != path.phases[-1].phase:
                    # Don't split the path at the first phase along the path.
                    split_path = len(path.phases) > 2

                    if split_path:
                        # TODO: shouldn't we search for the last occurrence of this false phase along the path? Can we
                        #  guarantee that only one phase could have been added?
                        prefix, suffix = path.split_at_index(len(path.phases)-1)
                        paths.append(prefix)

                    # Create a new path for this recently handled transition.
                    previous_path = path
                    path = Path(transition_edge.false_phase_node)
                    paths.append(path)
                    # Add this path to the phase node where this divergence occurs.
                    transition_edge.false_phase_node.paths.append(path)

                    # If we don't split the path, we still need to handle links between the prefixes of the previous
                    # path and the new path. Whatever path is a prefix of the previous path is also a previous of the
                    # new path.
                    if not split_path:
                        path.prefix_links = previous_path.prefix_links

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

                true_phase = transition_edge.true_phase_node

                # If the true phase has no paths through it currently, extend the path as usual and mark that this path
                # goes through the true phase. This is the only case where we check for new transitions to add to the
                # frontier. In all other cases, a suffix will handle that instead, and this path can be ignored.
                if not true_phase.paths:
                    # Extend the path.
                    path.phases.append(transition_edge.true_phase_node)
                    # Mark that this path goes through the true phase node.
                    true_phase.paths.append(path)
                    # Find transitions from this point to add to the frontier.
                    # Insert the new frontier nodes into the frontier, maintaining the sorted order (i.e. decreasing
                    # critical temperature).
                    frontier += self.new_frontier(transition_edge, phase_indexed_trans, path, True)
                    frontier.sort(key=lambda tr: tr.transition.properties.T_c, reverse=True)

                # If there is only one path going through the true phase node, we need to split it at the intersection
                # point unless the intersection point is the start of the other path.
                elif len(true_phase.paths) == 1:
                    # If the intersection point is the start of the other path, simply set the other path as the suffix
                    # of this path.
                    if true_phase.paths[0].phases[0] == true_phase:
                        path.suffix_links.append(true_phase.paths[0])
                        true_phase.paths[0].prefix_links.append(path)
                    # Otherwise, the other path needs to be split at the intersection point.
                    else:
                        prefix, suffix = true_phase.paths[0].split_at_node(true_phase)
                        # Add this path as a prefix of the suffix, and the suffix as a suffix of this path.
                        suffix.prefix_links.append(path)
                        path.suffix_links.append(suffix)
                # If there are already multiple paths going through the true phase node, we are guaranteed that they all
                # begin at that node (i.e. they have been split already as necessary). Simply add those paths as
                # suffixes of this path.
                else:
                    path.suffix_links += true_phase.paths

                    for suffix in true_phase.paths:
                        suffix.prefix_links.append(path)
            # TODO: I only just added this part (as of 20/01/2021). It was completely missing before. Need to make sure
            #  this does all that it needs to, and that a suffix wouldn't have already handled this frontier extension.
            # If the transition doesn't begin.
            else:
                # We only need to check the false phase side. Add the next highest temperature transition along this
                # phase to the frontier.
                # Insert the new frontier nodes into the frontier, maintaining the sorted order (i.e. decreasing
                # critical temperature).
                frontier += self.new_frontier(transition_edge, phase_indexed_trans, path, False)
                frontier.sort(key=lambda tr: tr.transition.properties.T_c, reverse=True)

        self.paths = paths
        self.set_is_valid()

    def set_is_valid(self):
        for p in self.paths:
            p.is_valid = not p.suffix_links and self.is_low_temperature_phase[p.phases[-1].phase]

    def new_frontier(self, transition_edge: TransitionEdge, phase_indexed_trans:
                     list[list[TransitionEdge]], path: Path, transition_starts: bool)\
            -> list[TransitionEdge]:
        new_frontier: list[TransitionEdge] = []
        false_phase: int = transition_edge.false_phase_node.phase
        true_phase: int = transition_edge.true_phase_node.phase

        # Find if there is a transition further down the false phase, and add it to the frontier if it could start
        # before the path transitions to a new phase either through the current transition or another transition along
        # the path.
        if transition_edge.index < len(phase_indexed_trans[false_phase])-1:
            Tc = phase_indexed_trans[false_phase][transition_edge.index+1].transition.properties.T_c
            already_transitioned = False
            # Check the transitions already in this path, which may include a recent transition from the current false
            # phase with a pending split based on the current transition.
            for transition in path.transitions:
                if transition.properties.T_c > transition_edge.transition.properties.T_c and transition.properties.T_f > Tc:
                    already_transitioned = True
                    break

            # Check the current transition.
            already_transitioned = already_transitioned or transition_edge.transition.properties.T_f is not None

            if not already_transitioned:
                new_frontier.append(phase_indexed_trans[false_phase][transition_edge.index+1])
                phase_indexed_trans[false_phase][transition_edge.index+1].path = path

        # Only add transitions from the true phase if the current transition got us to the true phase.
        if transition_starts:
            # Find if there is a transition further down the true phase.
            # Add all transitions from this phase above the current temperature that aren't reversed before the current
            # temperature. This represents all transitions that can occur as soon as we transition to this true phase.
            for i in range(len(phase_indexed_trans[true_phase])):
                new_transition = phase_indexed_trans[true_phase][i].transition

                # If the transition is no longer possible by the time we get to its false phase, don't consider it.
                if self.phase_structure.phases[new_transition.true_phase].T[0] > transition_edge.transition.properties.T_n:
                    continue

                # If the new transition could have started above the temperature we get to the true phase, then we may
                # be able to add it as a subcritical transition.
                if transition_edge.transition.properties.T_n < new_transition.properties.T_c:
                    # Make sure the transition isn't reversed, removing it as a possibility.
                    newtrue_phase = new_transition.true_phase
                    reversed_ = False

                    for j in range(len(phase_indexed_trans[newtrue_phase])):
                        reversed_transition = phase_indexed_trans[newtrue_phase][j].transition
                        # The transition must occur between the current transition's temperature (Tn) and the forward
                        # transition's temperature (Tc).
                        if reversed_transition.properties.T_c >= new_transition.properties.T_c:
                            continue
                        if reversed_transition.properties.T_c <= transition_edge.transition.properties.T_n:
                            break
                        if reversed_transition.true_phase == true_phase:
                            reversed_ = True
                            break

                    # If the transition is not reversed, then we can still follow this transition.
                    if not reversed_:
                        # Add it to the frontier.
                        new_frontier.append(phase_indexed_trans[true_phase][i])
                        phase_indexed_trans[true_phase][i].path = path

                # if phase_indexed_trans[true_phase][i].transition.Tc <= transition_edge.transition.Tn:
                # Otherwise, we can add it as a regular transition.
                else:
                    new_frontier.append(phase_indexed_trans[true_phase][i])
                    phase_indexed_trans[true_phase][i].path = path
                    break

        return new_frontier

    def max_trans_temperature(self, path, transition):
        if path.transitions:
            # Find the most recent transition that has a true phase equal to the current false phase (i.e. which
            # transition got us to this point).
            for tr in reversed(path.transitions):
                if tr.true_phase == transition.false_phase and tr.properties.T_n is not None:
                    return min(transition.properties.T_c, tr.properties.T_n)
        return transition.properties.T_c

    def min_trans_temperature_idxed(self, phase_indexed_trans:
                                    list[list[TransitionEdge]], transition_edge: TransitionEdge) -> float:
        Tmin = 0

        # Find if there is a reverse transition. This tells us that the forwards transition cannot occur below the
        # critical temperature of the reverse transition.
        for p in phase_indexed_trans[transition_edge.transition.true_phase][transition_edge.index+1:]:
            reverse_transition = p.transition
            if reverse_transition.properties.T_c < transition_edge.transition.properties.T_c and reverse_transition.true_phase == transition_edge.transition.false_phase:
                Tmin = reverse_transition.properties.T_c
                break

        return max(Tmin, self.phase_structure.phases[transition_edge.transition.false_phase].T.min(), self.phase_structure.phases[transition_edge.transition.true_phase].T.min())
