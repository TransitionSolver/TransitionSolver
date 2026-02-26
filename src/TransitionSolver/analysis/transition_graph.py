"""
Analyse transition graph
========================
"""


class PhaseNode:
    def __init__(self, phase_id: int, temperature: float):
        self.phase = phase_id
        self.temperature = temperature
        self.paths = []

    def __str__(self) -> str:
        return f"{self.phase}({self.temperature})"

    def __repr__(self) -> str:
        return str(self)


class Path:

    def __init__(self, *phases, transitions=None, suffix_links=None, prefix_links=None):
        self.phases = list(phases)
        self.transitions = [] if transitions is None else transitions
        self.suffix_links = [] if suffix_links is None else suffix_links
        self.prefix_links = [] if prefix_links is None else prefix_links
        self.is_valid = False

    def split_at_index(self, index: int):
        """
        Truncates this path so it excludes the subpath prior to index, and constructs and returns a path consisting of the
        truncated region prior to index. This paths remains as the suffix, while a new path is created for the prefix.

        @returns Paths before and after split
        """
        # Extract first 'index' phases for the prefix (SAME)
        prefix_phases = self.phases[:index]
        # Remove those phases from current path (SAME)
        del self.phases[:index]

        # Determine how many transitions go with prefix (SAME LOGIC)
        prefix_transitions = self.transitions[:index - (0 if self.suffix_links else 1)]
        # Remove those transitions from current (SAME LOGIC)
        del self.transitions[:index - (1 if self.suffix_links else 0)]

        # Save OLD prefixes before we modify self.prefix_links
        old_prefixes = self.prefix_links

        # Create SINGLE prefix path
        prefix_path = Path(
            *prefix_phases,  # Unpack phases as positional args
            transitions=prefix_transitions,  # Pass transitions as keyword
            suffix_links=[self],  # Prefix points to suffix (self)
            prefix_links=old_prefixes,  # Prefix gets old prefixes
        )

        # Update ALL old prefixes to point to new prefix instead of self
        for p in old_prefixes:
            if self in p.suffix_links:
                # Replace self with prefix_path in each old prefix's suffix_links
                p.suffix_links = [prefix_path if s is self else s for s in p.suffix_links]

        # Update current path to point to new prefix (CORRECT - points to actual prefix_path)
        self.prefix_links = [prefix_path]

        # Return the new prefix and modified self as suffix
        return prefix_path, self
    

    def split_at_node(self, node: PhaseNode):
        for i, p in enumerate(self.phases):
            if p == node:
                return self.split_at_index(i)
        raise ValueError(f"Did not find node {node}")

    def report(self) -> dict:
        report = {}
        report['valid'] = self.is_valid
        report['phases'] = [p.phase for p in self.phases]
        report['transitions'] = [str(t.ID) for t in self.transitions]
        return report

    def customPrint(self, bPrintPrefix: bool = True, bPrintSuffix: bool = True) -> str:
        transitionIndex = 0
        outputString = ""

        if len(self.prefix_links) > 0:
            if bPrintPrefix:
                outputString += str([prefix.customPrint(bPrintPrefix=True, bPrintSuffix=False) for prefix in
                    self.prefix_links])

            # Might not have a transition yet if the path has just been created.
            if len(self.transitions) > 0:
                outputString += f" --({self.transitions[0]})--> "

            transitionIndex += 1

        for i in range(len(self.phases) - 1):
            outputString += f"{self.phases[i]} --({self.transitions[transitionIndex]})--> "
            transitionIndex += 1

        if len(self.phases) > 0:
            outputString += f"{self.phases[-1]}"

        if len(self.suffix_links) > 0:
            outputString += f" --({self.transitions[-1]})--> "

            if bPrintSuffix:
                outputString += str([suffix.customPrint(bPrintPrefix=False, bPrintSuffix=True) for suffix in
                    self.suffix_links])

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


class TransitionEdge:
    def __init__(self, false_phase_node: PhaseNode, true_phase_node: PhaseNode, transition, index: int):
        self.false_phase_node = false_phase_node
        self.true_phase_node = true_phase_node
        self.transition = transition
        self.index = index
        self.path = None
