"""
Analyse transition graph
========================
"""


class PhaseNode:
    def __init__(self, phase_id: int, temperature: float):
        self.phase = phase_id
        self.temperature = temperature
        self.paths = []


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
        prefix_phases = self.phases[:index]
        del self.phases[:index]

        prefix_transitions = self.transitions[:index - (0 if self.suffix_links else 1)]
        del self.transitions[:index - (1 if self.suffix_links else 0)]

        prefix_path = Path(prefix_phases, prefix_transitions, suffix_links=[self], prefix_links=self.prefix_links)
        this_prefix_path = Path(prefix_phases, prefix_transitions, suffix_links=[prefix_path], prefix_links=self.prefix_links)
        self.prefix_links = [this_prefix_path]

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
        report['transitions'] = [t.ID for t in self.transitions]
        return report


class TransitionEdge:
    def __init__(self, falsePhaseNode: PhaseNode, truePhaseNode: PhaseNode, transition, index: int):
        self.falsePhaseNode = falsePhaseNode
        self.truePhaseNode = truePhaseNode
        self.transition = transition
        self.index = index
        self.path = None
