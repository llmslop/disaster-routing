from random import choices
from typing import override

from .conflict_graph import ConflictGraph
from .solver import DSASolver


class FPGADSASolver(DSASolver):
    num_attempts: int

    def __init__(self, conflict_graph: ConflictGraph, num_attempts: int):
        super().__init__(conflict_graph)
        self.num_attempts = num_attempts

    def solve_perm_with_first(self, first: int) -> list[int]:
        ordering = [first]
        remaining = set(range(len(self.conflict_graph.graph))).difference(ordering)

        start_indices = [0 for _ in range(len(self.conflict_graph.graph))]

        while len(remaining) > 0:
            node = min(
                remaining,
                key=lambda node: max(
                    (
                        start_indices[prev] + self.conflict_graph.num_fses[prev]
                        for prev in ordering
                        if self.conflict_graph.graph.has_edge(node, prev)
                    ),
                    default=0,
                ),
            )
            start_indices[node] = max(
                (
                    start_indices[prev] + self.conflict_graph.num_fses[prev]
                    for prev in ordering
                    if self.conflict_graph.graph.has_edge(node, prev)
                ),
                default=0,
            )

            remaining.remove(node)
            ordering.append(node)

        return ordering

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        start_indices = list(self.conflict_graph.graph.nodes)
        if self.num_attempts > 0:
            start_indices = choices(start_indices, k=self.num_attempts)
        return min(
            (self.solve_perm_with_first(start) for start in start_indices),
            key=lambda sol: self.calc_mofi(sol),
        )
