from typing import override

from ..random.random import Random
from .conflict_graph import ConflictGraph
from .solver import DSASolver


class FPGADSASolver(DSASolver):
    num_attempts: int
    random: Random | None

    def __init__(
        self,
        random: Random | None = None,
        num_attempts: int = 0,
    ):
        self.num_attempts = num_attempts
        self.random = random

    def solve_perm_with_first(
        self, conflict_graph: ConflictGraph, first: int
    ) -> list[int]:
        ordering = [first]
        remaining = set(range(len(conflict_graph.graph))).difference(ordering)

        start_indices = [0 for _ in range(len(conflict_graph.graph))]

        while len(remaining) > 0:
            node = min(
                remaining,
                key=lambda node: max(
                    (
                        start_indices[prev] + conflict_graph.num_fses[prev]
                        for prev in set(conflict_graph.graph.adj[node]).intersection(
                            ordering
                        )
                    ),
                    default=0,
                ),
            )
            start_indices[node] = max(
                (
                    start_indices[prev] + conflict_graph.num_fses[prev]
                    for prev in set(conflict_graph.graph.adj[node]).intersection(
                        ordering
                    )
                ),
                default=0,
            )

            remaining.remove(node)
            ordering.append(node)

        return ordering

    @override
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]:
        start_indices = list(conflict_graph.graph.nodes)
        if self.num_attempts > 0:
            assert self.random is not None
            start_indices = self.random.stdlib.choices(
                start_indices, k=self.num_attempts
            )
        return min(
            (
                self.solve_perm_with_first(conflict_graph, start)
                for start in start_indices
            ),
            key=lambda sol: self.calc_mofi(conflict_graph, sol),
        )
