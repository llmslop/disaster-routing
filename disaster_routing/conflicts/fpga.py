from typing import override

from .solver import DSASolver


class FPGADSASolver(DSASolver):
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
        return min(
            (self.solve_perm_with_first(start) for start in self.conflict_graph.graph),
            key=lambda sol: self.calc_mofi(sol),
        )
