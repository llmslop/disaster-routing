from abc import ABC, abstractmethod
from collections import defaultdict

from .conflict_graph import ConflictGraph


class DSASolver(ABC):
    def solve_odsa(
        self, conflict_graph: ConflictGraph, perm: list[int]
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        for i, node in enumerate(perm):
            start = max(
                (
                    result[prev] + conflict_graph.num_fses[prev]
                    for prev in set(conflict_graph.graph.adj[node]).intersection(
                        perm[:i]
                    )
                ),
                default=0,
            )
            if i > 0:
                start = max(start, result[perm[i - 1]])
            result[node] = start

        return result

    def check(self, conflict_graph: ConflictGraph, sol: dict[int, int]):
        graph = conflict_graph.graph
        num_fses = conflict_graph.num_fses
        for i in graph.nodes:
            for j in graph.nodes:
                if i >= j:
                    continue
                ii = set(range(sol[i], sol[i] + num_fses[i]))
                ji = set(range(sol[j], sol[j] + num_fses[j]))
                if graph.has_edge(i, j):
                    assert len(ii.intersection(ji)) == 0

    def calc_mofi(self, conflict_graph: ConflictGraph, sol: dict[int, int]) -> int:
        return max(
            sol[n] + conflict_graph.num_fses[n] for n in conflict_graph.graph.nodes
        )

    def calc_mofi_from_perm(
        self, conflict_graph: ConflictGraph, perm: list[int]
    ) -> int:
        return self.calc_mofi(conflict_graph, self.solve_odsa(conflict_graph, perm))

    @abstractmethod
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]: ...

    def solve(self, conflict_graph: ConflictGraph) -> tuple[dict[int, int], int]:
        best = self.solve_odsa(conflict_graph, self.solve_for_odsa_perm(conflict_graph))
        return best, self.calc_mofi(conflict_graph, best)
