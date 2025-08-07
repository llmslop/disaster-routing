from abc import ABC, abstractmethod

from .conflict_graph import ConflictGraph


class DSASolver(ABC):
    def solve_odsa(self, conflict_graph: ConflictGraph, perm: list[int]) -> list[int]:
        result = [0 for _ in perm]
        for i, node in enumerate(perm):
            start = max(
                (
                    result[prev] + conflict_graph.num_fses[prev]
                    for prev in perm[:i]
                    if conflict_graph.graph.has_edge(node, prev)
                ),
                default=0,
            )
            if len(result) > 0:
                start = max(start, result[-1])
            result[node] = start

        return result

    def check(self, conflict_graph: ConflictGraph, sol: list[int]):
        graph = conflict_graph.graph
        num_fses = conflict_graph.num_fses
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                ii = set(range(sol[i], sol[i] + num_fses[i]))
                ji = set(range(sol[j], sol[j] + num_fses[j]))
                if graph.has_edge(i, j):
                    assert len(ii.intersection(ji)) == 0

    def calc_mofi(self, conflict_graph: ConflictGraph, sol: list[int]) -> int:
        return max(a + b for a, b in zip(sol, conflict_graph.num_fses))

    def calc_mofi_from_perm(
        self, conflict_graph: ConflictGraph, perm: list[int]
    ) -> int:
        return self.calc_mofi(conflict_graph, self.solve_odsa(conflict_graph, perm))

    @abstractmethod
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]: ...

    def solve(self, conflict_graph: ConflictGraph) -> tuple[list[int], int]:
        best = self.solve_odsa(conflict_graph, self.solve_for_odsa_perm(conflict_graph))
        return best, self.calc_mofi(conflict_graph, best)
