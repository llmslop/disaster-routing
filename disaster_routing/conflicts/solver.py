from abc import ABC, abstractmethod

from .conflict_graph import ConflictGraph


class DSASolver(ABC):
    conflict_graph: ConflictGraph

    def __init__(self, conflict_graph: ConflictGraph):
        self.conflict_graph = conflict_graph

    def solve_odsa(self, perm: list[int]) -> list[int]:
        result = [0 for _ in perm]
        for i, node in enumerate(perm):
            start = max(
                (
                    result[prev] + self.conflict_graph.num_fses[prev]
                    for prev in perm[:i]
                    if self.conflict_graph.graph.has_edge(node, prev)
                ),
                default=0,
            )
            if len(result) > 0:
                start = max(start, result[-1])
            result[node] = start

        return result

    def check(self, sol: list[int]):
        graph = self.conflict_graph.graph
        num_fses = self.conflict_graph.num_fses
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                ii = set(range(sol[i], sol[i] + num_fses[i]))
                ji = set(range(sol[j], sol[j] + num_fses[j]))
                if graph.has_edge(i, j):
                    assert len(ii.intersection(ji)) == 0

    def calc_mofi(self, sol: list[int]) -> int:
        return max(a + b for a, b in zip(sol, self.conflict_graph.num_fses))

    def calc_mofi_from_perm(self, perm: list[int]) -> int:
        return self.calc_mofi(self.solve_odsa(perm))

    @abstractmethod
    def solve_for_odsa_perm(self) -> list[int]: ...

    def solve(self) -> tuple[list[int], int]:
        best = self.solve_odsa(self.solve_for_odsa_perm())
        return best, self.calc_mofi(best)
