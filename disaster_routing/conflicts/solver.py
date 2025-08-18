from abc import ABC, abstractmethod

from .conflict_graph import ConflictGraph


class DSASolver(ABC):
    @staticmethod
    def solve_odsa(conflict_graph: ConflictGraph, perm: list[int]) -> dict[int, int]:
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

    @staticmethod
    def check(conflict_graph: ConflictGraph, sol: dict[int, int]):
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

    @staticmethod
    def calc_mofi(conflict_graph: ConflictGraph, sol: dict[int, int]) -> int:
        return max(
            sol[n] + conflict_graph.num_fses[n] for n in conflict_graph.graph.nodes
        )

    @staticmethod
    def calc_mofi_from_perm(conflict_graph: ConflictGraph, perm: list[int]) -> int:
        return DSASolver.calc_mofi(
            conflict_graph, DSASolver.solve_odsa(conflict_graph, perm)
        )

    @abstractmethod
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]: ...

    @abstractmethod
    def name(self) -> str: ...

    def solve(self, conflict_graph: ConflictGraph) -> tuple[dict[int, int], int]:
        best = DSASolver.solve_odsa(
            conflict_graph, self.solve_for_odsa_perm(conflict_graph)
        )
        return best, DSASolver.calc_mofi(conflict_graph, best)
