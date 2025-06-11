from abc import ABC, abstractmethod

import networkx as nx

from .conflict_graph import ConflictGraph
from .check import check_dsa
from .odsa import solve_odsa
from .mofi import calc_mofi


class DSASolver(ABC):
    conflict_graph: ConflictGraph

    def __init__(self, conflict_graph: ConflictGraph):
        self.conflict_graph = conflict_graph

    def solve_odsa(self, perm: list[int]) -> list[int]:
        return solve_odsa(self.conflict_graph.graph, perm, self.conflict_graph.num_fses)

    def check(self, sol: list[int]):
        check_dsa(self.conflict_graph.graph, self.conflict_graph.num_fses, sol)

    def calc_mofi(self, sol: list[int]) -> int:
        return calc_mofi(self.conflict_graph.num_fses, sol)

    def calc_mofi_from_perm(self, perm: list[int]) -> int:
        return self.calc_mofi(self.solve_odsa(perm))

    @abstractmethod
    def solve_for_odsa_perm(self) -> list[int]: ...

    def solve(self) -> tuple[list[int], int]:
        best = self.solve_odsa(self.solve_for_odsa_perm())
        return best, self.calc_mofi(best)
