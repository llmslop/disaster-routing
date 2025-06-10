from abc import ABC, abstractmethod

import networkx as nx


from .check import check_dsa
from .odsa import solve_odsa
from .mofi import calc_mofi


class DSASolver(ABC):
    graph: nx.Graph
    num_fses: list[int]

    def __init__(self, graph: nx.Graph, num_fses: list[int]):
        self.graph = graph
        self.num_fses = num_fses

    def solve_odsa(self, perm: list[int]) -> list[int]:
        return solve_odsa(self.graph, perm, self.num_fses)

    def check(self, sol: list[int]):
        check_dsa(self.graph, self.num_fses, sol)

    def calc_mofi(self, sol: list[int]) -> int:
        return calc_mofi(self.num_fses, sol)

    def calc_mofi_from_perm(self, perm: list[int]) -> int:
        return self.calc_mofi(self.solve_odsa(perm))

    @abstractmethod
    def solve_for_odsa_perm(self) -> list[int]: ...

    def solve(self) -> tuple[list[int], int]:
        best = self.solve_odsa(self.solve_for_odsa_perm())
        return best, self.calc_mofi(best)
