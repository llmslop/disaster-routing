from typing import Any, override
import networkx as nx

from .solver import DSASolver
from ..optimize.perm_ga import permutation_genetic_algorithm


class GADSASolver(DSASolver):
    kwargs: dict[str, Any]

    def __init__(self, graph: nx.Graph, num_fses: list[int], **kwargs):
        super().__init__(graph, num_fses)
        self.kwargs = kwargs

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        best_perm, _ = permutation_genetic_algorithm(
            len(self.graph), self.calc_mofi_from_perm, **self.kwargs
        )

        return best_perm
