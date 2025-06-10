from typing import Any, override
import networkx as nx

from .solver import DSASolver
from ..optimize.perm_npm import permutation_npm
from ..conflicts.fpga import fpga


class NPMDSASolver(DSASolver):
    kwargs: dict[str, Any]

    def __init__(self, graph: nx.Graph, num_fses: list[int], **kwargs):
        super().__init__(graph, num_fses)
        self.kwargs = kwargs

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        initial_perm = fpga(self.graph, self.num_fses)
        best_perm, _ = permutation_npm(
            len(self.graph), self.calc_mofi_from_perm, initial_perm, **self.kwargs
        )

        return best_perm
