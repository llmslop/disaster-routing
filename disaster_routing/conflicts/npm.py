from typing import override

from .solver import DSASolver
from .conflict_graph import ConflictGraph
from ..optimize.perm_npm import permutation_npm
from ..conflicts.fpga import fpga


class NPMDSASolver(DSASolver):
    iter_count: int
    num_sampled_points: int
    lru_size: int

    def __init__(
        self,
        conflict_graph: ConflictGraph,
        iter_count: int = 1000,
        num_sampled_points: int = 10,
        lru_size: int = 10000,
    ):
        super().__init__(conflict_graph)
        self.iter_count = iter_count
        self.num_sampled_points = num_sampled_points
        self.lru_size = lru_size

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        initial_perm = fpga(self.conflict_graph.graph, self.conflict_graph.num_fses)
        best_perm, _ = permutation_npm(
            len(self.conflict_graph.graph),
            lambda x: self.calc_mofi_from_perm(list(x)),
            tuple(initial_perm),
            iter_count=self.iter_count,
            num_sampled_points=self.num_sampled_points,
            lru_size=self.lru_size,
        )

        return list(best_perm)
