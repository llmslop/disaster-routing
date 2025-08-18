from typing import override

from disaster_routing.conflicts.solver import DSASolver
from disaster_routing.conflicts.conflict_graph import ConflictGraph
from disaster_routing.optimize.perm_npm import permutation_npm
from disaster_routing.conflicts.fpga import FPGADSASolver
from disaster_routing.random.random import Random


class NPMDSASolver(DSASolver):
    random: Random
    iter_count: int
    num_sampled_points: int
    lru_size: int
    fpga: FPGADSASolver

    def __init__(
        self,
        random: Random,
        iter_count: int = 1000,
        num_sampled_points: int = 10,
        lru_size: int = 10000,
    ):
        self.random = random
        self.iter_count = iter_count
        self.num_sampled_points = num_sampled_points
        self.lru_size = lru_size
        self.fpga = FPGADSASolver()

    @override
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]:
        initial_perm = self.fpga.solve_for_odsa_perm(conflict_graph)
        best_perm, _ = permutation_npm(
            self.random,
            set(conflict_graph.graph.nodes),
            lambda x: self.calc_mofi_from_perm(conflict_graph, list(x)),
            tuple(initial_perm),
            iter_count=self.iter_count,
            num_sampled_points=self.num_sampled_points,
            lru_size=self.lru_size,
        )

        return list(best_perm)

    @override
    def name(self) -> str:
        return "npm"
