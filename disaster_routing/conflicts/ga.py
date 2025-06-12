from typing import override

from .solver import DSASolver
from .conflict_graph import ConflictGraph
from ..optimize.perm_ga import permutation_genetic_algorithm


class GADSASolver(DSASolver):
    pop_size: int
    generations: int
    mutation_rate: float
    lru_size: int

    def __init__(
        self,
        conflict_graph: ConflictGraph,
        pop_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.2,
        lru_size: int = 10000,
    ):
        super().__init__(conflict_graph)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.lru_size = lru_size

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        best_perm, _ = permutation_genetic_algorithm(
            len(self.conflict_graph.graph),
            lambda x: self.calc_mofi_from_perm(list(x)),
            population_size=self.pop_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            lru_size=self.lru_size,
        )

        return list(best_perm)
