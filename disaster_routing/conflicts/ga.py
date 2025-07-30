from typing import override

from .solver import DSASolver
from .conflict_graph import ConflictGraph
from ..optimize.perm_ga import permutation_genetic_algorithm
from ..random.random import Random


class GADSASolver(DSASolver):
    pop_size: int
    generations: int
    mutation_rate: float
    lru_size: int
    random: Random

    def __init__(
        self,
        conflict_graph: ConflictGraph,
        random: Random,
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
        self.random = random

    @override
    def solve_for_odsa_perm(self) -> list[int]:
        best_perm, _ = permutation_genetic_algorithm(
            self.random,
            len(self.conflict_graph.graph),
            lambda x: self.calc_mofi_from_perm(list(x)),
            population_size=self.pop_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            lru_size=self.lru_size,
        )

        return list(best_perm)
