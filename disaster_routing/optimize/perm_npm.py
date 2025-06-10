from functools import lru_cache
import random
from stringprep import b3_exceptions
from typing import Callable

from .best import BestRecord


def permutation_npm(
    n: int,
    objective: Callable[[list[int]], int],
    initial_solution: list[int] = [],
    iter_count: int = 1000,
    lru_size: int = 10000,
) -> tuple[tuple[int], int]:
    objective_fn = lru_cache(lru_size)(objective)
    record = BestRecord[tuple[int, ...], int](objective_fn)

    sol = initial_solution
    if len(sol) == n:
        record.update(tuple(sol))

    def random_from_prefix(prefix: list[int]) -> list[int]:
        remaining = list(set(range(n)).difference(set(prefix)))
        random.shuffle(remaining)
        return prefix + remaining

    def random_from_subregions(sol: list[int], k: int = 10) -> list[list[list[int]]]:
        sols: list[list[list[int]]] = []
        if len(sol) > 0:
            sols.append([random_from_prefix(sol[:-1]) for _ in range(k)])
        else:
            sols.append([])
        for x in set(range(n)).difference(set(sol)):
            sols.append([random_from_prefix(sol + [x]) for _ in range(k)])
        return sols

    for _ in range(iter_count):
        samples = random_from_subregions(sol, 10)
        sample_costs = [
            min(
                (objective_fn(tuple(sample)) for sample in sample_group),
                default=float("inf"),
            )
            for sample_group in samples
        ]
        best_sample_idx = min(range(len(sample_costs)), key=lambda i: sample_costs[i])

        if (
            record.cost is None
            or sample_costs[best_sample_idx] < record.cost
            or (sample_costs[best_sample_idx] == record.cost and random.random() < 0.5)
        ):
            if best_sample_idx == 0:
                sol.pop()
            else:
                sol = min(
                    samples[best_sample_idx], key=lambda s: objective_fn(tuple(s))
                )[: len(sol) + 1]

    return record.get()
