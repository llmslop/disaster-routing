from functools import lru_cache
import random
from typing import Callable

from .best import BestRecord


def permutation_npm(
    n: int,
    objective: Callable[[tuple[int]], int],
    initial_solution: tuple[int, ...] = (),
    iter_count: int = 1000,
    num_sampled_points: int = 10,
    lru_size: int = 10000,
) -> tuple[tuple[int, ...], int]:
    objective_fn = lru_cache(lru_size)(objective)
    record = BestRecord[tuple[int, ...], int](objective_fn)

    sol = initial_solution
    if len(sol) == n:
        _ = record.update(sol)

    def random_from_prefix(prefix: tuple[int, ...]) -> tuple[int, ...]:
        remaining = list(set(range(n)).difference(set(prefix)))
        random.shuffle(remaining)
        return prefix + tuple(remaining)

    def random_from_subregions(
        sol: tuple[int, ...], k: int = 10
    ) -> list[list[tuple[int, ...]]]:
        sols: list[list[tuple[int, ...]]] = []
        if len(sol) > 0:
            sols.append([random_from_prefix(sol[:-1]) for _ in range(k)])
        else:
            sols.append([])
        for x in set(range(n)).difference(set(sol)):
            sols.append([random_from_prefix(sol + (x,)) for _ in range(k)])
        return sols

    for _ in range(iter_count):
        samples = random_from_subregions(sol, k=num_sampled_points)
        sample_costs = [
            min(
                (objective_fn(tuple(sample)) for sample in sample_group),
                default=float("inf"),
            )
            for sample_group in samples
        ]

        best_sample_indices = [
            i for i in range(len(sample_costs)) if sample_costs[i] == min(sample_costs)
        ]
        best_sample_idx = random.choice(best_sample_indices)

        if (
            record.cost is None
            or sample_costs[best_sample_idx] < record.cost
            or (sample_costs[best_sample_idx] == record.cost and random.random() < 0.5)
        ):
            if best_sample_idx == 0:
                sol = sol[:-1]
            else:
                sol = min(
                    samples[best_sample_idx], key=lambda s: objective_fn(tuple(s))
                )[: len(sol) + 1]
                if len(sol) == n:
                    _ = record.update(sol)

    best, cost = record.get()
    assert best is not None
    assert cost is not None
    return best, cost
