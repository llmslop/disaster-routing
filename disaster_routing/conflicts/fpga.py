import logging
from typing import Callable, cast, override

from disaster_routing.conflicts.conflict_graph import ConflictGraph
from disaster_routing.conflicts.solver import DSASolver
from disaster_routing.random.random import Random
from disaster_routing.utils.structlog import SL

try:
    from dr_native import (
        fpga_solve_for_odsa_perm,  # pyright: ignore[reportAttributeAccessIssue]
    )
except ImportError:
    fpga_solve_for_odsa_perm = None

log = logging.getLogger(__name__)


class FPGADSASolver(DSASolver):
    num_attempts: int
    random: Random | None
    rust_backend: bool = True

    def __init__(
        self,
        random: Random | None = None,
        num_attempts: int = 0,
        rust_backend: bool = True,
    ):
        self.num_attempts = num_attempts
        self.random = random
        if fpga_solve_for_odsa_perm is None and rust_backend:
            log.warn(
                SL("Rust FPGA backend is not available, using Python implementation.")
            )
            rust_backend = False
        self.rust_backend = rust_backend

    @override
    def name(self) -> str:
        return "fpga"

    @staticmethod
    def min[K](keys: set[K], value_fn: Callable[[K], int]) -> tuple[K, int] | None:
        best: tuple[K, int] | None = None
        for key in keys:
            value = value_fn(key)
            if best is None or value < best[1]:
                best = key, value
        return best

    def solve_perm_with_first(
        self, conflict_graph: ConflictGraph, first: int
    ) -> tuple[list[int], int]:
        ordering = [first]
        remaining = set(conflict_graph.graph.nodes).difference(ordering)
        start_indices = {first: 0}
        mofi = 0

        while len(remaining) > 0:
            tup = FPGADSASolver.min(
                remaining,
                lambda node: max(
                    (
                        start_indices[prev] + conflict_graph.num_fses[prev]
                        for prev in set(conflict_graph.graph.adj[node]).intersection(
                            ordering
                        )
                    ),
                    default=0,
                ),
            )
            assert tup is not None
            node, start_indices[node] = tup
            mofi = max(mofi, start_indices[node] + conflict_graph.num_fses[node])

            remaining.remove(node)
            ordering.append(node)

        return ordering, mofi

    @override
    def solve_for_odsa_perm(self, conflict_graph: ConflictGraph) -> list[int]:
        start_indices = list(conflict_graph.graph.nodes)
        if self.num_attempts > 0 and self.num_attempts < len(start_indices):
            assert self.random is not None
            start_indices = self.random.stdlib.sample(
                start_indices, k=self.num_attempts
            )
        if self.rust_backend:
            return self.solve_for_odsa_perm_rust(conflict_graph, start_indices)
        else:
            return self.solve_for_odsa_perm_python(conflict_graph, start_indices)

    def solve_for_odsa_perm_python(
        self, conflict_graph: ConflictGraph, start_indices: list[int]
    ) -> list[int]:
        return min(
            (
                self.solve_perm_with_first(conflict_graph, first)
                for first in start_indices
            ),
            key=lambda sol: sol[1],
        )[0]

    def solve_for_odsa_perm_rust(
        self, conflict_graph: ConflictGraph, start_indices: list[int]
    ) -> list[int]:
        # if self.num_attempts < len(conflict_graph.graph.nodes):
        #     self.num_attempts = 1
        index_map: dict[int, int] = {}
        idx = 0
        for node in conflict_graph.graph.nodes:
            index_map[node] = idx
            idx += 1
        edges = [
            (index_map[edge[0]], index_map[edge[1]])
            for edge in conflict_graph.graph.edges
        ]
        num_fses = [
            conflict_graph.num_fses[index_map[i]]
            for i in range(len(conflict_graph.graph.nodes))
        ]

        # print("Python", self.solve_perm_with_first(conflict_graph, start_indices[0]))
        assert fpga_solve_for_odsa_perm is not None
        perm = cast(
            list[int],
            fpga_solve_for_odsa_perm(
                len(conflict_graph.graph.nodes),
                edges,
                [index_map[node] for node in start_indices],
                num_fses,
            ),
        )
        # print("Rust", perm)

        return [index_map[i] for i in perm]
