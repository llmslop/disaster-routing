from abc import ABC, abstractmethod
from math import ceil

from disaster_routing.conflicts.conflict_graph import ConflictGraph
from disaster_routing.conflicts.solver import DSASolver
from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.instances.instance import Instance
from disaster_routing.routing.routing_algo import RoutingAlgorithm
from disaster_routing.solver.solution import CDPSolution


class CDPSolver(ABC):
    evaluator: Evaluator

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    @abstractmethod
    def solve(
        self,
        inst: Instance,
        content_placement: dict[int, set[int]],
    ) -> CDPSolution: ...

    @abstractmethod
    def name(self) -> str: ...

    @staticmethod
    def check(
        inst: Instance,
        content_placement: dict[int, set[int]],
        sol: CDPSolution,
    ):
        idx = 0
        for routes, req in zip(sol.all_routes, inst.requests):
            RoutingAlgorithm.check_solution(
                req, content_placement[req.content_id], routes
            )
            for route in routes:
                num_fs = sol.num_fses[idx]
                assert num_fs == ceil(
                    req.bpsk_fs_count
                    / (len(routes) - 1)
                    / route.format.relative_bpsk_rate()
                )
                idx += 1
        conflict_graph = ConflictGraph(inst, sol.all_routes)
        assert sol.total_fs() == conflict_graph.total_fs()
        assert sol.mofi() == DSASolver.calc_mofi(conflict_graph, sol.start_indices)
        DSASolver.check(conflict_graph, sol.start_indices)
