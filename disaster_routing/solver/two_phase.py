from typing import override

from hydra.utils import instantiate

from disaster_routing.conflicts.config import DSASolverConfig
from disaster_routing.conflicts.conflict_graph import ConflictGraph
from disaster_routing.conflicts.solver import DSASolver
from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.instances.instance import Instance
from disaster_routing.routing.config import RoutingAlgorithmConfig
from disaster_routing.routing.routing_algo import RoutingAlgorithm
from disaster_routing.solver.solution import CDPSolution
from disaster_routing.solver.solver import CDPSolver


class TwoPhaseSolver(CDPSolver):
    router: RoutingAlgorithm
    dsa_solver: DSASolver

    def __init__(
        self,
        evaluator: Evaluator,
        router: RoutingAlgorithmConfig,
        dsa_solver: DSASolverConfig,
    ):
        super().__init__(evaluator)
        self.dsa_solver = instantiate(dsa_solver)
        self.router = instantiate(
            router, evaluator=self.evaluator, dsa_solver=self.dsa_solver
        )

    @override
    def name(self) -> str:
        return f"{self.router.name()}+{self.dsa_solver.name()}"

    @override
    def solve(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> CDPSolution:
        all_routes = self.router.route_instance(inst, content_placement)
        all_routes = self.router.sort_routes(all_routes)
        conflict_graph = ConflictGraph(inst, all_routes)
        start_indices, mofi = self.dsa_solver.solve(conflict_graph)
        sol = CDPSolution(all_routes, start_indices, conflict_graph.num_fses)
        assert sol.total_fs() == conflict_graph.total_fs()
        assert sol.mofi() == mofi
        return sol
