import logging
from typing import override

from hydra.utils import instantiate

from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.instances.instance import Instance
from disaster_routing.routing.routing_algo import InfeasibleRouteError
from disaster_routing.solver.config import CDPSolverConfig
from disaster_routing.solver.solution import CDPSolution
from disaster_routing.solver.solver import CDPSolver
from disaster_routing.utils.structlog import SL

log = logging.getLogger(__name__)


class LSSolver(CDPSolver):
    base: CDPSolver
    f_max: int

    def __init__(
        self, evaluator: Evaluator, base: CDPSolverConfig, f_max: int = 100
    ) -> None:
        super().__init__(evaluator)
        self.base = instantiate(base, evaluator=evaluator)
        self.f_max = f_max

    @override
    def solve(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> CDPSolution:
        num_passes = 1
        sol = self.base.solve(inst, content_placement)
        mofi_edges = self.edges_with_mofi(sol)
        best = self.evaluator.evaluate_solution(sol)

        for iter in range(self.f_max):
            change = False
            for edge in mofi_edges:
                ls_inst = inst.remove_edge(edge)
                try:
                    num_passes += 1
                    log.debug(
                        SL("LS pass", iter=iter, num_passes=num_passes, edge=edge)
                    )
                    ls_sol = self.base.solve(ls_inst, content_placement)
                    ls_mofi_edges = self.edges_with_mofi(ls_sol)
                    if self.evaluator.evaluate_solution(ls_sol) < best:
                        log.debug(SL("Improvement found", edge=edge))
                        mofi_edges.discard(edge)
                        mofi_edges.update(ls_mofi_edges)
                        sol = ls_sol
                        best = self.evaluator.evaluate_solution(ls_sol)
                        change = True
                        break
                except InfeasibleRouteError:
                    pass
            if not change:
                break

        log.debug(SL("Total passes", num_passes=num_passes))
        return sol

    def edges_with_mofi(self, sol: CDPSolution) -> set[tuple[int, int]]:
        mofi = sol.mofi()
        routes = [route for routes in sol.all_routes for route in routes]
        return {
            edge
            for i, route in enumerate(routes)
            if sol.start_indices[i] + sol.num_fses[i] == mofi
            for edge in route.edges()
        }

    @override
    def name(self) -> str:
        return f"{self.base.name()}+ls"
