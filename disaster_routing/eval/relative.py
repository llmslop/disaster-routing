import logging
from math import isnan
from typing import cast, override

from hydra.utils import instantiate

from disaster_routing.conflicts.config import DSASolverConfig
from disaster_routing.conflicts.conflict_graph import ConflictGraph
from disaster_routing.conflicts.solver import DSASolver
from disaster_routing.routing.config import RoutingAlgorithmConfig
from disaster_routing.routing.routing_algo import RoutingAlgorithm
from disaster_routing.utils.structlog import SL
from disaster_routing.instances.instance import Instance
from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.eval.weighted_sum import WeightedSumEvaluator

log = logging.getLogger(__name__)


class RelativeEvaluator(Evaluator):
    evaluator: WeightedSumEvaluator

    def __init__(
        self,
        total_fs_weight: float,
        mofi_weight: float,
        instance: Instance,
        content_placement: dict[int, set[int]],
        approximate_dsa_solver: DSASolverConfig,
        router: RoutingAlgorithmConfig,
    ):
        if isnan(total_fs_weight) and isnan(mofi_weight):
            raise ValueError("Weighted sum evaluator weights must not be both NaN")
        if (isnan(total_fs_weight) and mofi_weight > 1.0) or (
            isnan(mofi_weight) and total_fs_weight > 1.0
        ):
            raise ValueError("Automatic weight must not exceed 1.0")
        total_fs_weight = (
            total_fs_weight if not isnan(total_fs_weight) else (1.0 - mofi_weight)
        )
        mofi_weight = mofi_weight if not isnan(mofi_weight) else (1.0 - total_fs_weight)

        # the base router should be a deterministic routing algorithm that does
        # not rely on the base_evaluator, such as the greedy or flow algorithms.
        base_evaluator = WeightedSumEvaluator(total_fs_weight, mofi_weight)

        router_algo = cast(
            RoutingAlgorithm,
            instantiate(
                router, evaluator=base_evaluator, dsa_solver=approximate_dsa_solver
            ),
        )
        dsa_solver = cast(DSASolver, instantiate(approximate_dsa_solver))

        routing_results = router_algo.route_instance(instance, content_placement)
        conflict_graph = ConflictGraph(instance, routing_results)
        _, mofi = dsa_solver.solve(conflict_graph)
        total_fs = conflict_graph.total_fs()

        log.debug(SL("Base routing results", total_fs=total_fs, mofi=mofi))

        self.evaluator = WeightedSumEvaluator(
            total_fs_weight / total_fs, mofi_weight / mofi
        )

    @override
    def evaluate(self, total_fs: int, mofi: int) -> float:
        return self.evaluator.evaluate(total_fs, mofi)

    @override
    def get_weights(self) -> tuple[float, float] | None:
        return self.evaluator.get_weights()
