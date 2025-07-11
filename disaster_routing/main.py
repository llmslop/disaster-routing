from dataclasses import dataclass, field
import logging
from typing import Any, cast

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .instances.generate import (
    InstanceGeneratorConfig,
    load_or_gen_instance,
)
from .eval.config import EvaluationConfig, register_evaluator_configs
from .eval.evaluator import Evaluator
from .utils.structlog import SL
from .topologies.nsfnet import nsfnet
from .routing.routing_algo import RoutingAlgorithm
from .routing.config import RoutingAlgorithmConfig, register_routing_algo_configs
from .conflicts.conflict_graph import ConflictGraph
from .conflicts.config import DSASolverConfig, register_dsa_solver_configs
from .conflicts.solver import DSASolver


topology = nsfnet()
dc_positions = [2, 5, 6, 9, 11]


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dsa_solver": "ga"},
            {"eval": "weightedsum"},
        ]
    )
    router: RoutingAlgorithmConfig | None = None
    dsa_solver: DSASolverConfig = MISSING
    instance: InstanceGeneratorConfig = field(default_factory=InstanceGeneratorConfig)
    eval: EvaluationConfig = MISSING
    safety_checks: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
register_evaluator_configs()
register_dsa_solver_configs()
register_routing_algo_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def my_main(cfg: MainConfig):
    instance = load_or_gen_instance(cfg.instance)
    evaluator = cast(Evaluator, instantiate(cfg.eval))
    # dc_placement = solve_dc_placement(instance, dc_positions)
    contents = set(req.content_id for req in instance.requests)
    dc_placement = [list(contents) for _ in dc_positions]
    content_placement = {
        content: [
            dc
            for dc, placement in zip(dc_positions, dc_placement)
            if content in placement
        ]
        for content in contents
    }
    log.debug(SL("Content placement", placement=content_placement))

    if cfg.router is not None:
        router = cast(RoutingAlgorithm, instantiate(cfg.router, evaluator=evaluator))
        all_routes = router.route_instance(instance, content_placement)
        if cfg.safety_checks:
            for routes, req in zip(all_routes, instance.requests):
                router.check_solution(req, content_placement[req.content_id], routes)

        conflict_graph = ConflictGraph(instance, all_routes)
        dsa_solver = cast(DSASolver, instantiate(cfg.dsa_solver, conflict_graph))
        start_indices, mofi = dsa_solver.solve()
        if cfg.safety_checks:
            dsa_solver.check(start_indices)

        log.info(
            SL(
                "Final solution",
                total_fs=conflict_graph.total_fs(),
                mofi=mofi,
                score=evaluator.evaluate(conflict_graph.total_fs(), mofi),
            )
        )


if __name__ == "__main__":
    my_main()
