from dataclasses import dataclass, field
import logging
import time
from typing import Any, cast

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from .random.config import register_random_configs

from .placement.config import (
    ContentPlacementConfig,
    register_placement_configs,
)
from .placement.strategy import ContentPlacementStrategy

from .ilp.cdp import ILPCDP

from .instances.generate import (
    InstanceGeneratorConfig,
    load_or_gen_instance,
)
from .eval.config import EvaluationConfig, register_evaluator_configs
from .eval.evaluator import Evaluator
from .utils.structlog import SL, color_enabled
from .routing.routing_algo import InfeasibleRouteError, RoutingAlgorithm
from .routing.config import RoutingAlgorithmConfig, register_routing_algo_configs
from .conflicts.conflict_graph import ConflictGraph
from .conflicts.config import DSASolverConfig, register_dsa_solver_configs
from .conflicts.solver import DSASolver


def int_dict_to_list(d: dict[int, int]):
    return [d.get(k, -1) for k in range(max(d) + 1)]


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"approximate_dsa_solver": "ga"},
            {"dsa_solver": "ga"},
            {"eval": "weightedsum"},
            {"content_placement": "greedy"},
            {"instance/random": "unseeded"},
        ]
    )
    router: RoutingAlgorithmConfig | None = None
    approximate_dsa_solver: DSASolverConfig = MISSING
    dsa_solver: DSASolverConfig = MISSING
    instance: InstanceGeneratorConfig = field(default_factory=InstanceGeneratorConfig)
    eval: EvaluationConfig = MISSING
    content_placement: ContentPlacementConfig = MISSING
    safety_checks: bool = True
    ilp_check: bool | None = None
    ilp_solve: bool = False


OmegaConf.register_new_resolver(
    "if", lambda cond, if_true, if_false: if_true if cond else if_false
)
OmegaConf.register_new_resolver("colorlog", lambda: color_enabled)

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
register_evaluator_configs()
register_dsa_solver_configs()
register_dsa_solver_configs("approximate_dsa_solver")
register_routing_algo_configs()
register_placement_configs()
register_random_configs("instance")

log = logging.getLogger(__name__)

inf = int(1e16)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def my_main(cfg: MainConfig):
    log.debug(SL("Running on instance", instance=cfg.instance.path))
    instance = load_or_gen_instance(cfg.instance)

    if cfg.ilp_check is None:
        # skip ILP check for large instances to avoid OOM/TLE
        cfg.ilp_check = len(instance.requests) <= 50

    log.debug(SL("Instance info", instance=instance.to_json()))

    content_placement_strategy = cast(
        ContentPlacementStrategy, instantiate(cfg.content_placement)
    )
    content_placement = content_placement_strategy.place_content(instance)
    if cfg.safety_checks:
        content_placement_strategy.verify_placement(instance, content_placement)
    log.debug(SL("Content placement", placement=content_placement))

    evaluator = cast(
        Evaluator,
        instantiate(cfg.eval, instance=instance, content_placement=content_placement),
    )

    ilp: ILPCDP | None = None
    if cfg.ilp_solve or cfg.ilp_check:
        ilp = ILPCDP(instance, evaluator)
    if cfg.ilp_solve:
        assert ilp is not None
        _ = ilp.solve()

    if cfg.router is not None:
        router = cast(
            RoutingAlgorithm,
            instantiate(
                cfg.router,
                evaluator=evaluator,
                dsa_solver=cfg.approximate_dsa_solver,
            ),
        )

        start = time.time()
        try:
            all_routes = router.route_instance(instance, content_placement)
            all_routes = router.sort_routes(all_routes)
            if cfg.safety_checks:
                for routes, req in zip(all_routes, instance.requests):
                    router.check_solution(
                        req, content_placement[req.content_id], routes
                    )
            log.debug(
                SL(
                    "Routing results",
                    route_nodes=[
                        [r.node_list for r in routes] for routes in all_routes
                    ],
                    route_formats=[
                        [r.format.name for r in routes] for routes in all_routes
                    ],
                    time=time.time() - start,
                )
            )

            conflict_graph = ConflictGraph(instance, all_routes)
            dsa_solver = cast(DSASolver, instantiate(cfg.dsa_solver))
            start_indices, mofi = dsa_solver.solve(conflict_graph)
            if cfg.safety_checks:
                dsa_solver.check(conflict_graph, start_indices)
            log.debug(
                SL(
                    "DSA results",
                    start_indices=int_dict_to_list(start_indices),
                    num_fses=int_dict_to_list(conflict_graph.num_fses),
                )
            )

            log.info(
                SL(
                    "Final solution",
                    total_fs=conflict_graph.total_fs(),
                    mofi=mofi,
                    score=evaluator.evaluate(conflict_graph.total_fs(), mofi),
                )
            )
            if cfg.ilp_check:
                assert ilp is not None
                ilp.check_solution(all_routes, start_indices, content_placement)
        except InfeasibleRouteError:
            log.info(
                SL(
                    "Final solution",
                    total_fs=inf,
                    mofi=inf,
                    score=evaluator.evaluate(inf, inf),
                )
            )


if __name__ == "__main__":
    my_main()
