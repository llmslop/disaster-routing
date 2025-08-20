import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from disaster_routing.eval.config import EvaluationConfig, register_evaluator_configs
from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.ilp.cdp import ILPCDP
from disaster_routing.instances.generate import (
    InstanceGeneratorConfig,
    load_or_gen_instance,
)
from disaster_routing.placement.config import (
    ContentPlacementConfig,
    register_placement_configs,
)
from disaster_routing.placement.strategy import ContentPlacementStrategy
from disaster_routing.random.config import register_random_configs
from disaster_routing.routing.routing_algo import InfeasibleRouteError
from disaster_routing.solver.config import CDPSolverConfig, register_solver_configs
from disaster_routing.solver.solver import CDPSolver
from disaster_routing.utils.structlog import SL, color_enabled


def int_dict_to_list(d: dict[int, int]):
    return [d.get(k, -1) for k in range(max(d) + 1)]


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"eval": "relative"},
            {"content_placement": "greedy"},
            {"instance.random": "unseeded"},
        ]
    )
    solver: CDPSolverConfig | None = None
    instance: InstanceGeneratorConfig = field(default_factory=InstanceGeneratorConfig)
    eval: EvaluationConfig = MISSING
    content_placement: ContentPlacementConfig = MISSING
    safety_checks: bool = True
    ilp_check: bool | None = None


OmegaConf.register_new_resolver(
    "if", lambda cond, if_true, if_false: if_true if cond else if_false
)
OmegaConf.register_new_resolver("colorlog", lambda: color_enabled)

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
register_evaluator_configs()
register_placement_configs()
register_random_configs("instance.random")
register_solver_configs()

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

    if cfg.solver is not None:
        solver = cast(CDPSolver, instantiate(cfg.solver, evaluator=evaluator))
        log.info(SL("Solver details", name=solver.name()))

        start = time.time()
        try:
            sol = solver.solve(instance, content_placement)
            if sol.content_placement_override:
                log.debug(SL("Content placement override", placement=content_placement))
            log.debug(
                SL(
                    "Routing results",
                    route_nodes=[
                        [r.node_list for r in routes] for routes in sol.all_routes
                    ],
                    route_formats=[
                        [r.format.name for r in routes] for routes in sol.all_routes
                    ],
                    time=time.time() - start,
                )
            )
            log.debug(
                SL(
                    "DSA results",
                    start_indices=int_dict_to_list(sol.start_indices),
                    num_fses=int_dict_to_list(sol.num_fses),
                )
            )

            log.info(
                SL(
                    "Final solution",
                    total_fs=sol.total_fs(),
                    mofi=sol.mofi(),
                    score=evaluator.evaluate_solution(sol),
                )
            )
            if cfg.safety_checks:
                CDPSolver.check(instance, content_placement, sol)
            if cfg.ilp_check:
                ilp = ILPCDP(instance, evaluator)
                ilp.check_solution(
                    sol.all_routes,
                    sol.start_indices,
                    content_placement,
                    sol.total_fs(),
                    sol.mofi(),
                )
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
