from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from math import ceil
import pickle as pkl
from random import random, choices, randint, shuffle
from typing import Any, cast

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

from omegaconf import MISSING

# import networkx as nx
# import matplotlib.pyplot as plt

from .instances.instance import Instance
from .instances.generate import (
    InstanceGenerator,
    InstanceGeneratorConfig,
    load_or_gen_instance,
)
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
        ]
    )
    router: RoutingAlgorithmConfig | None = None
    dsa_solver: DSASolverConfig = MISSING
    instance: InstanceGeneratorConfig = field(default_factory=InstanceGeneratorConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
register_dsa_solver_configs()
register_routing_algo_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def my_main(cfg: MainConfig):
    instance = load_or_gen_instance(cfg.instance)
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
    log.debug(f"Content placement: {content_placement}")

    if cfg.router is not None:
        log.debug(f"Using {cfg.router} router, {cfg.dsa_solver} DSA solver")
        router = cast(RoutingAlgorithm, instantiate(cfg.router))
        all_routes = router.route_instance(instance, content_placement)
        conflict_graph = ConflictGraph(instance, all_routes)
        dsa_solver = cast(DSASolver, instantiate(cfg.dsa_solver, conflict_graph))
        _, mofi = dsa_solver.solve()

        log.info(f"Final solution: {conflict_graph.total_fs()} FS, MOFI {mofi}")


if __name__ == "__main__":
    my_main()
