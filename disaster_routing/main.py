from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from math import ceil
import pickle as pkl
from random import random, choices, randint, shuffle
from typing import Any, cast
from hydra.utils import instantiate
from omegaconf import MISSING

import hydra
from hydra.core.config_store import ConfigStore

# import networkx as nx
# import matplotlib.pyplot as plt

from .instances.instance import Instance
from .instances.request import Request
from .topologies.nsfnet import nsfnet
from .routing.greedy import GreedyRoutingAlgorithm
from .routing.flow import FlowRoutingAlgorithm
from .routing.routing_algo import RoutingAlgorithm
from .routing.config import RoutingAlgorithmConfig, register_routing_algo_configs
from .conflicts.conflict_graph import ConflictGraph
from .conflicts.config import DSASolverConfig, register_dsa_solver_configs


topology = nsfnet()
dc_positions = [2, 5, 6, 9, 11]
# dc_positions = [2, 5, 9]

content_count = 10
transmission_rate_range = (0, 10)


def gen_requests(n: int) -> list[Request]:
    source_nodes: set[int] = set(topology.graph)
    source_nodes.difference_update(dc_positions)

    sources = choices(list(source_nodes), k=n)
    contents = [randint(0, content_count - 1) for _ in range(n)]
    trans_rate = [
        random() * (transmission_rate_range[1] - transmission_rate_range[0])
        + transmission_rate_range[0]
        for _ in range(n)
    ]

    return [
        Request(
            sources[i],
            cast(int, topology.graph.in_degree[sources[i]]),
            contents[i],
            ceil(trans_rate[i]),
        )
        for i in range(n)
    ]


def gen_instance(n: int) -> Instance:
    return Instance(topology, gen_requests(n))


def load_or_gen_instance(
    n: int, path: str = "instances/temp_instance.pkl", force_recreate: bool = False
) -> Instance:
    try:
        if force_recreate:
            raise IOError()
        with open(path, "rb") as f:
            return cast(Instance, pkl.load(f))
    except IOError:
        instance = gen_instance(n)
        with open(path, "wb") as f:
            pkl.dump(instance, f)
        return instance


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"dsa_solver": "ga"}, {"router": "greedy"}]
    )
    router: RoutingAlgorithmConfig = MISSING
    dsa_solver: DSASolverConfig = MISSING
    instance: str = "instances/temp_instance.pkl"
    force_recreate: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
register_dsa_solver_configs()
register_routing_algo_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def my_main(cfg: MainConfig):
    instance = load_or_gen_instance(
        10, path=cfg.instance, force_recreate=cfg.force_recreate
    )
    # dc_placement = solve_dc_placement(instance, dc_positions)
    contents = set(req.content_id for req in instance.requests)
    dc_placement = [list(contents) for _ in dc_positions]
    content_placement = [
        [
            dc
            for dc, placement in zip(dc_positions, dc_placement)
            if content in placement
        ]
        for content in range(content_count)
    ]
    log.debug(f"Content placement: {content_placement}")

    log.debug(f"Using {cfg.router} router, {cfg.dsa_solver} DSA solver")
    router: RoutingAlgorithm = instantiate(cfg.router)
    all_routes = router.route_instance(instance, content_placement)
    conflict_graph = ConflictGraph(instance, all_routes)
    dsa_solver = instantiate(cfg.dsa_solver, conflict_graph)
    best, mofi = dsa_solver.solve()

    log.info(f"Final solution: {conflict_graph.total_fs()} FS, MOFI {mofi}")


if __name__ == "__main__":
    my_main()
