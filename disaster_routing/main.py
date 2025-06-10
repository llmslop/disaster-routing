from dataclasses import dataclass
import logging
from math import ceil
import pickle as pkl
from random import random, choices, randint, shuffle
from typing import cast

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
from .conflicts.conflict_graph import make_conflict_graph
from .conflicts.solver import DSASolver
from .conflicts.ga import GADSASolver
from .conflicts.npm import NPMDSASolver


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


def load_or_gen_instance(n: int, path: str = "instances/temp_instance.pkl") -> Instance:
    try:
        with open(path, "rb") as f:
            return cast(Instance, pkl.load(f))
    except IOError:
        instance = gen_instance(n)
        with open(path, "wb") as f:
            pkl.dump(instance, f)
        return instance


@dataclass
class MainConfig:
    router: str = "greedy"
    dsa_solver: str = "ga"
    instance: str = "instances/temp_instance.pkl"


def create_router(router: str) -> RoutingAlgorithm:
    match router:
        case "greedy":
            return GreedyRoutingAlgorithm()
        case "flow":
            return FlowRoutingAlgorithm()
        case _:
            raise ValueError(f"{router} is not a valid routing algorithm")


def create_dsa_solver(solver: str, *args, **kwargs) -> DSASolver:
    match solver:
        case "ga":
            return GADSASolver(*args, **kwargs)
        case "npm":
            return NPMDSASolver(*args, **kwargs)
        case _:
            raise ValueError(f"{solver} is not a valid DSA solver")


cs = ConfigStore.instance()
cs.store("main", node=MainConfig)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_main(cfg: MainConfig):
    instance = load_or_gen_instance(10, path=cfg.instance)
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
    log.debug("Content placement", content_placement)

    router = create_router(cfg.router)
    log.debug(f"Using {cfg.router} router")
    all_routes = [
        router.route_request_checked(
            req, instance.topology, content_placement[req.content_id]
        )
        for req in instance.requests
    ]

    conflict_graph, num_fses = make_conflict_graph(instance, all_routes)
    log.debug(f"Using {cfg.dsa_solver} DSA solver")
    dsa_solver = create_dsa_solver(cfg.dsa_solver, conflict_graph, num_fses)
    best, mofi = dsa_solver.solve()
    log.info(f"Final solution: {sum(num_fses)} FS, MOFI {mofi}")


if __name__ == "__main__":
    my_main()
