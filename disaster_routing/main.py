from dataclasses import dataclass
import logging
from math import ceil
from os import path as pth
import pickle as pkl
from random import random, choices, randint, shuffle
from typing import cast

import hydra
from hydra.core.config_store import ConfigStore
import networkx as nx
import matplotlib.pyplot as plt

from .topologies.random import random_topology
from .instances.instance import Instance
from .instances.request import Request
from .topologies.nsfnet import nsfnet
from .placement.placement import solve_dc_placement
from .routing.greedy import GreedyRoutingAlgorithm
from .routing.flow import FlowRoutingAlgorithm
from .routing.routing_algo import Route, RoutingAlgorithm
from .conflicts.conflict_graph import make_conflict_graph
from .conflicts.odsa import solve_odsa
from .conflicts.mofi import calc_mofi
from .conflicts.fpga import fpga
from .optimize.perm_ga import permutation_genetic_algorithm
from .optimize.perm_npm import permutation_npm


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
    log.info(f"Using {cfg.router} router")
    # router = FlowRoutingAlgorithm()
    all_routes: list[list[Route]] = []
    for i, req in enumerate(instance.requests):
        dcs = content_placement[req.content_id]
        routes = router.route_request(req, instance.topology, dcs)
        router.check_solution(req, dcs, routes)

        route_infos: list[str] = []
        for j, route in enumerate(routes):
            num_fs = ceil(
                req.bpsk_fs_count
                / route.format.relative_bpsk_rate()
                / (len(routes) - 1)
            )

            route_infos.append(f"{route.node_list} ({route.format.name}, {num_fs} FS)")
        all_routes.append(routes)

    conflict_graph, num_fses = make_conflict_graph(instance, all_routes)
    match cfg.dsa_solver:
        case "ga":
            best, mofi = permutation_genetic_algorithm(
                len(conflict_graph),
                lambda perm: max(
                    a + b
                    for a, b in zip(
                        num_fses, solve_odsa(conflict_graph, perm, num_fses)
                    )
                ),
                generations=max(10, len(instance.topology.graph)),
            )
            log.debug("Using GA:", best, mofi)
        case "npm":
            fpga_order = fpga(conflict_graph, num_fses)
            best = solve_odsa(conflict_graph, fpga_order, num_fses)
            mofi = calc_mofi(num_fses, best)
            best, mofi = permutation_npm(
                len(conflict_graph),
                lambda perm: max(
                    a + b
                    for a, b in zip(
                        num_fses, solve_odsa(conflict_graph, perm, num_fses)
                    )
                ),
                fpga_order,
            )
            log.debug("Using NPM:", best, mofi)

    # print("Conflict graph: ", conflict_graph)
    # nx.draw(conflict_graph, pos=nx.layout.circular_layout(conflict_graph))
    # plt.show()

    log.info(f"Final solution: {sum(num_fses)} FS, MOFI {mofi}")


if __name__ == "__main__":
    my_main()
