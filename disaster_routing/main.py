from math import ceil
from os import path as pth
import pickle as pkl
from random import random, choices, randint
from typing import cast

from .instances.instance import Instance
from .instances.request import Request
from .topologies.nsfnet import nsfnet
from .placement.placement import solve_dc_placement
from .routing.greedy import GreedyRoutingAlgorithm

topology = nsfnet()
dc_positions = [2, 5, 6, 9, 11]

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


if __name__ == "__main__":
    instance = load_or_gen_instance(10)
    dc_placement = solve_dc_placement(instance, dc_positions)
    content_placement = [
        [
            dc
            for dc, placement in zip(dc_positions, dc_placement)
            if content in placement
        ]
        for content in range(content_count)
    ]
    print(content_placement)

    router = GreedyRoutingAlgorithm()

    for req in instance.requests:
        routes = router.route_request(
            req, instance.topology, content_placement[req.content_id]
        )
        router.check_solution(req, content_placement[req.content_id], routes)
