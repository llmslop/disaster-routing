from math import ceil
from ..routing.routing_algo import Route
from ..instances.request import Request
from ..instances.instance import Instance

import networkx as nx


class RouteInfo:
    idx: int
    route: Route
    num_fs: int

    def __init__(self, idx: int, route: Route, num_fs: int):
        self.idx = idx
        self.route = route
        self.num_fs = num_fs


def make_conflict_graph(
    inst: Instance, routes: list[list[Route]]
) -> tuple[nx.Graph, list[int]]:
    i = 0
    route_infos: list[RouteInfo] = []
    for req, route_set in zip(inst.requests, routes):
        for route in route_set:
            num_fs = ceil(
                req.bpsk_fs_count
                / route.format.relative_bpsk_rate()
                / (len(route_set) - 1)
            )
            route_infos.append(RouteInfo(i, route, num_fs))
            i += 1

    G = nx.Graph()
    G.add_nodes_from(range(i))
    for j in range(i):
        G.nodes[j]["num_fs"] = route_infos[j].num_fs
        for k in range(j + 1, i):
            r1, r2 = route_infos[j], route_infos[k]
            if len(set(r1.route.edges()).intersection(set(r2.route.edges()))) != 0:
                G.add_edge(j, k)
    return G, list(map(lambda info: info.num_fs, route_infos))
