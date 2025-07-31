from math import ceil

from ..routing.routing_algo import Route
from ..instances.instance import Instance
from ..topologies.graphs import Graph
from ..utils.ilist import ilist

import networkx as nx


class RouteInfo:
    idx: int
    route: Route
    num_fs: int

    def __init__(self, idx: int, route: Route, num_fs: int):
        self.idx = idx
        self.route = route
        self.num_fs = num_fs


class ConflictGraph:
    graph: Graph
    num_fses: list[int]
    num_total_fs: int

    def __init__(self, inst: Instance, routes: ilist[ilist[Route]]):
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

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(i))
        for j in range(i):
            self.graph.nodes[j]["num_fs"] = route_infos[j].num_fs
            for k in range(j + 1, i):
                r1, r2 = route_infos[j], route_infos[k]
                if len(set(r1.route.edges()).intersection(set(r2.route.edges()))) != 0:
                    self.graph.add_edge(j, k)

        self.num_fses = list(map(lambda info: info.num_fs, route_infos))
        self.num_total_fs = sum(
            route.num_fs * len(route.route.edges()) for route in route_infos
        )

    def total_fs(self) -> int:
        return self.num_total_fs
