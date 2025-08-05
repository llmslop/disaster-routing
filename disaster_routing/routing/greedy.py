from math import ceil
from typing import cast, override
import networkx as nx
import numpy as np

from .routing_algo import InfeasibleRouteError, RoutingAlgorithm, Route
from ..instances.modulation import ModulationFormat
from ..instances.request import Request
from ..topologies.topology import Topology
from ..utils.ilist import ilist


class GreedyRoutingAlgorithm(RoutingAlgorithm):
    def __init__(self, *_: object, **__: object) -> None:
        pass

    @override
    def route_request(self, req: Request, top: Topology, dst: set[int]) -> ilist[Route]:
        assert len(dst) >= 2
        dst = set(dst)
        graph = top.graph.copy()

        routes: list[Route] = []
        best_route_set: ilist[Route] | None = None
        best_route_set_cost = np.inf

        for K in range(len(dst)):
            distances, paths = cast(
                tuple[dict[int, int], dict[int, list[int]]],
                nx.single_source_dijkstra(
                    graph,
                    req.source,
                    weight=None,
                ),
            )

            nearest_node: int | None = min(
                dst,
                key=lambda dst_node: distances.get(dst_node, np.inf),
                default=None,
            )

            if nearest_node is None or nearest_node not in paths:
                break

            path = ilist[int](paths[nearest_node])
            for dz in top.dzs:
                if dz.affects_path(path):
                    dz.remove_from_graph(graph, exclude_node=req.source)

            assert req.source in graph
            dst.intersection_update(graph.nodes)

            dist = nx.path_weight(top.graph, path, "weight")
            format = ModulationFormat.best_rate_format_with_distance(dist)
            if format is None:
                break

            routes.append(Route(top, path, format))

            if len(routes) >= 2:
                cost = self.route_set_cost(routes, req.bpsk_fs_count)
                if cost < best_route_set_cost:
                    best_route_set = ilist(routes)
                    best_route_set_cost = cost

        if best_route_set is None:
            raise InfeasibleRouteError()
        return best_route_set

    def route_set_cost(
        self,
        routes: list[Route],
        total_fs_bpsk: int,
    ) -> int:
        assert len(routes) >= 2
        num_working_paths = len(routes) - 1
        cost = 0
        for route in routes:
            num_links = len(route.node_list) - 1
            num_fs_per_link = ceil(
                total_fs_bpsk / route.format.relative_bpsk_rate() / num_working_paths
            )

            cost += num_links * num_fs_per_link
        return cost
