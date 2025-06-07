from typing import cast, override
import networkx as nx
import numpy as np

from .routing_algo import RoutingAlgorithm, Route
from ..instances.request import Request
from ..topologies.topology import Topology


def cast_node_type(node: int) -> str:
    return cast(str, cast(object, node))


class GreedyRoutingAlgorithm(RoutingAlgorithm):
    @override
    def route_request(self, req: Request, top: Topology, dst: list[int]) -> list[Route]:
        assert len(dst) >= 2

        graph = top.graph.copy()

        routes: list[Route] = []

        for K in range(len(dst)):
            result: tuple[dict[int, int], dict[int, list[int]]] = (
                nx.single_source_dijkstra(
                    top.graph,
                    cast_node_type(req.source),
                    weight=None,
                )
            )

            distances, paths = result
            nearest_node: int | None = min(
                dst,
                key=lambda dst_node: distances.get(dst_node, np.inf),
                default=None,
            )

            if nearest_node is None:
                break

            path = paths[nearest_node]
            for dz in top.dzs:
                if dz.contains_path(path):
                    dz.remove_from_graph(graph)

            routes.append(Route(top, path))
        return routes
