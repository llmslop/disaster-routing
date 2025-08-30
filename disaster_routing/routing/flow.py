import logging
from collections.abc import Iterable
from itertools import product
from math import ceil
from typing import cast, override

import networkx as nx
import numpy as np

from disaster_routing.instances.request import Request
from disaster_routing.routing.routing_algo import (
    InfeasibleRouteError,
    Route,
    RoutingAlgorithm,
)
from disaster_routing.topologies.graphs import Graph, StrDiGraph
from disaster_routing.topologies.topology import Topology
from disaster_routing.utils.ilist import ilist

log = logging.getLogger(__name__)


def init_free_nodes(
    top: Topology, req: Request, dz_set_lists: list[list[set[int]]]
) -> set[int]:
    free_nodes: set[int] = set(top.graph.nodes)
    mentioned_nodes = {
        node
        for dz_set_list in dz_set_lists
        for dz_set in dz_set_list
        for node in dz_set
    }
    for node in mentioned_nodes:
        if any(req.source not in dz.nodes and node in dz.nodes for dz in top.dzs):
            free_nodes.remove(node)
    return free_nodes


def extract_dz_index(name: str) -> int | None:
    name = name.replace("_source", "").replace("_sink", "").replace("DZ", "")
    try:
        return int(name)
    except ValueError:
        return None


def remove_nones[T](seq: list[T | None]) -> list[T]:
    return [x for x in seq if x is not None]


def remove_consecutive_duplicates[T](seq: list[T]) -> list[T]:
    if not seq:
        return []
    result: list[T] = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result


def extract_all_flow_paths(
    G: StrDiGraph,
    source: str,
    sink: str,
    flow_dict: dict[str, dict[str, int]],
) -> list[list[str]]:
    # flow_graph = {u: dict(v) for u, v in flow_dict.items()}

    paths: list[list[str]] = []

    def find_path() -> list[str] | None:
        path = [source]
        current = source
        while current != sink:
            for neighbor, flow in flow_dict[current].items():
                if flow > 0:
                    path.append(neighbor)
                    flow_dict[current][neighbor] -= 1  # consume flow
                    current = neighbor
                    break
            else:
                # Dead end (should not happen in correct flow network)
                return None
        return path

    # Keep extracting paths until all flow is consumed
    while True:
        path = find_path()
        if path is None:
            break
        paths.append(path)

    return paths


def reconstruct_min_hop_path(
    G: Graph,
    source: int,
    avail_outs: Iterable[int],
    group_path: list[set[int]],
    free_nodes: set[int],
) -> ilist[int]:
    nodes = group_path[0].union(*group_path[1:]).union(free_nodes)
    outs = nodes.intersection(avail_outs)
    graph = G.subgraph(nodes)
    length, path = nx.single_source_dijkstra(graph, source)
    outs = [out for out in outs if out in length]
    out = min(outs, key=lambda x: length.get(x, np.inf), default=None)
    if out is None:
        return ()
    return ilist[int](path[out])


class FlowRoutingAlgorithm(RoutingAlgorithm):
    def __init__(self, *_: object, **__: object) -> None:
        self.inv_alpha: float = 1
        pass

    @override
    def name(self) -> str:
        return "flow"

    @override
    def route_request(self, req: Request, top: Topology, dst: set[int]) -> ilist[Route]:
        assert len(dst) >= 2

        best_route_set: ilist[Route] | None = None
        best_route_set_cost = np.inf

        for K in range(2, min(len(dst), req.max_path_count) + 1):
            flow_graph = StrDiGraph()

            flow_graph.add_node("source", demand=-K)
            flow_graph.add_node("sink", demand=K)

            source_nodes: dict[int, str] = {}
            sink_nodes: dict[int, str] = {}

            for i, dz in enumerate(top.dzs):
                if req.source in dz.nodes:
                    flow_graph.add_node(f"DZ{i}", demand=0)
                    flow_graph.add_edge("source", f"DZ{i}")
                    source_nodes[i] = sink_nodes[i] = f"DZ{i}"
                else:
                    flow_graph.add_node(f"DZ{i}_source")
                    flow_graph.add_node(f"DZ{i}_sink")
                    flow_graph.add_edge(f"DZ{i}_source", f"DZ{i}_sink", capacity=1)
                    source_nodes[i] = f"DZ{i}_source"
                    sink_nodes[i] = f"DZ{i}_sink"

                if len(dz.nodes.intersection(dst)) > 0:
                    if req.source in dz.nodes:
                        flow_graph.add_edge(
                            sink_nodes[i],
                            "sink",
                            capacity=len(dz.nodes.intersection(dst)),
                        )
                    else:
                        flow_graph.add_edge(sink_nodes[i], "sink", capacity=1)

            dists = nx.single_source_dijkstra_path_length(top.graph, req.source)
            beta = 0.7

            for i, dzi in enumerate(top.dzs):
                for j in range(i + 1, len(top.dzs)):
                    dzj = top.dzs[j]

                    min_dist = min(
                        (
                            cast(float, top.graph.edges[u, v]["weight"])
                            for u, v in product(dzi.nodes, dzj.nodes)
                            if top.graph.has_edge(u, v)
                        ),
                        default=None,
                    )

                    if min_dist is not None:
                        if req.source in dzi.nodes:
                            min_dist_from_src = min(dists[v] for v in dzj.nodes)
                            min_dist = min_dist * beta + (1 - beta) * min_dist_from_src
                        elif req.source in dzj.nodes:
                            min_dist_from_src = min(dists[v] for v in dzi.nodes)
                            min_dist = min_dist * beta + (1 - beta) * min_dist_from_src
                        weight = int(min_dist + self.inv_alpha)
                        flow_graph.add_edge(
                            sink_nodes[i],
                            source_nodes[j],
                            weight=weight,
                        )
                        flow_graph.add_edge(
                            sink_nodes[j],
                            source_nodes[i],
                            weight=weight,
                        )

            try:
                flow = cast(dict[str, dict[str, int]], nx.min_cost_flow(flow_graph))
                dz_paths = extract_all_flow_paths(flow_graph, "source", "sink", flow)
                dz_set_lists: list[list[set[int]]] = []
                for dz_path in dz_paths:
                    dz_index_list: list[int] = remove_consecutive_duplicates(
                        remove_nones([extract_dz_index(node) for node in dz_path])
                    )
                    dz_set_lists.append([set(top.dzs[i].nodes) for i in dz_index_list])
                for i in range(len(dz_set_lists)):
                    for j in range(len(dz_set_lists)):
                        if i == j:
                            continue
                        dz_route_i, dz_route_j = dz_set_lists[i], dz_set_lists[j]
                        for k1 in range(0, len(dz_route_i)):
                            for k2 in range(1, len(dz_route_j)):
                                dz_route_i[k1].difference_update(dz_route_j[k2])
                routes: list[Route] = []
                free_nodes = init_free_nodes(top, req, dz_set_lists)
                affected_nodes: set[int] = set()
                remaining_dsts = set(dst)
                for dz_set_list in dz_set_lists:
                    for dz_set in dz_set_list:
                        dz_set.difference_update(affected_nodes)
                    path = reconstruct_min_hop_path(
                        top.graph,
                        req.source,
                        remaining_dsts,
                        dz_set_list,
                        free_nodes,
                    )
                    if len(path) == 0:
                        break
                    for node in free_nodes.intersection(path):
                        if any(
                            req.source not in dz.nodes and node in dz.nodes
                            for dz in top.dzs
                        ):
                            free_nodes.remove(node)
                    for i, node in enumerate(path):
                        if node in remaining_dsts:
                            path = path[: i + 1]
                            break
                    route = Route(top, path)
                    dzs = route.affected_dzs()
                    dzs = {dz for dz in dzs if req.source not in dz.nodes}
                    affected_nodes.update(node for dz in dzs for node in dz.nodes)
                    routes.append(Route(top, path))
                    remaining_dsts.remove(path[-1])
                    if len(routes) >= 2:
                        cost = self.route_set_cost(routes, req.bpsk_fs_count)
                        if cost < best_route_set_cost:
                            best_route_set = tuple(routes)
                            best_route_set_cost = cost
            except nx.NetworkXUnfeasible:
                break
            except InfeasibleRouteError:
                pass

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
