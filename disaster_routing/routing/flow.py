from math import ceil
from typing import cast, override
from collections import defaultdict
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .routing_algo import RoutingAlgorithm, Route
from ..instances.modulation import ModulationFormat
from ..instances.request import Request
from ..topologies.topology import Topology


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


def cast_node_type(node: int) -> str:
    return cast(str, cast(object, node))


def extract_all_flow_paths(G, source, sink, flow_dict=None):
    if flow_dict is None:
        flow_dict = nx.min_cost_flow(G)

    # Convert flow_dict into a modifiable flow graph
    flow_graph = {u: dict(v) for u, v in flow_dict.items()}

    paths = []

    def find_path():
        path = [source]
        current = source
        while current != sink:
            for neighbor, flow in flow_graph[current].items():
                if flow > 0:
                    path.append(neighbor)
                    flow_graph[current][neighbor] -= 1  # consume flow
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


def reconstruct_min_hop_path(G, group_path) -> list[int]:
    # group_path: list of sets of nodes (G1, G2, ..., Gk)
    # Returns: (min_path, hop_count)

    # For each node in the current group, store:
    # (min_total_hops_to_get_here, path_so_far)
    dp = {v: (0, [v]) for v in group_path[0]}

    for i in range(1, len(group_path)):
        next_dp = {}
        for u in group_path[i]:
            best = None
            for v in group_path[i - 1]:
                if v in dp:
                    try:
                        path = nx.shortest_path(G, dp[v][1][-1], u)
                        hops = len(path) - 1
                        total_hops = dp[v][0] + hops
                        if best is None or total_hops < best[0]:
                            best = (total_hops, dp[v][1] + path[1:])
                    except nx.NetworkXNoPath:
                        continue
            if best:
                next_dp[u] = best
        dp = next_dp

        if not dp:
            return []  # no path possible

    # Get the best among last group
    end_node, (total_hops, path) = min(dp.items(), key=lambda x: x[1][0])
    return path


class FlowRoutingAlgorithm(RoutingAlgorithm):
    @override
    def route_request(self, req: Request, top: Topology, dst: list[int]) -> list[Route]:
        assert len(dst) >= 2

        routes: list[Route] = []
        best_route_set: list[Route] | None = None
        best_route_set_cost = np.inf

        for K in range(1, len(dst) + 1):
            flow_graph = nx.DiGraph()

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
                    flow_graph.add_edge(sink_nodes[i], "sink", capacity=1)

            for i, dzi in enumerate(top.dzs):
                for j in range(i + 1, len(top.dzs)):
                    dzj = top.dzs[j]

                    if any(
                        top.graph.has_edge(u, v) for u, v in zip(dzi.nodes, dzj.nodes)
                    ):
                        flow_graph.add_edge(sink_nodes[i], source_nodes[j], weight=1)
                        flow_graph.add_edge(sink_nodes[j], source_nodes[i], weight=1)

            try:
                flow = nx.min_cost_flow(flow_graph)
                dz_paths = extract_all_flow_paths(flow_graph, "source", "sink", flow)
                routes: list[Route] = []
                for dz_path in dz_paths:
                    dz_index_list: list[int] = remove_consecutive_duplicates(
                        remove_nones([extract_dz_index(node) for node in dz_path])
                    )
                    dz_set_list = [top.dzs[i].nodes for i in dz_index_list]
                    path = reconstruct_min_hop_path(top.graph, dz_set_list)
                    if len(path) == 0:
                        break
                    routes.append(Route(top, path))
                    self.check_solution(req, dst, routes)
                if len(routes) >= 2:
                    cost = self.route_set_cost(top, routes, req.bpsk_fs_count)
                    if cost < best_route_set_cost:
                        best_route_set = list(routes)
                        best_route_set_cost = cost
            except nx.NetworkXUnfeasible:
                break
            except AssertionError:
                pass

        assert best_route_set is not None
        return best_route_set

    def route_set_cost(
        self,
        top: Topology,
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
