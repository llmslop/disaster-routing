from collections.abc import Iterable
from itertools import combinations, count

from disaster_routing.routing.routing_algo import InfeasibleRouteError, Route

from ..instances.request import Request
from ..random.random import Random
from ..topologies.topology import Topology
from ..utils.ilist import ilist
from ..topologies.graphs import Graph

import networkx as nx


class DistMap:
    dist_map: dict[ilist[int], dict[int, float]]

    def __init__(self, dist_map: dict[ilist[int], dict[int, float]]) -> None:
        self.dist_map = dist_map

    @staticmethod
    def generate_dist_map(graph: Graph, dcs: set[int]) -> "DistMap":
        dc_power_set = {
            tuple(sorted(subset))
            for r in range(1, len(dcs) + 1)
            for subset in combinations(dcs, r)
        }

        return DistMap(
            {
                dc_subset: nx.multi_source_dijkstra_path_length(
                    graph, dc_subset, weight=None
                )
                for dc_subset in dc_power_set
            }
        )

    def get_for_dests(self, dests: Iterable[int]):
        key = tuple(sorted(dests))
        return self.dist_map[key]


class PathGenerator:
    top: Topology
    content_placement: dict[int, set[int]]
    dist_map: DistMap

    def __init__(self, top: Topology, content_placement: dict[int, set[int]]) -> None:
        self.top = top
        self.content_placement = content_placement
        self.dist_map = DistMap.generate_dist_map(
            top.graph, {dc for dcs in content_placement.values() for dc in dcs}
        )

    def randomized_dfs(
        self, random: Random, start: int, end: set[int], graph: Graph | None
    ) -> ilist[int] | None:
        if len(end) == 0:
            return None
        graph = graph or self.top.graph
        visited: set[int] = set()
        path: list[int] = []
        dist_map = self.dist_map.get_for_dests(end)

        def dfs(start: int) -> list[int] | None:
            visited.add(start)
            path.append(start)

            if start in end:
                return path

            neighbors = list(graph.adj[start])
            # TODO: make this configurable
            neighbors.sort(
                key=lambda node: dist_map[node] * 0.5 + random.stdlib.random()
            )

            for neighbor in neighbors:
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result is not None:
                        return result

            _ = path.pop()
            visited.remove(start)
            return None

        nodes = dfs(start)
        return tuple(nodes) if nodes is not None else None

    def shorten_route(self, route: ilist[int], dcs: set[int]) -> ilist[int]:
        k = next(i for i in range(len(route)) if route[i] in dcs)
        return route[: k + 1]

    def generate_request_route(
        self,
        random: Random,
        req: Request,
        routes: ilist[Route],
        dcs: set[int],
    ) -> Route | None:
        top = self.top.copy()
        for dz in top.dzs:
            if (
                any(dz.affects_path(route.node_list) for route in routes)
                and req.source not in dz.nodes
            ):
                dz.remove_from_graph(top.graph)
        avail_dcs = set(dcs)
        while len(dcs) > 0:
            route = self.randomized_dfs(random, req.source, avail_dcs, graph=top.graph)
            if route is None:
                return None
            route = self.shorten_route(route, dcs)
            try:
                return Route(self.top, route)
            except InfeasibleRouteError:
                if route[-1] in avail_dcs:
                    avail_dcs.remove(route[-1])
        return None

    def generate_request_routes(
        self,
        random: Random,
        req: Request,
        num_retries: int | None = None,
        early_term_chance: float = 0.25,
    ) -> tuple[ilist[Route] | None, int]:
        for retry in range(num_retries) if num_retries is not None else count():
            routes: ilist[Route] = ()
            dcs = set(self.content_placement[req.content_id])

            while len(dcs) > 0:
                route = self.generate_request_route(random, req, routes, dcs)
                if route is None:
                    break
                routes = routes + (route,)
                dcs.remove(route.node_list[-1])
                if len(routes) >= 2 and random.stdlib.random() < early_term_chance:
                    break
            if len(routes) >= 2:
                return routes, retry
        return None, 0
