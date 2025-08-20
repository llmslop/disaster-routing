from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import override

from disaster_routing.instances.instance import Instance
from disaster_routing.instances.modulation import ModulationFormat
from disaster_routing.instances.request import Request
from disaster_routing.topologies.topology import Topology
from disaster_routing.utils.ilist import ilist


class InfeasibleRouteError(Exception):
    def __init__(self):
        super().__init__("No route found via routing algorithm")


class Route:
    top: Topology
    node_list: ilist[int]
    format: ModulationFormat

    def __init__(
        self,
        top: Topology,
        node_list: ilist[int],
        format: ModulationFormat | None = None,
    ):
        self.top = top
        self.node_list = node_list
        format = (
            ModulationFormat.best_rate_format_with_distance(self.distance())
            if format is None
            else format
        )
        if format is None:
            raise InfeasibleRouteError
        self.format = format

    def distance(self) -> int:
        return sum(self.top.graph.edges[u, v]["weight"] for u, v in self.edges())

    def edges(self) -> list[tuple[int, int]]:
        return [
            (self.node_list[i], self.node_list[i + 1])
            for i in range(len(self.node_list) - 1)
        ]

    def has_edge(self, edge: tuple[int, int]) -> bool:
        return any(
            self.node_list[i] == edge[0] and self.node_list[i + 1] == edge[1]
            for i in range(len(self.node_list) - 1)
        )

    @override
    def __repr__(self) -> str:
        return str(self.node_list)

    @override
    def __hash__(self) -> int:
        return self.node_list.__hash__()

    @override
    def __eq__(self, value: object, /) -> bool:
        return self.node_list == value.node_list if isinstance(value, Route) else False


class RouteInfo:
    route: Route
    num_fs: int
    start_idx: int

    def __init__(self, route: Route, num_fs: int, start_idx: int):
        self.route = route
        self.num_fs = num_fs
        self.start_idx = start_idx


class RoutingAlgorithm(ABC):
    @abstractmethod
    def route_request(
        self, req: Request, top: Topology, dst: set[int]
    ) -> ilist[Route]: ...

    @abstractmethod
    def name(self) -> str: ...

    @staticmethod
    def check_solution(req: Request, dst: Iterable[int], routes: ilist[Route]):
        tops = set(route.top for route in routes)
        assert len(tops) == 1

        top = list(tops)[0]
        assert all(route.node_list[0] == req.source for route in routes)
        assert all(route.node_list[-1] in dst for route in routes)

        source_dz = set(dz for dz in top.dzs if req.source in dz.nodes)
        dzs = [
            set(dz for dz in route.top.dzs if dz.affects_path(route.node_list))
            for route in routes
        ]

        for i in range(len(dzs)):
            for j in range(i + 1, len(dzs)):
                dz1, dz2 = dzs[i], dzs[j]
                assert dz1.intersection(dz2).issubset(source_dz)

    @staticmethod
    def num_avg_hops(routes: list[Route]) -> float:
        return mean(len(route.node_list) - 1 for route in routes)

    def route_instance(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> ilist[ilist[Route]]:
        return tuple(
            self.route_request(req, inst.topology, content_placement[req.content_id])
            for req in inst.requests
        )

    @staticmethod
    def sort_routes(
        all_routes: ilist[ilist[Route]],
        num_fses: dict[int, int] | None = None,
        start_indices: dict[int, int] | None = None,
    ) -> ilist[ilist[Route]]:
        if start_indices is None:
            start_indices = defaultdict(int)
        if num_fses is None:
            num_fses = defaultdict(int)
        all_route_infos: list[list[RouteInfo]] = []
        idx = 0
        for routes in all_routes:
            route_infos: list[RouteInfo] = []
            for route in routes:
                route_infos.append(RouteInfo(route, num_fses[idx], start_indices[idx]))
                idx += 1
            route_infos.sort(key=lambda r: r.route.node_list)
            all_route_infos.append(route_infos)

        start_indices.clear()
        num_fses.clear()
        sorted_routes: ilist[ilist[Route]] = ()
        idx = 0
        for route_infos in all_route_infos:
            routes = tuple(ri.route for ri in route_infos)
            for ri in route_infos:
                start_indices[idx] = ri.start_idx
                num_fses[idx] = ri.num_fs
                idx += 1
            sorted_routes += (routes,)
        return sorted_routes
