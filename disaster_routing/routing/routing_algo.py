from abc import ABC, abstractmethod
from statistics import mean
from typing import override

from ..instances.instance import Instance
from ..instances.modulation import ModulationFormat
from ..instances.request import Request
from ..topologies.topology import Topology


class InfeasibleRouteError(Exception):
    def __init__(self):
        super().__init__("No route found via routing algorithm")


class Route:
    top: Topology
    node_list: list[int]
    format: ModulationFormat

    def __init__(
        self,
        top: Topology,
        node_list: list[int],
        format: ModulationFormat | None = None,
    ):
        self.top = top
        self.node_list = node_list
        format = (
            ModulationFormat.best_rate_format_with_distance(self.distance())
            if format is None
            else format
        )
        assert format is not None
        self.format = format

    def distance(self) -> int:
        return sum(self.top.graph.edges[u, v]["weight"] for u, v in self.edges())

    def edges(self) -> list[tuple[int, int]]:
        return [
            (self.node_list[i], self.node_list[i + 1])
            for i in range(len(self.node_list) - 1)
        ]

    @override
    def __repr__(self) -> str:
        return str(self.node_list)

    # only copies the node_list
    def shallow_copy(self) -> "Route":
        return Route(self.top, self.node_list.copy(), self.format)


class RoutingAlgorithm(ABC):
    @abstractmethod
    def route_request(
        self, req: Request, top: Topology, dst: list[int]
    ) -> list[Route]: ...

    def check_solution(self, req: Request, dst: list[int], routes: list[Route]):
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

    def num_avg_hops(self, routes: list[Route]) -> float:
        return mean(len(route.node_list) - 1 for route in routes)

    def route_instance(
        self, inst: Instance, content_placement: dict[int, list[int]]
    ) -> list[list[Route]]:
        return [
            self.route_request(req, inst.topology, content_placement[req.content_id])
            for req in inst.requests
        ]
