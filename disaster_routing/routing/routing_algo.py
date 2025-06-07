from abc import ABC, abstractmethod
from typing import override

from ..instances.request import Request
from ..topologies.topology import Topology


class Route:
    top: Topology
    node_list: list[int]

    def __init__(self, top: Topology, node_list: list[int]):
        self.top = top
        self.node_list = node_list


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
                print(dz1, dz2, source_dz)
                assert dz1.intersection(dz2).issubset(source_dz)
