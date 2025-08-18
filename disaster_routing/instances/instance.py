from copy import deepcopy
from typing import cast

from disaster_routing.utils.ilist import ilist
from disaster_routing.topologies.topology import Topology
from disaster_routing.instances.request import Request


class Instance:
    topology: Topology
    requests: list[Request]
    possible_dc_positions: ilist[int]
    dc_count: int

    def __init__(
        self,
        topology: Topology,
        requests: list[Request],
        possible_dc_positions: ilist[int],
        dc_count: int,
    ):
        self.topology = topology
        self.requests = requests
        self.possible_dc_positions = possible_dc_positions
        self.dc_count = dc_count

    def copy(self) -> "Instance":
        return Instance(
            self.topology.copy(),
            deepcopy(self.requests),
            self.possible_dc_positions,
            self.dc_count,
        )

    def remove_edge(self, edge: tuple[int, int]) -> "Instance":
        inst = self.copy()
        u, v = edge
        inst.topology.graph.remove_edge(u, v)
        inst.topology.graph.remove_edge(v, u)
        return inst

    @staticmethod
    def from_json(json: dict[str, object]) -> "Instance":
        return Instance(
            Topology.from_json(cast(dict[str, object], json["topology"])),
            [
                Request.from_json(req)
                for req in cast(list[dict[str, object]], json["requests"])
            ],
            ilist[int](cast(list[int], json["possible_dc_positions"])),
            cast(int, json["dc_count"]),
        )

    def to_json(self):
        return {
            "topology": self.topology.to_json(),
            "requests": [req.to_json() for req in self.requests],
            "possible_dc_positions": list(self.possible_dc_positions),
            "dc_count": self.dc_count,
        }
