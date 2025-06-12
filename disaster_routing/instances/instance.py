from typing import cast
from ..topologies.topology import Topology
from .request import Request


class Instance:
    topology: Topology
    requests: list[Request]

    def __init__(self, topology: Topology, requests: list[Request]):
        self.topology = topology
        self.requests = requests

    @staticmethod
    def from_json(json: dict[str, object]) -> "Instance":
        return Instance(
            Topology.from_json(cast(dict[str, object], json["topology"])),
            [
                Request.from_json(req)
                for req in cast(list[dict[str, object]], json["requests"])
            ],
        )

    def to_json(self):
        return {
            "topology": self.topology.to_json(),
            "requests": [req.to_json() for req in self.requests],
        }
