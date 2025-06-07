from ..topologies.topology import Topology
from .request import Request


class Instance:
    topology: Topology
    requests: list[Request]

    def __init__(self, topology: Topology, requests: list[Request]):
        self.topology = topology
        self.requests = requests
