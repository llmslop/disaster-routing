from disaster_routing.topologies.nsfnet import nsfnet
from disaster_routing.topologies.cost239 import cost239
from disaster_routing.topologies.topology import Topology


def get_topology(name: str) -> Topology:
    match name:
        case "nsfnet":
            return nsfnet()
        case "cost239":
            return cost239()
        case _:
            raise ValueError(f"Invalid topology name: {name}")
