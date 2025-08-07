from .nsfnet import nsfnet
from .cost239 import cost239
from .topology import Topology


def get_topology(name: str) -> Topology:
    match name:
        case "nsfnet":
            return nsfnet()
        case "cost239":
            return cost239()
        case _:
            raise ValueError(f"Invalid topology name: {name}")
