from .nsfnet import nsfnet
from .topology import Topology


def get_topology(name: str) -> Topology:
    match name:
        case "nsfnet":
            return nsfnet()
        case _:
            raise ValueError(f"Invalid topology name: {name}")
