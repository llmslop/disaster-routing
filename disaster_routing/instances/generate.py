from dataclasses import dataclass
import json
from math import ceil
from random import choices, randint, random
from typing import cast

from .instance import Instance
from ..instances.request import Request
from ..topologies.topology import Topology
from ..topologies.topologies import get_topology


class InstanceGenerator:
    topology: Topology
    content_count: int
    transmission_rate_range: tuple[float, float]

    def __init__(
        self,
        topology: Topology,
        content_count: int = 10,
        transmission_rate_range: tuple[float, float] = (0, 10),
    ):
        self.topology = topology
        self.content_count = content_count
        self.transmission_rate_range = transmission_rate_range

    def gen_requests(self, n: int) -> list[Request]:
        source_nodes: set[int] = set(self.topology.graph)

        sources = choices(list(source_nodes), k=n)
        contents = [randint(0, self.content_count - 1) for _ in range(n)]
        trans_rate = [
            random()
            * (self.transmission_rate_range[1] - self.transmission_rate_range[0])
            + self.transmission_rate_range[0]
            for _ in range(n)
        ]

        return [
            Request(
                sources[i],
                cast(int, self.topology.graph.in_degree[sources[i]]),
                contents[i],
                ceil(trans_rate[i]),
            )
            for i in range(n)
        ]

    def gen_instance(self, n: int) -> Instance:
        return Instance(self.topology, self.gen_requests(n))


@dataclass
class InstanceGeneratorConfig:
    num_requests: int = 10
    topology_name: str = "nsfnet"
    path: str = "instances/temp_instance.json"
    force_recreate: bool = False


def load_or_gen_instance(config: InstanceGeneratorConfig) -> Instance:
    try:
        if config.force_recreate:
            raise IOError()
        with open(config.path, "rb") as f:
            obj = cast(dict[str, object], json.load(f))
            return Instance.from_json(obj)
    except IOError:
        generator = InstanceGenerator(get_topology(config.topology_name))
        instance = generator.gen_instance(config.num_requests)
        with open(config.path, "w") as f:
            obj = instance.to_json()
            json.dump(obj, f)
        return instance
