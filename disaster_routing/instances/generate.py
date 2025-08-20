import json
from dataclasses import dataclass, field
from math import ceil
from typing import Any, cast

from hydra.utils import instantiate
from omegaconf import MISSING

from disaster_routing.instances.instance import Instance
from disaster_routing.instances.request import Request
from disaster_routing.random.config import RandomConfig
from disaster_routing.random.random import Random
from disaster_routing.topologies.topologies import get_topology
from disaster_routing.topologies.topology import Topology
from disaster_routing.utils.ilist import ilist


class InstanceGenerator:
    topology: Topology
    possible_dc_positions: ilist[int]
    content_count: int
    transmission_rate_range: tuple[float, float]
    random: Random

    def __init__(
        self,
        random: Random,
        topology: Topology,
        possible_dc_positions: ilist[int],
        content_count: int = 10,
        transmission_rate_range: tuple[float, float] = (0, 10),
    ):
        self.random = random
        self.topology = topology
        self.possible_dc_positions = possible_dc_positions
        self.content_count = content_count
        self.transmission_rate_range = transmission_rate_range

    def gen_requests(self, n: int) -> list[Request]:
        source_nodes: set[int] = set(self.topology.graph)
        source_nodes.difference_update(self.possible_dc_positions)

        sources = self.random.stdlib.choices(list(source_nodes), k=n)
        contents = [
            self.random.stdlib.randint(0, self.content_count - 1) for _ in range(n)
        ]
        trans_rate = [
            self.random.stdlib.random()
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

    def gen_instance(self, n: int, dc_count: int) -> Instance:
        return Instance(
            self.topology, self.gen_requests(n), self.possible_dc_positions, dc_count
        )


@dataclass
class InstanceGeneratorConfig:
    defaults: list[Any] = field(default_factory=lambda: [{"random": "unseeded"}])
    random: RandomConfig = MISSING
    num_requests: int = 10
    topology_name: str = "nsfnet"
    possible_dc_positions: ilist[int] = (2, 5, 6, 9, 11)
    content_count: int = 10
    dc_count: int = 3
    transmission_rate_range: tuple[float, float] = (0, 10)
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
        generator = InstanceGenerator(
            instantiate(config.random),
            get_topology(config.topology_name),
            config.possible_dc_positions,
            config.content_count,
            config.transmission_rate_range,
        )
        instance = generator.gen_instance(config.num_requests, config.dc_count)
        with open(config.path, "w") as f:
            obj = instance.to_json()
            json.dump(obj, f)
            _ = f.write("\n")
        return instance
