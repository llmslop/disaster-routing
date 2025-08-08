from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from disaster_routing.random.config import RandomConfig, register_random_configs


@dataclass
class RoutingAlgorithmConfig:
    pass


@dataclass
class GreedyRoutingAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.greedy.GreedyRoutingAlgorithm"
    _short_: str = "greedy"


@dataclass
class FlowRoutingAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.flow.FlowRoutingAlgorithm"
    _short_: str = "flow"


@dataclass
class GreedyLocalSearchAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.ls.GreedyLSRoutingAlgorithm"
    _short_: str = "greedy+ls"


@dataclass
class FlowLocalSearchAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.ls.FlowLSRoutingAlgorithm"
    _short_: str = "flow+ls"


@dataclass
class SGARoutingAlgorithmConfig(RoutingAlgorithmConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.routing.sga.SGARoutingAlgorithm"
    _short_: str = "sga"
    random: RandomConfig = MISSING
    num_gens: int = 100
    pop_size: int = 100
    cr_rate: float = 0.7
    mut_rate: float = 0.1
    cr_num_retries_per_req: int = 10
    mut_num_retries_per_req: int = 10
    elitism_rate: float = 0.04


def register_routing_algo_configs(group: str = "router"):
    cs = ConfigStore.instance()
    cs.store(group=group, name="greedy", node=GreedyRoutingAlgorithmConfig)
    cs.store(group=group, name="greedy+ls", node=GreedyLocalSearchAlgorithmConfig)
    cs.store(group=group, name="flow", node=FlowRoutingAlgorithmConfig)
    cs.store(group=group, name="flow+ls", node=FlowLocalSearchAlgorithmConfig)
    cs.store(group=group, name="sga", node=SGARoutingAlgorithmConfig)

    register_random_configs(group)
