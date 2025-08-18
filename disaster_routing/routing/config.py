from dataclasses import dataclass

from hydra.core.config_store import ConfigStore



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


def register_routing_algo_configs(group: str = "router"):
    cs = ConfigStore.instance()
    cs.store(group=group, name="greedy", node=GreedyRoutingAlgorithmConfig)
    cs.store(group=group, name="flow", node=FlowRoutingAlgorithmConfig)
