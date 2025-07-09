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


@dataclass
class GreedyLocalSearchAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.ls.GreedyLSRoutingAlgorithm"
    _short_: str = "greedy+ls"


@dataclass
class FlowLocalSearchAlgorithmConfig(RoutingAlgorithmConfig):
    _target_: str = "disaster_routing.routing.ls.FlowLSRoutingAlgorithm"
    _short_: str = "flow+ls"


def register_routing_algo_configs():
    cs = ConfigStore.instance()
    cs.store(group="router", name="greedy", node=GreedyRoutingAlgorithmConfig)
    cs.store(group="router", name="greedy+ls", node=GreedyLocalSearchAlgorithmConfig)
    cs.store(group="router", name="flow", node=FlowRoutingAlgorithmConfig)
    cs.store(group="router", name="flow+ls", node=FlowLocalSearchAlgorithmConfig)
