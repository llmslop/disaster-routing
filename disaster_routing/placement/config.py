from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class ContentPlacementConfig:
    pass


@dataclass
class NaiveContentPlacementConfig(ContentPlacementConfig):
    _target_: str = "disaster_routing.placement.naive.NaiveContentPlacement"


@dataclass
class StochasticContentPlacementConfig(ContentPlacementConfig):
    _target_: str = "disaster_routing.placement.stochastic.StochasticContentPlacement"


@dataclass
class GreedyContentPlacementConfig(ContentPlacementConfig):
    _target_: str = "disaster_routing.placement.greedy.GreedyContentPlacement"


def register_placement_configs():
    cs = ConfigStore.instance()
    cs.store(group="content_placement", name="naive", node=NaiveContentPlacementConfig)
    cs.store(
        group="content_placement",
        name="stochastic",
        node=StochasticContentPlacementConfig,
    )
    cs.store(
        group="content_placement", name="greedy", node=GreedyContentPlacementConfig
    )
