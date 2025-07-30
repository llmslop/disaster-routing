from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ..random.config import RandomConfig, register_random_configs


@dataclass
class ContentPlacementConfig:
    pass


@dataclass
class NaiveContentPlacementConfig(ContentPlacementConfig):
    _target_: str = "disaster_routing.placement.naive.NaiveContentPlacement"


@dataclass
class StochasticContentPlacementConfig(ContentPlacementConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.placement.stochastic.StochasticContentPlacement"
    random: RandomConfig = MISSING


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

    register_random_configs("content_placement")
