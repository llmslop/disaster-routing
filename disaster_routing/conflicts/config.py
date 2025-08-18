from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from disaster_routing.random.config import RandomConfig, register_random_configs


@dataclass
class DSASolverConfig:
    pass


@dataclass
class GADSASolverConfig(DSASolverConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.conflicts.ga.GADSASolver"
    pop_size: int = 100
    generations: int = 20
    mutation_rate: float = 0.2
    lru_size: int = 10000
    random: RandomConfig = MISSING


@dataclass
class NPMDSASolverConfig(DSASolverConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.conflicts.npm.NPMDSASolver"
    lru_size: int = 10000
    iter_count: int = 1000
    num_sampled_points: int = 10
    random: RandomConfig = MISSING


@dataclass
class FPGADSASolverConfig(DSASolverConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.conflicts.npm.FPGADSASolver"
    num_attempts: int = 5
    random: RandomConfig = MISSING


def register_dsa_solver_configs(group: str = "dsa_solver"):
    cs = ConfigStore.instance()
    cs.store(group=group, name="ga", node=GADSASolverConfig)
    cs.store(group=group, name="fpga", node=FPGADSASolverConfig)
    cs.store(group=group, name="npm", node=NPMDSASolverConfig)

    register_random_configs(f"{group}/random")
