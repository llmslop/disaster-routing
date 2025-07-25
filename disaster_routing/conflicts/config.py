from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class DSASolverConfig:
    pass


@dataclass
class GADSASolverConfig(DSASolverConfig):
    _target_: str = "disaster_routing.conflicts.ga.GADSASolver"
    pop_size: int = 100
    generations: int = 20
    mutation_rate: float = 0.2
    lru_size: int = 10000


@dataclass
class NPMDSASolverConfig(DSASolverConfig):
    _target_: str = "disaster_routing.conflicts.npm.NPMDSASolver"
    lru_size: int = 10000
    iter_count: int = 1000
    num_sampled_points: int = 10


@dataclass
class FPGADSASolverConfig(DSASolverConfig):
    _target_: str = "disaster_routing.conflicts.npm.FPGADSASolver"
    num_attempts: int = 5


def register_dsa_solver_configs():
    cs = ConfigStore.instance()
    cs.store(group="dsa_solver", name="ga", node=GADSASolverConfig)
    cs.store(group="dsa_solver", name="fpga", node=FPGADSASolverConfig)
    cs.store(group="dsa_solver", name="npm", node=NPMDSASolverConfig)
