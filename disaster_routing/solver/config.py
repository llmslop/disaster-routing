from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from disaster_routing.conflicts.config import (
    DSASolverConfig,
    register_dsa_solver_configs,
)
from disaster_routing.random.config import RandomConfig, register_random_configs
from disaster_routing.routing.config import (
    RoutingAlgorithmConfig,
    register_routing_algo_configs,
)


@dataclass
class CDPSolverConfig:
    pass


@dataclass
class TwoPhaseSolverConfig(CDPSolverConfig):
    _recursive_: bool = False
    _target_: str = "disaster_routing.solver.two_phase.TwoPhaseSolver"
    router: RoutingAlgorithmConfig = MISSING
    dsa_solver: DSASolverConfig = MISSING


@dataclass
class LSSolverConfig(CDPSolverConfig):
    _recursive_: bool = False
    _target_: str = "disaster_routing.solver.ls.LSSolver"
    base: CDPSolverConfig = MISSING
    f_max: int = 100


@dataclass
class ELSSolverConfig(CDPSolverConfig):
    _recursive_: bool = False
    _target_: str = "disaster_routing.solver.ls.ELSSolver"
    base: CDPSolverConfig = MISSING
    f_max: int = 100


@dataclass
class ILPSolverConfig(CDPSolverConfig):
    _target_: str = "disaster_routing.solver.ilp.ILPCDPSolver"
    msg: bool = False


@dataclass
class SGASolverConfig(CDPSolverConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"random": "seeded"},
        ]
    )
    _target_: str = "disaster_routing.solver.sga.SGASolver"
    random: RandomConfig = MISSING
    approximate_dsa_solver: DSASolverConfig = MISSING
    dsa_solver: DSASolverConfig = MISSING
    num_gens: int = 100
    pop_size: int = 100
    cr_rate: float = 0.7
    mut_rate: float = 0.1
    cr_num_retries_per_req: int = 10
    mut_num_retries_per_req: int = 10
    elitism_rate: float = 0.04


def register_solver_configs(group: str = "solver", recursive_limit: int = 3):
    cs = ConfigStore.instance()
    cs.store(group=group, name="two_phase", node=TwoPhaseSolverConfig)
    cs.store(group=group, name="ilp", node=ILPSolverConfig)
    cs.store(group=group, name="sga", node=SGASolverConfig)
    cs.store(group=group, name="ls", node=LSSolverConfig)
    cs.store(group=group, name="els", node=ELSSolverConfig)

    register_routing_algo_configs(f"{group}.router")
    register_dsa_solver_configs(f"{group}.dsa_solver")
    register_dsa_solver_configs(f"{group}.approximate_dsa_solver")
    register_random_configs(f"{group}/random")

    if recursive_limit > 0:
        register_solver_configs(f"{group}.base", recursive_limit=recursive_limit - 1)
