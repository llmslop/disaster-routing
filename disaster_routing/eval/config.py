from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ..conflicts.config import (
    DSASolverConfig,
    register_dsa_solver_configs,
)
from ..routing.config import (
    RoutingAlgorithmConfig,
    register_routing_algo_configs,
)


@dataclass
class EvaluationConfig:
    pass


@dataclass
class WeightedSumEvaluationConfig(EvaluationConfig):
    _target_: str = "disaster_routing.eval.weighted_sum.WeightedSumEvaluator"
    total_fs_weight: float = 0.5
    mofi_weight: float = float("nan")


@dataclass
class RelativeEvaluationConfig(EvaluationConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"approximate_dsa_solver": "ga"},
            {"router": "greedy"},
        ]
    )
    _recursive_: bool = False
    _target_: str = "disaster_routing.eval.relative.RelativeEvaluator"
    total_fs_weight: float = 0.5
    mofi_weight: float = float("nan")
    approximate_dsa_solver: DSASolverConfig = MISSING
    router: RoutingAlgorithmConfig = MISSING


def register_evaluator_configs():
    cs = ConfigStore.instance()
    cs.store(group="eval", name="weightedsum", node=WeightedSumEvaluationConfig)
    cs.store(group="eval", name="relative", node=RelativeEvaluationConfig)

    register_routing_algo_configs("eval/router")
    register_dsa_solver_configs("eval/approximate_dsa_solver")
