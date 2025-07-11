from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class EvaluationConfig:
    pass


@dataclass
class WeightedSumEvaluationConfig(EvaluationConfig):
    _target_: str = "disaster_routing.eval.weighted_sum.WeightedSumEvaluator"
    total_fs_weight: float = 0.5
    mofi_weight: float = float("nan")


def register_evaluator_configs():
    cs = ConfigStore.instance()
    cs.store(group="eval", name="weightedsum", node=WeightedSumEvaluationConfig)
