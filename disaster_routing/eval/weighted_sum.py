from math import isnan
from typing import override

from ..instances.instance import Instance
from .evaluator import Evaluator


class WeightedSumEvaluator(Evaluator):
    total_fs_weight: float
    mofi_weight: float

    def __init__(self, total_fs_weight: float, mofi_weight: float, **kwargs: object):
        if isnan(total_fs_weight) and isnan(mofi_weight):
            raise ValueError("Weighted sum evaluator weights must not be both NaN")
        if (isnan(total_fs_weight) and mofi_weight > 1.0) or (
            isnan(mofi_weight) and total_fs_weight > 1.0
        ):
            raise ValueError("Automatic weight must not exceed 1.0")
        self.total_fs_weight = (
            total_fs_weight if not isnan(total_fs_weight) else (1.0 - mofi_weight)
        )
        self.mofi_weight = (
            mofi_weight if not isnan(mofi_weight) else (1.0 - total_fs_weight)
        )

    @override
    def evaluate(self, total_fs: int, mofi: int) -> float:
        return total_fs * self.total_fs_weight + mofi * self.mofi_weight

    @override
    def get_weights(self) -> tuple[float, float] | None:
        return self.total_fs_weight, self.mofi_weight
