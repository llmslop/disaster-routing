from abc import ABC, abstractmethod

from ..instances.instance import Instance


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, total_fs: int, mofi: int) -> float: ...

    def get_weights(self, _inst: Instance) -> tuple[float, float] | None:
        return None
