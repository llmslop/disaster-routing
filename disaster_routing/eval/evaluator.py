from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, total_fs: int, mofi: int) -> float: ...
