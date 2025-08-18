from abc import ABC, abstractmethod

from disaster_routing.solver.solution import CDPSolution


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, total_fs: int, mofi: int) -> float: ...

    def get_weights(self) -> tuple[float, float] | None:
        return None

    def evaluate_solution(self, sol: CDPSolution) -> float:
        return self.evaluate(sol.total_fs(), sol.mofi())
