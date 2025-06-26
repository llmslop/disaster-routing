from typing import Callable


class BestRecord[T, V]:
    value: T | None
    cost: V | None
    objective_fn: Callable[[T], V]

    def __init__(self, objective_fn: Callable[[T], V]):
        self.value = None
        self.cost = None
        self.objective_fn = objective_fn

    def update(self, value: T) -> V:
        obj = self.objective_fn(value)
        if self.cost is None or obj < self.cost:
            self.value, self.cost = value, obj
        return obj

    def get(self) -> tuple[T | None, V | None]:
        return self.value, self.cost
