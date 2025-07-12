class RunningStats:
    n: int
    mean: float
    m2: float

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def avg(self) -> float:
        return self.mean

    def variance(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.m2 / (self.n - 1)

    def stddev(self) -> float:
        return self.variance() ** 0.5

    def info(self) -> object:
        return {"mean": self.mean, "variance": self.variance(), "stddev": self.stddev()}
