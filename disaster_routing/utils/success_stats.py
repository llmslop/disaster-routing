class SuccessRateStats:
    successes: int = 0
    failures: int = 0

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def total(self) -> int:
        return self.successes + self.failures

    def mean(self) -> float:
        total = self.total()
        return self.successes / total if total > 0 else 1.0

    def variance(self) -> float:
        if self.total() == 0:
            return 0.0
        p = self.mean()
        return p * (1 - p)

    def stddev(self) -> float:
        return self.variance() ** 0.5

    def info(self) -> object:
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "stddev": self.stddev(),
            "successes": self.successes,
            "failures": self.failures,
            "total": self.total(),
        }
