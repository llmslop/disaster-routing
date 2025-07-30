import random
import numpy as np
from .random import Random


class SeededRandom(Random):
    def __init__(self, seed: int):
        super().__init__(stdlib=random.Random(seed), numpy=np.random.default_rng(seed))
