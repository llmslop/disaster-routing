import random
import numpy as np
from .random import Random


class UnseededRandom(Random):
    def __init__(self):
        super().__init__(stdlib=random.Random(), numpy=np.random.default_rng())
