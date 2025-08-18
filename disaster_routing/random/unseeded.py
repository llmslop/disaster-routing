import random
import numpy as np
from disaster_routing.random.random import Random


class UnseededRandom(Random):
    def __init__(self):
        super().__init__(stdlib=random.Random(), numpy=np.random.default_rng())
