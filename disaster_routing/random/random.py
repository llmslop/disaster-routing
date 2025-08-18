import random

import numpy as np


class Random:
    stdlib: random.Random
    numpy: np.random.Generator

    def __init__(self, stdlib: random.Random, numpy: np.random.Generator):
        self.stdlib = stdlib
        self.numpy = numpy
