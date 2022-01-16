import functools

import numpy as np


@functools.total_ordering
class RangedNumber():
    def __init__(self, min_, max_, start_value):
        self.n = start_value
        self.min = min_
        self.max = max_

    def __add__(self, other):
        if isinstance(other, RangedNumber):
            other = other.n
        return RangedNumber(self.min, self.max, np.clip(self.n + other, self.min, self.max))

    def __radd__(self, other):
        return np.clip(other + self.n, self.min, self.max)

    def __sub__(self, other):
        if isinstance(other, RangedNumber):
            other = other.n
        return RangedNumber(self.min, self.max, np.clip(self.n - other, self.min, self.max))

    def __rsub__(self, other):
        return np.clip(other - self.n, self.min, self.max)

    def __eq__(self, other):
        if isinstance(other, RangedNumber):
            other = other.n
        return self.n == other

    def __lt__(self, other):
        if isinstance(other, RangedNumber):
            other = other.n
        return self.n < other

    def __int__(self):
        return int(self.n)

    def __str__(self):
        return str(self.n)

    def __repr__(self):
        return self.n.__repr__()