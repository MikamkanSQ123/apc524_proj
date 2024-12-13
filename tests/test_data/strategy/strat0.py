from simple_backtester import Strategy
import numpy as np


class Zombie(Strategy):
    def evaluate(self):
        return np.zeros_like(self.setup.universe, dtype=float)
