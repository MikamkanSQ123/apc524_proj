from simple_backtester import Strategy
import numpy as np


class MeanReversion(Strategy):
    def evaluate(self):
        data = self.features.data
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        ma_base = np.mean(data[-self.parameters.ma_base :], axis=0)
        ma_window = np.vstack(
            [np.mean(data[-w:], axis=0) for w in self.parameters.ma_window]
        )
        signal = np.dot((ma_window - ma_base).T, self.parameters.cross_weight)
        return (signal - np.mean(signal)) / np.std(signal)
