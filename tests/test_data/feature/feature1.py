from simple_backtester import Feature
from typing import Union, Any
import pandas as pd
import numpy as np
from numpy.typing import NDArray


class StrangeFeature(Feature):
    def eval(
        self, data: Union[NDArray[np.float64], pd.Series[Any]]
    ) -> Union[NDArray[np.float64], pd.Series[np.float64]]:
        self.func = lambda x: np.exp(x)
        self.feature: Union[NDArray[np.float64], pd.Series[np.float64]] = self.func(
            data
        )
        return self.feature
