import yaml
from abc import ABC, abstractmethod
from typing import final, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.typing import NDArray


class Feature(ABC):
    "Abstract class for self-designed features (may also include simple features)"

    @final
    def __init__(self, config_path: Union[str, Path]):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        config: dict[str, Any] = yaml.safe_load(config_path.read_text())

        required_sections = {"feature_parameters"}
        missing_sections = required_sections - config.keys()
        if missing_sections:
            raise ValueError(f"Missing mandatory sections in YAML: {missing_sections}")

        self.__freq = config["setup"]["frequency"]
        self.__featurparams = config["feature_parameters"]

    @abstractmethod
    def eval(
        self, data: Union[NDArray[np.float64], "pd.Series[Any]", pd.DataFrame]
    ) -> Any:
        "Some evaluation function"
        self.func = lambda x: np.exp(x)
        self.feature: Any = self.func(data)  # type: ignore[no-untyped-call]
        return self.feature


class Techlib(object):
    "Class for technical indicators"

    def __init__(self) -> None:
        pass

    @staticmethod
    def ma(
        data: Union["pd.Series[Any]", pd.DataFrame], window: int
    ) -> Union["pd.Series[Any]", pd.DataFrame]:
        return data.rolling(window=window).mean()

    @staticmethod
    def macd(
        data: Union["pd.Series[Any]", pd.DataFrame], window1: int, window2: int
    ) -> Union["pd.Series[Any]", pd.DataFrame]:
        wd1, wd2 = min(window1, window2), max(window1, window2)
        return Techlib.ma(data, wd1) - Techlib.ma(data, wd2)

    @staticmethod
    def rsi(
        data: Union["pd.Series[Any]", pd.DataFrame], window: int = 14
    ) -> Union["pd.Series[Any]", pd.DataFrame]:
        "Input should be price data"
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # type: ignore[arg-type, operator]
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # type: ignore[arg-type, operator]
        rs = gain / loss
        return 100 - (100 / (1 + rs))
