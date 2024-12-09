import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, final, Any, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@dataclass
class SetupConfig:
    start_date: str
    end_date: str
    warm_up: int
    look_back: int
    initial_capital: float
    universe: List[str]


class ParametersConfig:
    pass


@dataclass
class RiskConfig:
    stop_loss: float = float("inf")
    cool_down: int = 0
    take_profit: float = float("inf")


class Strategy(ABC):
    """
    Abstract class for strategies
    """

    @final
    def __init__(self, config_path: Union[str, Path]):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        config: dict[str, Any] = yaml.safe_load(config_path.read_text())

        required_sections = {"setup"}
        missing_sections = required_sections - config.keys()
        if missing_sections:
            raise ValueError(f"Missing mandatory sections in YAML: {missing_sections}")

        self._setup = SetupConfig(**config["setup"])
        self._parameters = ParametersConfig()
        if "parameters" in config:
            for key, value in config["parameters"].items():
                setattr(self.parameters, key, value)
        self._risk = RiskConfig(**config["risk"]) if "risk" in config else RiskConfig()

        self.__pnl = [0]
        self.__running_pnl = 0
        self.__cool_downcount = 0

    @abstractmethod
    def evaluate(self) -> NDArray[np.float64]:
        pass

    @final
    def eval(self) -> NDArray[np.float64]:
        # Check if we are in cool down period
        if self.__cool_downcount > 0:
            self.__cool_downcount -= 1
            return np.zeros_like(self.setup.universe)

        self.__running_pnl += self.__pnl[-1]

        # Trigger stop loss or take profit
        if (
            self.__running_pnl < -self.risk.stop_loss
            or self.__running_pnl > self.risk.take_profit
        ):
            self.__running_pnl = 0
            self.__cool_downcount = self.risk.cool_down
            return np.zeros_like(self.setup.universe)

        # Compute weights as user defined
        return self.evaluate()

    @property
    def setup(self) -> SetupConfig:
        return self._setup

    @property
    def parameters(self) -> ParametersConfig:
        return self._parameters

    @property
    def risk(self) -> RiskConfig:
        return self._risk
