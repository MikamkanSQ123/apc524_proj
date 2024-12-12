from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, final, Any, Union
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from .utils.yaml_helper import YamlParser


@dataclass(frozen=True)
class SetupConfig:
    start_date: str
    end_date: str
    warm_up: int
    look_back: int
    initial_capital: float
    universe: List[str]
    features: List[str]


@dataclass(frozen=True)
class RiskConfig:
    stop_loss: float = float("inf")
    cool_down: int = 0
    take_profit: float = float("inf")


class Strategy(ABC):
    """
    Abstract base class for defining a trading strategy.

    This class provides a framework for implementing trading strategies with
    configurable setup, parameters, and risk management. It enforces mandatory
    sections in the configuration file and initializes the strategy accordingly.

    Attributes:
        setup (SetupConfig): Configuration for the setup section, such as the trading universe.
        parameters (ParametersConfig): Configuration for strategy parameters.
        risk (RiskConfig): Configuration for risk management settings like stop loss and take profit.
        __pnl (list[float]): List tracking profit and loss over time.
        __running_pnl (float): Cumulative profit and loss.
        __cool_downcount (int): Remaining cool-down period in evaluations.

    Methods:
        __init__(config_path: Union[str, Path]):
            Initializes the strategy using the configuration file at the specified path.

        evaluate() -> NDArray[np.float64]:
            Abstract method to compute strategy weights. Must be implemented by subclasses.

        eval() -> NDArray[np.float64]:
            Evaluates the strategy while applying risk management rules, including stop loss,
            take profit, and cool-down periods.
    """

    @final
    def __init__(self, config: dict[str, Any]):
        # if isinstance(config_path, str):
        #     config_path = Path(config_path)
        # config: dict[str, Any] = yaml.safe_load(config_path.read_text())

        required_sections = {"setup"}
        missing_sections = required_sections - config.keys()
        if missing_sections:
            raise ValueError(f"Missing mandatory sections in YAML: {missing_sections}")

        self.features = SimpleNamespace()
        self._setup = SetupConfig(**config["setup"])
        self._parameters = SimpleNamespace()
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
            return np.zeros_like(self.setup.universe, dtype=np.float64)

        self.__running_pnl += self.__pnl[-1]

        # Trigger stop loss or take profit
        if (
            self.__running_pnl < -self.risk.stop_loss
            or self.__running_pnl > self.risk.take_profit
        ):
            self.__running_pnl = 0
            self.__cool_downcount = self.risk.cool_down
            return np.zeros_like(self.setup.universe, dtype=np.float64)

        # Compute weights as user defined
        return self.evaluate()

    @property
    def setup(self) -> SetupConfig:
        return self._setup

    @property
    def parameters(self) -> SimpleNamespace:
        return self._parameters

    @property
    def risk(self) -> RiskConfig:
        return self._risk

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> List["Strategy"]:
        return [cls(config) for config in YamlParser(config_path).load_yaml_matrix()]
