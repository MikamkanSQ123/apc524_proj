from pathlib import Path
import importlib.util
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import Strategy
from .preprocess.dataloader import DataLoader
from typing import Any, Union, Optional, Type
from datetime import datetime, timedelta


class Backtester:
    def __init__(self, strategy_module_path: str, config_path: Union[str, Path]):
        """
        Initialize the backtester.

        Args:
            strategy_module_path (str): Path to the strategy module file.
            config_path (Union[str, Path]): Path to the configuration file.
        """
        # Dynamically load the strategy class
        self.strategy_class = self.load_strategy_class(strategy_module_path, Strategy)

        # Ensure the loaded class is a concrete subclass of Strategy
        if inspect.isabstract(self.strategy_class):
            raise ValueError(
                f"The loaded strategy class must not be abstract: {self.strategy_class}."
            )

        # Instantiate the strategy
        self.strategy = self.strategy_class(config_path)

        self.start = self.strategy.setup.start_date
        self.end = self.strategy.setup.end_date
        self.lookback = self.strategy.setup.look_back
        self.symbols = self.strategy.setup.universe
        self.features = self.strategy.setup.features

    @staticmethod
    def load_strategy_class(module_path: str, base_class: Type) -> Type[Strategy]:
        """
        Dynamically load a strategy class from a given module.

        Args:
            module_path (str): Path to the Python module file.
            base_class (Type): The base class to filter valid strategy classes.

        Returns:
            Type: The dynamically loaded strategy class.

        Raises:
            ValueError: If no valid strategy class is found or multiple classes are ambiguous.
        """
        spec = importlib.util.spec_from_file_location("strategy_module", module_path)
        if spec is None:
            raise ValueError(f"Cannot load module spec from path: {module_path}")
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ValueError(f"Module spec has no loader for path: {module_path}")
        spec.loader.exec_module(module)

        # Find all classes in the module that subclass the specified base class
        strategy_classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, base_class) and cls is not base_class
        ]

        if not strategy_classes:
            raise ValueError(f"No valid strategy class found in {module_path}.")
        if len(strategy_classes) > 1:
            raise ValueError(
                f"Multiple strategy classes found in {module_path}. Please specify one."
            )

        cls = strategy_classes[0]
        if inspect.isabstract(cls):
            raise ValueError(
                f"The class {cls.__name__} is abstract and cannot be instantiated."
            )

        return cls

    def get_time_n_minutes_before(self, start_time: str, n: int) -> str:
        """
        Calculate the datetime n minutes before a given start time.

        Args:
            start_time (str): The start time in "YYYY-MM-DD HH:MM:SS" format.
            n (int): The number of minutes to subtract.

        Returns:
            str: The datetime n minutes before in "YYYY-MM-DD HH:MM:SS" format.
        """
        start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        result_time = start_time_dt - timedelta(minutes=n)
        return result_time.strftime("%Y-%m-%d %H:%M:%S")

    def run(self, config: dict[str, Any]) -> None:
        """
        Run the backtest with the given price data.

        :param price_data: A numpy array where each column corresponds to a security in the universe.
        :return: None
        """
        start = self.get_time_n_minutes_before(self.start, self.lookback)
        dl = DataLoader(config)
        data: dict[str, Optional[pd.DataFrame]] = dl.load_data(
            start=start, end=self.end, symbols=self.symbols, features=self.features
        )

        if not data:
            raise ValueError("Loaded data is None or empty.")

        first_data = list(data.values())[0]
        if first_data is None:
            raise ValueError("First item in loaded data is None.")

        close = dl.load_data(
            start=self.get_time_n_minutes_before(self.start, 1),
            end=self.end,
            symbols=self.symbols,
            features=["close"],
        )

        assert close is not None
        assert close["close"] is not None

        close_to_close = close["close"].diff(1).dropna().reset_index()
        universe_size = len(self.strategy.setup.universe)
        pnl_history = []

        # Initialize with zeros for PnL tracking
        init_weights = np.zeros(universe_size)

        for i in range(len(first_data) - self.lookback):
            # Simulate strategy evaluation and trading
            for feature_name in self.features:
                feature_data = data.get(feature_name)
                if feature_data is None:
                    raise ValueError(
                        f"Feature {feature_name} is missing in the loaded data."
                    )

                # Dynamically set each feature as an attribute of `self.strategy.features`
                setattr(
                    self.strategy.features,
                    feature_name,
                    feature_data.iloc[i : (i + self.lookback)][self.symbols].to_numpy(),
                )

            curr_weights = self.strategy.eval()
            period_pnl: float = np.sum(
                (curr_weights - init_weights) * close_to_close.iloc[i][self.symbols]
            )
            init_weights = curr_weights
            pnl_history.append(period_pnl)

        self.pnl_history = np.array(pnl_history)

        # Calculate additional metrics
        self.cumulative_pnl: float = np.sum(self.pnl_history)
        self.volatility = np.std(self.pnl_history)
        self.sharpe_ratio = (
            (np.mean(self.pnl_history) / self.volatility) if self.volatility != 0 else 0
        )
        drawdown: float = np.maximum.accumulate(
            np.cumsum(self.pnl_history)
        ) - np.cumsum(self.pnl_history)
        self.max_drawdown: float = np.max(drawdown)

        print("Backtest complete.")

    def get_results(self) -> dict[str, Any]:
        """
        Return the PnL history and any other relevant metrics.

        :return: A dictionary containing PnL history, cumulative PnL, Sharpe Ratio, Volatility, and Max Drawdown.
        """
        return {
            "pnl_history": self.pnl_history,
            "cumulative_pnl": self.cumulative_pnl,
            "Sharpe Ratio": self.sharpe_ratio,
            "Volatility": self.volatility,
            "Max Drawdown": self.max_drawdown,
        }

    def plot_pnl(self) -> None:
        """
        Plot the cumulative PnL over time.

        :return: None
        """
        cumulative_pnl = np.cumsum(self.pnl_history)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_pnl, label="Cumulative PnL")
        plt.title("Cumulative PnL Over Time")
        plt.xlabel("Time Periods")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Load the YAML configuration and the new strategy
    strategy_module_path = "tests/test_data/strategy/strat1.py"
    config_path = "tests/test_data/strategy/strat1.yaml"
    # config = yaml.safe_load(Path(config_path).read_text())

    # Dynamically load the strategy class
    backtest = Backtester(strategy_module_path, config_path)
    path = "./src/simple_backtester/data/feature/"
    config = {
        "data_path": path,
        # "tech_indicators": ["ma", "macd", "rsi"],
        "features": [file.name[:-4] for file in Path(path).iterdir() if file.is_file()],
    }
    backtest.run(config)

    # Get and display results
    results = backtest.get_results()
    print("Cumulative PnL:", results["cumulative_pnl"])
    print("Sharpe Ratio:", results["Sharpe Ratio"])
    print("Volatility:", results["Volatility"])
    print("Max Drawdown:", results["Max Drawdown"])
    print("PnL History:", results["pnl_history"][:10])

    # Plot the PnL
    backtest.plot_pnl()
