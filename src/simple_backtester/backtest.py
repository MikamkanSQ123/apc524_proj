from pathlib import Path
import importlib.util
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import Strategy
from .preprocess.dataloader import DataLoader
from typing import Any, Union, Optional, Type, List
from numpy.typing import NDArray
from .data_protocol import Numeric
from numpy import floating
from datetime import datetime, timedelta
from tqdm import tqdm


class Backtester:
    def __init__(
        self,
        strategy_module_path: str,
        config_path: Union[str, Path],
    ) -> None:
        """
        Initialize the backtester.

        Args:
            strategy_module_path (str): Path to the strategy module file.
            config_path (Union[str, Path]): Path to the configuration file.
        """
        # Dynamically load the strategy class
        self.strategy_class = self.load_strategy_class(strategy_module_path)

        # Ensure the loaded class is a concrete subclass of Strategy
        if inspect.isabstract(self.strategy_class):
            raise ValueError(
                f"The loaded strategy class must not be abstract: {self.strategy_class}."
            )

        # Instantiate the strategy
        self.strategy_list = self.strategy_class.from_yaml(config_path)

        self.pnl_history: List[NDArray[floating[Any]]] = []
        self.cumulative_pnl: List[Numeric] = []
        self.sharpe_ratio: List[Numeric] = []
        self.volatility: List[Numeric] = []
        self.max_drawdown: List[Numeric] = []

    @staticmethod
    def load_strategy_class(module_path: str) -> Type[Strategy]:
        """
        Dynamically load a strategy class from a given module.

        Args:
            module_path (str): Path to the Python module file.
            base_class (type): The abstract base class (Strategy).

        Returns:
            Type[Strategy]: The dynamically loaded concrete subclass of Strategy.

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
            if issubclass(cls, Strategy) and cls is not Strategy
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

        # Explicitly cast to Type[Strategy] to satisfy the type checker
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

    def run(self, config: dict[str, Any], nthread: int = 1) -> None:
        """
        Executes a batch run of backtests for each strategy in the strategy list.

        Args:
            config (dict[str, Any]): A dictionary containing configuration parameters for the backtest.
        Returns:
            None
        """

        if nthread > 1:
            raise ValueError("Multithreading is not supported yet.")
        self.clear_results()
        for strategy in self.strategy_list:
            self.strategy = strategy
            self.start = self.strategy.setup.start_date
            self.end = self.strategy.setup.end_date
            self.lookback = self.strategy.setup.look_back
            self.symbols = self.strategy.setup.universe
            self.features = self.strategy.setup.features
            self._run(config)

        print("Backtest complete.")

    def _run(self, config: dict[str, Any]) -> None:
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

        close_to_close = close["close"].diff(1).fillna(0).reset_index()
        universe_size = len(self.strategy.setup.universe)
        pnl_history = []

        # Initialize with zeros for PnL tracking
        init_weights = np.zeros(universe_size)

        for i in tqdm(range(len(first_data) - self.lookback), desc="Backtesting Progress"):
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
            transaction_cost = self.strategy.setup.rate_transaction_cost * np.abs(curr_weights - init_weights).sum()
            period_pnl: float = np.sum(
                (curr_weights - init_weights) * close_to_close.iloc[i][self.symbols]
            )
            period_pnl -= transaction_cost  # Subtract transaction cost from PnL
            self.strategy._Strategy__pnl.append(period_pnl)  # type: ignore[attr-defined]
            init_weights = curr_weights
            pnl_history.append(period_pnl)

        self.pnl_history.append(np.array(pnl_history))

        # Calculate additional metrics
        self.cumulative_pnl.append(np.sum(pnl_history))
        volatility = np.std(pnl_history)
        self.volatility.append(volatility)
        self.sharpe_ratio.append(
            (np.mean(pnl_history) / volatility) if volatility != 0 else 0.0
        )
        drawdown: float = np.maximum.accumulate(np.cumsum(pnl_history)) - np.cumsum(
            pnl_history
        )
        self.max_drawdown.append(np.max(drawdown))

    def get_results(self) -> Union[dict[str, Any], List[dict[str, Any]]]:
        """
        Return the PnL history and any other relevant metrics.

        :return: A dictionary containing PnL history, cumulative PnL, Sharpe Ratio, Volatility, and Max Drawdown.
        """
        results = []
        for res in zip(
            self.pnl_history,
            self.cumulative_pnl,
            self.sharpe_ratio,
            self.volatility,
            self.max_drawdown,
        ):
            results.append(
                dict(
                    zip(
                        [
                            "pnl_history",
                            "cumulative_pnl",
                            "Sharpe Ratio",
                            "Volatility",
                            "Max Drawdown",
                        ],
                        res,
                    )
                )
            )

        if len(results) == 1:
            return results[0]
        return results

    def plot_pnl(self) -> None:
        """
        Plot the cumulative PnL over time.

        :return: None
        """
        fig, axes = plt.subplots(
            len(self.pnl_history),
            1,
            figsize=(8, 6 * len(self.pnl_history)),
            sharex=True,
            sharey=True,
        )
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        for i, ax in enumerate(axes):
            cumulative_pnl = np.cumsum(self.pnl_history[i])
            ax.plot(cumulative_pnl)
            ax.set_xlabel("Time Periods")
            ax.set_ylabel("Cumulative PnL")
            ax.grid()
        plt.title("Cumulative PnL Over Time")
        plt.show()

    def clear_results(self) -> None:
        """
        Clear the results of the backtest.

        :return: None
        """
        self.pnl_history = []
        self.cumulative_pnl = []
        self.sharpe_ratio = []
        self.volatility = []
        self.max_drawdown = []


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
    results: dict[str, Any] = backtest.get_results()  # type: ignore[assignment]
    print("Cumulative PnL:", results["cumulative_pnl"])
    print("Sharpe Ratio:", results["Sharpe Ratio"])
    print("Volatility:", results["Volatility"])
    print("Max Drawdown:", results["Max Drawdown"])
    print("PnL History:", results["pnl_history"][:10])

    # Plot the PnL
    backtest.plot_pnl()
