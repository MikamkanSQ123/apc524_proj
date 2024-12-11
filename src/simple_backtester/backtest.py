import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import Strategy
from .preprocess import Dataloader
from typing import Any, Union
from datetime import datetime, timedelta
from tests.test_data.strategy.strat1 import MeanReversion

class Backtest:
    def __init__(self, strategy_class: type[Strategy], config_path: Union[str, Path]):
        self.strategy = strategy_class(config_path)

        self.start = self.strategy.setup.start_date
        self.end = self.strategy.setup.end_date
        self.lookback = self.strategy.setup.look_back
        self.symbols = self.strategy.setup.universe
        self.features = self.strategy.setup.features

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
        dl = Dataloader(config)
        data = dl.load_data(start=start, end=self.end, symbols=self.symbols, features=self.features)
        universe_size = len(self.strategy.setup.universe)
        pnl_history = []

        # Initialize with zeros for PnL tracking
        weights = np.zeros(universe_size)

        for i in range(self.strategy.setup.warm_up, price_data.shape[0]):
            # Simulate strategy evaluation and trading
            weights = self.strategy.eval()
            
            # Calculate PnL for this period (simplified example)
            period_pnl = np.sum(weights * (price_data[i] - price_data[i - 1]))
            pnl_history.append(period_pnl)

        self.pnl_history = np.array(pnl_history)
        
        # Calculate additional metrics
        self.cumulative_pnl = np.sum(self.pnl_history)
        self.volatility = np.std(self.pnl_history)
        self.sharpe_ratio = (np.mean(self.pnl_history) / self.volatility) if self.volatility != 0 else 0
        drawdown = np.maximum.accumulate(np.cumsum(self.pnl_history)) - np.cumsum(self.pnl_history)
        self.max_drawdown = np.max(drawdown)

        print("Backtest complete.")

    def get_results(self) -> dict:
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
    # Load the YAML configuration
    config_path = "tests/test_data/strategy/strat1.yaml"
    config = yaml.safe_load(Path(config_path).read_text())

    # Simulated price data (replace with actual market data)
    # Rows are time periods, columns are securities in the universe
    np.random.seed(42)
    price_data = np.cumsum(np.random.randn(500, len(config["setup"]["universe"])), axis=0)

    # Initialize and run the backtest
    backtest = Backtest(MeanReversion, config_path)
    backtest.run(price_data)

    # Get and display results
    results = backtest.get_results()
    print("Cumulative PnL:", results["cumulative_pnl"])
    print("Sharpe Ratio:", results["Sharpe Ratio"])
    print("Volatility:", results["Volatility"])
    print("Max Drawdown:", results["Max Drawdown"])
    print("PnL History:", results["pnl_history"][:10])

    # Plot the PnL
    backtest.plot_pnl()
