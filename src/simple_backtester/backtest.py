import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from .config import Strategy
from typing import Dict, Any


class Backtester:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the backtester with the configuration."""
        self.config = config
        #self.strategy = Strategy(config)

    def load_data(self) -> pd.DataFrame:
        """Load market data from a CSV file specified in the configuration."""
        file_path = self.config["data"]["file_path"]
        data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        return data

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics like Sharpe ratio and volatility."""
        strategy_returns = data["Strategy_Return"].dropna()
        risk_free_rate = self.config["metrics"]["risk_free_rate"]

        # Sharpe Ratio
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Volatility
        volatility = strategy_returns.std() * np.sqrt(252)

        # Max Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1
        max_drawdown = drawdown.min()

        metrics = {
            "Sharpe Ratio": sharpe_ratio,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
        }
        return metrics

    def run(self) -> pd.DataFrame:
        """Run the backtest."""
        data = self.load_data()
        #data = self.strategy.generate_signals(data)

        # Calculate returns
        data["Daily_Return"] = data["Close"].pct_change()
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Daily_Return"]

        # Calculate cumulative returns
        data["Cumulative_Market_Return"] = (1 + data["Daily_Return"]).cumprod()
        data["Cumulative_Strategy_Return"] = (1 + data["Strategy_Return"]).cumprod()

        # Calculate and display performance metrics
        metrics = self.calculate_performance_metrics(data)
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        self.plot_results(data)
        return data

    def plot_results(self, data: pd.DataFrame) -> None:
        """Plot the cumulative returns of the strategy and the market."""
        plt.figure(figsize=(12, 6))
        plt.plot(data["Cumulative_Market_Return"], label="Market Return")
        plt.plot(data["Cumulative_Strategy_Return"], label="Strategy Return")
        plt.legend()
        plt.title("Backtest Results")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Load configuration from YAML
    with open("config.yaml", "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    # Run the backtest
    backtester = Backtester(config)
    results = backtester.run()

    # Optionally save results to a file
    output_path = config["output"]["file_path"]
    results.to_csv(output_path)
