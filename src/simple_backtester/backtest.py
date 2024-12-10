import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Strategy
from typing import Dict, Any


class Backtester:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backtester with the configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary with settings for the backtest.
        """
        self.config: Dict[str, Any] = config
        self.strategy: Strategy = Strategy(config)

    def load_data(self) -> pd.DataFrame:
        """
        Load market data from a CSV file specified in the configuration.

        Returns:
            pd.DataFrame: Loaded market data.
        """
        file_path: str = self.config["data"]["file_path"]
        try:
            data: pd.DataFrame = pd.read_csv(
                file_path, parse_dates=["Date"], index_col="Date"
            )
            if "Close" not in data.columns:
                raise ValueError("Input data must contain a 'Close' column.")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {file_path}.")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics like Sharpe ratio, volatility, and max drawdown.

        Args:
            data (pd.DataFrame): Data containing strategy returns.

        Returns:
            Dict[str, float]: Performance metrics.
        """
        strategy_returns: pd.Series[float] = data["Strategy_Return"].dropna()
        risk_free_rate: float = self.config["metrics"]["risk_free_rate"]

        # Sharpe Ratio
        excess_returns: pd.Series[float] = strategy_returns - risk_free_rate / 252
        sharpe_ratio: float = (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        )

        # Volatility
        volatility: float = strategy_returns.std() * np.sqrt(252)

        # Max Drawdown
        cumulative_returns: pd.Series[float] = (1 + strategy_returns).cumprod()
        rolling_max: pd.Series[float] = cumulative_returns.cummax()
        drawdown: pd.Series[float] = cumulative_returns / rolling_max - 1
        max_drawdown: float = drawdown.min()

        metrics: Dict[str, float] = {
            "Sharpe Ratio": sharpe_ratio,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
        }
        return metrics

    def run(self) -> pd.DataFrame:
        """
        Run the backtest and compute strategy performance.

        Returns:
            pd.DataFrame: Backtest results including cumulative returns and signals.
        """
        data: pd.DataFrame = self.load_data()
        data = self.strategy.generate_signals(data)

        # Calculate returns
        data["Daily_Return"] = data["Close"].pct_change()
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Daily_Return"]

        # Calculate cumulative returns
        data["Cumulative_Market_Return"] = (1 + data["Daily_Return"]).cumprod()
        data["Cumulative_Strategy_Return"] = (1 + data["Strategy_Return"]).cumprod()

        # Calculate and display performance metrics
        metrics: Dict[str, float] = self.calculate_performance_metrics(data)
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        self.plot_results(data)
        return data

    def plot_results(self, data: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot the cumulative returns of the strategy and the market.

        Args:
            data (pd.DataFrame): Backtest results including cumulative returns.
            save_path (str, optional): If provided, save the plot to this path. Defaults to None.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data["Cumulative_Market_Return"], label="Market Return")
        plt.plot(data["Cumulative_Strategy_Return"], label="Strategy Return")
        plt.legend()
        plt.title("Backtest Results")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}.")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    # Load configuration from YAML
    with open("config.yaml", "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    # Run the backtest
    backtester = Backtester(config)
    try:
        results: pd.DataFrame = backtester.run()

        # Optionally save results to a file
        output_path: str = config["output"]["file_path"]
        results.to_csv(output_path)
        print(f"Results saved to {output_path}.")

        # Save the plot if specified in config
        if "plot_file_path" in config["output"]:
            backtester.plot_results(results, save_path=config["output"]["plot_file_path"])
    except Exception as e:
        print(f"An error occurred during the backtest: {e}")
