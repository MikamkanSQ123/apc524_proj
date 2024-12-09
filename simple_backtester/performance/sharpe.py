import numpy as np


def sharpe_ratio(returns, risk_free_rate):
    """
    Calculate the Sharpe Ratio for a given set of returns and a risk-free rate.

    Parameters:
        returns (list or np.array): A list or array of portfolio returns.
        risk_free_rate (float): The risk-free rate.

    Returns:
        float: The Sharpe Ratio.
    """

    excess_returns = np.array(returns) - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)

    if std_dev_excess_return == 0:
        return np.nan  # Avoid division by zero

    return mean_excess_return / std_dev_excess_return


if __name__ == "__main__":
    returns = [0.1, 0.12, 0.08, 0.09, 0.11]  # Example returns
    risk_free_rate = 0.02

    calculated_sharpe = sharpe_ratio(returns, risk_free_rate)

    # Example test case (the expected value depends on the sample returns provided)
    expected_sharpe = (np.mean([0.1, 0.12, 0.08, 0.09, 0.11]) - 0.02) / np.std(
        [0.1, 0.12, 0.08, 0.09, 0.11]
    )

    assert abs(calculated_sharpe - expected_sharpe) < 1e-6
    print("Sharpe Ratio calculated successfully:", calculated_sharpe)
