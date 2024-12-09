import numpy as np
from simple_backtester.performance.sharpe import sharpe_ratio
from pytest import approx


def test_sharpe_ratio():
    returns = [0.1, 0.12, 0.08, 0.09, 0.11]  # Example returns
    risk_free_rate = 0.02

    calculated_sharpe = sharpe_ratio(returns, risk_free_rate)

    # Example test case (the expected value depends on the sample returns provided)
    expected_sharpe = (np.mean([0.1, 0.12, 0.08, 0.09, 0.11]) - 0.02) / np.std(
        [0.1, 0.12, 0.08, 0.09, 0.11]
    )

    assert calculated_sharpe == approx(expected_sharpe)
