import numpy as np
from typing import List, Union, Any
from numpy.typing import NDArray
from ..data_protocol import Numeric


def sharpe_ratio(
    returns: Union[List[Numeric], NDArray[np.floating[Any]]],
    risk_free_rate: Numeric,
) -> Numeric:
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
