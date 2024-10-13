import math

def discount(r, t, freq=None):
    """
    Calculate the discount factor for a given interest rate and time period.
    
    Parameters:
        r (float): The interest rate.
        t (float): The time period.
        freq (int, optional): The compounding frequency. If None, continuous compounding is assumed.

    Returns:
        float: The discount factor.
    """

    if freq is None:
        return math.exp(-r * t)
    else:
        return 1 / (1 + r / freq) ** (freq * t)

if __name__ == "__main__":
    assert(abs(discount(0.05, 1) - 0.9512294) < 1e-4)
    assert(abs(discount(0.05, 1, 2) - 0.9518144 < 1e-4))
