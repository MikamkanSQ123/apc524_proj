import math

def discount(r, t):
    return math.exp(-r * t)

if __name__ == "__main__":
    assert(abs(discount(0.05, 1) - 0.9512294) < 1e-4)
