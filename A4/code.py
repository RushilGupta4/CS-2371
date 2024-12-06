import math
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def option_price_difference(sigma, S, K, T, r, market_price):
    return black_scholes_call(S, K, T, r, sigma) - market_price


S = 15
K = 13
T = 0.25
r = 0.05
market_price = 2.5

# Initial bounds for volatility (between 0.0001 and 5.0 to ensure convergence)
sigma_lower = 0.0001
sigma_upper = 5.0

try:
    # Find the implied volatility using Brent's method
    implied_vol = brentq(
        option_price_difference,
        sigma_lower,
        sigma_upper,
        args=(S, K, T, r, market_price),
        xtol=1e-10,  # Tolerance for convergence
    )
    print(f"The implied volatility is {implied_vol:.4%}")
except ValueError:
    print("Implied volatility not found within the given bounds.")
