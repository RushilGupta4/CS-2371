# Consider a call option priced at Rs. 5 with a strike price of Rs. 100. Draw the payoff diagram for the call option. On the maturity date, if the stock price is Rs. 102, would it be wise to exercise the option?

# b) A stock is currently priced at Rs. 40. A 1-year European put option with a strike price of Rs. 30 is priced at Rs. 7, while a 1-year European call option with a strike price of Rs. 50 is priced at Rs. 5. Draw the payoff diagram for a portfolio consisting of a long position in 100 shares, a short position in 100 call options, and a long position in 100 put options.

import numpy as np
import matplotlib.pyplot as plt


def q1a():
    K = 100  # Strike price
    Premium = 5  # Option price
    S_T = np.linspace(80, 120, 1000)  # Stock price at expiration

    profit = np.maximum(0, S_T - K) - Premium

    plt.figure()
    plt.plot(S_T, profit, label="Call Option Profit")
    plt.xlabel("Stock Price at Expiration (Rs.)")
    plt.ylabel("Profit (Rs.)")
    plt.title("Profit Diagram for Call Option")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(K, color="grey", linestyle="--", linewidth=0.5)
    plt.savefig("q1a.png")
    plt.close()


def q1b():
    S_T = np.linspace(0, 80, 1000)  # Stock price at expiration
    S_0 = 40  # Initial stock price

    # Positions
    n_shares = 100  # Long 100 shares
    n_calls = 100  # Long 100 call options
    n_puts = 100  # Long 100 put options

    # Call option
    K_c = 50  # Call strike price
    Premium_c = 5  # Call premium

    # Put option
    K_p = 30  # Put strike price
    Premium_p = 7  # Put premium

    # Profit from shares
    profit_shares = n_shares * (S_T - S_0)

    # Profit from call options (long position)
    profit_calls = n_calls * (Premium_c - np.maximum(0, S_T - K_c))

    # Profit from put options
    profit_puts = n_puts * (-Premium_p + np.maximum(0, K_p - S_T))

    # Total profit
    total_profit = profit_shares + profit_calls + profit_puts

    plt.figure()
    plt.plot(S_T, total_profit, label="Portfolio Profit")
    plt.xlabel("Stock Price at Expiration (Rs.)")
    plt.ylabel("Profit (Rs.)")
    plt.title("Profit Diagram for Portfolio")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(
        K_p, color="grey", linestyle="--", linewidth=0.5, label="Put Strike Price"
    )
    plt.axvline(
        K_c, color="grey", linestyle="--", linewidth=0.5, label="Call Strike Price"
    )
    plt.savefig("q1b.png")
    plt.close()


def q2b():
    K = 20  # Strike price
    S_T = np.linspace(0, 40, 400)  # Stock price range at maturity

    # Payoff calculations
    payoff_short_stock = -S_T
    payoff_short_put = -np.maximum(K - S_T, 0)
    payoff_long_call = np.maximum(S_T - K, 0)
    total_payoff = payoff_short_stock + payoff_short_put + payoff_long_call

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        S_T, total_payoff, label="Total Portfolio Payoff", color="blue", linewidth=2
    )
    plt.title("Payoff Diagram for Part b)")
    plt.xlabel("Stock Price at Maturity (S_T)")
    plt.ylabel("Payoff (Rs.)")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.savefig("q2b.png")


def q2c():
    K = 20  # Strike price
    S_T = np.linspace(0, 40, 400)  # Stock price range at maturity

    # Payoff calculations
    payoff_long_stock = S_T
    payoff_long_put = np.maximum(K - S_T, 0)
    payoff_short_call = -np.maximum(S_T - K, 0)
    total_payoff = payoff_long_stock + payoff_long_put + payoff_short_call

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        S_T, total_payoff, label="Total Portfolio Payoff", color="green", linewidth=2
    )
    plt.title("Payoff Diagram for Part c)")
    plt.xlabel("Stock Price at Maturity (S_T)")
    plt.ylabel("Payoff (Rs.)")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.savefig("q2c.png")


def q5():
    S0 = 50  # Current stock price

    r = 0.05  # Risk-free interest rate per period (continuous compounding)
    u = 1.2  # Upward movement factor
    d = 0.8  # Downward movement factor
    dt = 1  # Time per period

    # Compute risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Range of strike prices
    strike_prices = np.arange(30, 61)  # From Rs. 30 to Rs. 60 inclusive

    option_prices = []

    for K in strike_prices:
        # Possible stock prices at t=2
        Suu = S0 * u * u
        Sud = S0 * u * d  # Same as Sdu
        Sdd = S0 * d * d

        # Payoffs at t=2
        payoff_Suu = max(Suu - K, 0)
        payoff_Sud = max(Sud - K, 0)
        payoff_Sdd = max(Sdd - K, 0)

        # Probabilities
        prob_Suu = p * p
        prob_Sud = 2 * p * (1 - p)
        prob_Sdd = (1 - p) * (1 - p)

        # Expected payoff at t=2
        expected_payoff = (
            prob_Suu * payoff_Suu + prob_Sud * payoff_Sud + prob_Sdd * payoff_Sdd
        )

        # Present value of expected payoff
        option_price = np.exp(-2 * r * dt) * expected_payoff

        option_prices.append(option_price)

    # Plotting the option price against the strike price
    plt.figure(figsize=(10, 6))
    plt.plot(strike_prices, option_prices, marker="o")
    plt.title("European Call Option Price vs. Strike Price")
    plt.xlabel("Strike Price (Rs.)")
    plt.ylabel("Option Price (Rs.)")
    plt.grid(True)
    plt.savefig("q5.png")


def q7():
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    S0 = 10  # Initial stock price
    Su = 12  # Stock price in up state
    Sm = 10  # Stock price in middle state
    Sd = 8  # Stock price in down state
    K = 9  # Strike price
    r = 0.10  # Risk-free interest rate (continuous compounding)
    T = 1  # Time to maturity in years

    # Discount factor
    discount_factor = np.exp(-r * T)

    # Option payoffs at maturity
    Cu = max(Su - K, 0)
    Cm = max(Sm - K, 0)
    Cd = max(Sd - K, 0)

    # Constraints for super-replication (upper bound)
    def super_replication():
        x_values = np.linspace(0, 1.5, 1000)
        min_C0 = float("inf")
        optimal_x = None
        for x in x_values:
            # Calculate z from constraints
            z1 = Cu - Su * x  # From up state
            z2 = Cm - Sm * x  # From middle state
            z3 = Cd - Sd * x  # From down state
            z = max(z1, z2, z3)  # Super-replication requires z >= max(z1, z2, z3)

            # Calculate initial cost C0
            C0 = S0 * x + z * discount_factor

            # Check if constraints are satisfied in all states
            if (Su * x + z >= Cu) and (Sm * x + z >= Cm) and (Sd * x + z >= Cd):
                if C0 < min_C0:
                    min_C0 = C0
                    optimal_x = x
        return min_C0, optimal_x

    # Constraints for sub-replication (lower bound)
    def sub_replication():
        x_values = np.linspace(-1, 1, 1000)
        max_C0 = float("-inf")
        optimal_x = None
        for x in x_values:
            # Calculate z from constraints
            z1 = Cu - Su * x  # From up state
            z2 = Cm - Sm * x  # From middle state
            z3 = Cd - Sd * x  # From down state
            z = min(z1, z2, z3)  # Sub-replication requires z <= min(z1, z2, z3)

            # Calculate initial cost C0
            C0 = S0 * x + z * discount_factor

            # Check if constraints are satisfied in all states
            if (Su * x + z <= Cu) and (Sm * x + z <= Cm) and (Sd * x + z <= Cd):
                if C0 > max_C0:
                    max_C0 = C0
                    optimal_x = x
        return max_C0, optimal_x

    # Calculate upper and lower bounds
    upper_bound_price, upper_x = super_replication()
    lower_bound_price, lower_x = sub_replication()

    print(f"Upper Bound Price: Rs. {upper_bound_price:.2f}")
    print(f"Optimal x for Upper Bound: {upper_x:.4f}")
    print(f"Lower Bound Price: Rs. {lower_bound_price:.2f}")
    print(f"Optimal x for Lower Bound: {lower_x:.4f}")

    # Final no-arbitrage price range
    print(
        f"\nAll possible no-arbitrage prices of the call option are between Rs. {lower_bound_price:.2f} and Rs. {upper_bound_price:.2f}."
    )


q1a()
q1b()
q2b()
q2c()
q5()
q7()
