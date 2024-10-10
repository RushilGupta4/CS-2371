import numpy as np
import matplotlib.pyplot as plt


def q1():
    # Given parameters
    P0 = 100  # Initial principal amount (Rs. 100)
    t = 2  # Time in years
    r = 8 / 100  # Annual interest rate (8%)

    # Discretize the interval of n into small steps
    n_values = np.linspace(1, 10000, 100)  # n ∈ [0, 10000], with 1000 steps

    # Calculate compound interest for each n
    A_values = P0 * (1 + r / n_values) ** (n_values * t)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, A_values, color="blue", linewidth=1)
    plt.title("Compound Interest as a Function of Compounding Frequency n")
    plt.xlabel("Number of Compounding Periods (n)")
    plt.ylabel("Amount (A) after 2 years")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("Q1.png")


def q7():
    times = [
        0.00,
        0.25,
        0.50,
        0.75,
        0.99,
        1.00,
        1.25,
        1.50,
        1.75,
        1.99,
        2.00,
        2.25,
        2.50,
        2.75,
        2.99,
        3.00,
        3.25,
        3.50,
        3.75,
        3.99,
        4.00,
    ]
    npvs = [
        1050.00,
        1060.12,
        1072.11,
        1084.23,
        1096.49,
        1036.49,
        1048.89,
        1060.75,
        1072.74,
        1084.87,
        1024.87,
        1037.13,
        1048.86,
        1060.72,
        1072.71,
        1012.71,
        1024.84,
        1036.43,
        1048.15,
        1060.00,
        1000.00,
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(times, npvs, marker="o")
    plt.title("NPV of Bond Over Time")
    plt.xlabel("Time (Years)")
    plt.ylabel("NPV ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Q7.png")


if __name__ == "__main__":
    q7()
