def q1():
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum

    # Define the problem
    problem = LpProblem("Maximize_Chocolatier_Profit", LpMaximize)

    # Define the decision variables
    x = LpVariable(
        "Super_Dark_Bars", lowBound=0, cat="Continuous"
    )  # Number of Super Dark bars
    y = LpVariable(
        "Special_Dark_Bars", lowBound=0, cat="Continuous"
    )  # Number of Special Dark bars

    # Objective function: Maximize profit
    problem += 1 * x + 2 * y, "Profit"

    # Constraints
    problem += 90 * x + 80 * y <= 1260, "Chocolate_Limit"
    problem += 10 * x + 20 * y <= 240, "Sugar_Limit"

    # Solve the problem
    problem.solve()

    # Fetch the results
    super_dark_bars = x.value()
    special_dark_bars = y.value()
    max_profit = problem.objective.value()

    print(f"Number of Super Dark bars: {super_dark_bars}")
    print(f"Number of Special Dark bars: {special_dark_bars}")
    print(f"Maximum Profit: Rs.{max_profit}")


def q2():
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value

    # Create the linear programming problem
    prob = LpProblem("Project Selection", LpMaximize)

    # Define decision variables
    projects = range(1, 6)
    x = LpVariable.dicts("project", projects, cat="Binary")

    # Define the objective function
    returns = {1: 20, 2: 40, 3: 20, 4: 15, 5: 30}
    prob += lpSum([returns[i] * x[i] for i in projects])

    # Define the constraints
    expenditures = {
        1: {1: 5, 2: 4, 3: 3, 4: 7, 5: 8},
        2: {1: 1, 2: 7, 3: 9, 4: 4, 5: 6},
        3: {1: 8, 2: 10, 3: 2, 4: 1, 5: 10},
    }

    for year in range(1, 4):
        prob += lpSum([expenditures[year][i] * x[i] for i in projects]) <= 25

    # Solve the problem
    prob.solve()

    # Print the results
    print("Status:", LpStatus[prob.status])
    print("\nSelected projects:")
    for i in projects:
        if x[i].value() == 1:
            print(f"Project {i}")

    print("\nTotal return:", value(prob.objective))

    print("\nYearly expenditures:")
    for year in range(1, 4):
        expenditure = sum(expenditures[year][i] * x[i].value() for i in projects)
        print(f"Year {year}: {expenditure}")


def q6():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read the Excel file
    df = pd.read_excel("Assign_2_RVs.xlsx")  # Adjust the filename if needed
    df.rename(columns={"Normal Random Variables": "Returns"}, inplace=True)

    # a) Measure the mean return and risk (standard deviation) of the portfolio
    mean_return = df["Returns"].mean()
    risk = df["Returns"].std()

    print(f"a) Mean return: {mean_return:.4f}")
    print(f"   Risk (standard deviation): {risk:.4f}")

    # b) Measure the percentage of data within specified bounds
    within_1_sigma = (
        np.sum((df["Returns"] >= -risk) & (df["Returns"] <= risk)) / len(df) * 100
    )
    within_2_sigma = (
        np.sum((df["Returns"] >= -2 * risk) & (df["Returns"] <= 2 * risk))
        / len(df)
        * 100
    )
    within_3_sigma = (
        np.sum((df["Returns"] >= -3 * risk) & (df["Returns"] <= 3 * risk))
        / len(df)
        * 100
    )

    print(f"\nb) Percentage of data:")
    print(f"   Within 1 sigma: {within_1_sigma:.2f}%")
    print(f"   Within 2 sigma: {within_2_sigma:.2f}%")
    print(f"   Within 3 sigma: {within_3_sigma:.2f}%")

    # c) Plot the histogram of returns
    plt.figure(figsize=(10, 6))
    plt.hist(df["Returns"], bins=50, edgecolor="black")
    plt.title("Histogram of Hedge Fund Returns")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.savefig("q6.png")
    plt.close()


def q7():
    import matplotlib.pyplot as plt
    import numpy as np

    # Define the risk-free rate and risky assets
    risk_free_rate = 0.03

    def e(wa):
        return wa * 0.2 + (1 - wa) * 0.15

    def sigma(wa):
        return np.sqrt(wa**2 * 0.5**2 + (1 - wa) ** 2 * 0.33**2)

    w_a = np.linspace(0, 1)
    e_p = [e(wa) for wa in w_a]
    sigma_p = [sigma(wa) for wa in w_a]

    print(w_a)
    print(e_p)
    print(sigma_p)

    plt.figure(figsize=(10, 6))
    plt.plot(sigma_p, e_p, label="Portfolio")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Expected Return")
    plt.title("Portfolio Expected Return vs Standard Deviation")
    plt.legend()
    plt.savefig("q7.png")
    plt.close()


def q7_d():
    import numpy as np
    from scipy.optimize import root_scalar

    def e(wa):
        return wa * 0.2 + (1 - wa) * 0.15

    def sigma(wa):
        return np.sqrt(wa**2 * 0.5**2 + (1 - wa) ** 2 * 0.33**2)

    def m(wa):
        return (e(wa) - 0.03) / sigma(wa)

    def m_prime(wa):
        # Analytical derivative of m(wa)
        numerator = 0.05 * sigma(wa) - (e(wa) - 0.03) * (
            wa * 0.5**2 - (1 - wa) * 0.33**2
        ) / sigma(wa)
        denominator = sigma(wa) ** 2
        return numerator / denominator

    # Find the root of m_prime(wa) = 0
    result = root_scalar(m_prime, bracket=[0, 1], method="brentq")
    optimal_wa = result.root

    print(f"Optimal weight for asset A: {optimal_wa:.4f}")
    print(f"Optimal weight for asset B: {1 - optimal_wa:.4f}")
    print(f"Maximum Sharpe ratio: {m(optimal_wa):.4f}")
    print(f"Expected return: {e(optimal_wa):.4f}")
    print(f"Risk: {sigma(optimal_wa):.4f}")


def q7_e():
    import numpy as np
    import pandas as pd

    def r(wa):
        return wa * 0.1691 + (1 - wa) * 0.03

    def sigma(wa):
        return wa * 0.2794

    def u(r, sigma, A):
        return r - 1 / 2 * A * sigma**2

    A_0 = 2.5
    r_vals = [0.03, 0.09, 0.15, 0.20]

    ws = []
    i = 0
    dw = 0.000001
    for w in range(0, int(2 / dw)):
        w_scaled = w * dw
        _r = r(w_scaled)
        # print(w, _r, abs(r_vals[i] - _r) < 0.01)
        if abs(r_vals[i] - _r) < 0.0000001:
            ws.append(w_scaled)
            i += 1

        if i == len(r_vals):
            break

    print(ws)
    res = []
    for w in ws:
        res.append(
            {
                "Weight A": w,
                "Return": r(w),
                "Risk": sigma(w),
                "Utility": u(r(w), sigma(w), A_0),
            }
        )

    res = pd.DataFrame(res)

    print(res)
    # print(res.to_latex(index=False))


q7_e()
