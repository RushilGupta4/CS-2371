import numpy as np
import scipy
import scipy.integrate
from scipy.optimize import fsolve


s_arr = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
C_arr = [50000, 50000, 50000, 50000, 50000, 1050000]

P = 0
for i in range(6):
    d = 1 / ((1 + s_arr[i]) ** (i + 1))
    P += d * C_arr[i]

# print(P)

l = 0.063491

P_l = 0
for i in range(6):
    d = 1 / ((1 + l) ** (i + 1))
    P_l += d * C_arr[i]

# print(l, P - P_l)

# print()


C_arr = [50000, 50000, 55000, 60000, 65000, 1070000]

P2 = 0
for i in range(6):
    d = 1 / ((1 + s_arr[i]) ** (i + 1))
    P2 += d * C_arr[i]

# print(P2)

l = 0.063391

P_l = 0
for i in range(6):
    d = 1 / ((1 + l) ** (i + 1))
    P_l += d * C_arr[i]

# print(l, P2 - P_l)

a = -5


def sigma_p(a):
    return np.sqrt(
        (a**2) * (0.05**2)
        + 2 * (-1) * (a) * (1 - a) * (0.05) * (0.07)
        + ((1 - a) ** 2) * (0.07**2)
    )


# print(0.8 - (0.2 * a + (1 - a) * 0.3))


a_c = 0.58333
r_c = a_c * 0.2 + (1 - a_c) * 0.3
# print(r_c)

r_c = 0.241667

d = -42.19966
# print(d, 2 - (d * 0.2 + (1 - d) * r_c))

c_a = 1 - d

# print(c_a * a_c, c_a * (1 - a_c))


purse = 100
# Buy d * purse amounts of r_f
purse -= d * 100
# print(purse)

# Buy (1 - d) * purse of portfolio C
purse -= (1 - d) * 100
# print(purse)


def P_bond(flows, d=0.1):
    P = 0
    for i, f in enumerate(flows):
        P += f / ((1 + d) ** (i + 1))

    return P


def D_bond(flows, d=0.1):
    D = 0
    P = P_bond(flows, d)
    for i, f in enumerate(flows):
        D += (f * (i + 1)) / P

    return D * 1 / (1 + d)


def C_bond(flows, d=0.1):
    C = 0
    P = P_bond(flows, d)
    for i, f in enumerate(flows):
        cur = (i + 1) * (i + 2) * (f / ((1 + d) ** (i + 1)))
        cur = cur / P

        C += cur

    return C / ((1 + d) ** 2)


A = [12, 12, 12, 112]
B = [108]
C = [5, 105]
D = [0, 0, 150]

P_a = P_bond(A)
D_a = D_bond(A)
C_a = C_bond(A)
print(f"{P_a=} | {D_a=} | {C_a=}")
P_b = P_bond(B)
D_b = D_bond(B)
C_b = C_bond(B)
print(f"{P_b=} | {D_b=} | {C_b=}")
P_c = P_bond(C)
D_c = D_bond(C)
C_c = C_bond(C)
print(f"{P_c=} | {D_c=} | {C_c=}")
P_o = P_bond(D)
D_o = D_bond(D)
C_o = C_bond(D)
print(f"{P_o=} | {D_o=} | {C_o=}")


e = 0.25


def equations(vars):
    i, j = vars
    e1 = 3.63 - (i * 4.445 + j + (1 - i - j) * 2.14)
    e2 = 9.917 - (i * 13.36 + j * 1.653 + (1 - i - j) * 4.794)
    return [e1, e2]


initial_guess = [0.5, 0.5]
solution = fsolve(equations, initial_guess)

i, j = solution
print(f"Solution: i = {i:.6f}, j = {j:.6f}, k = {1 - j - i:.6f}")

print(i * 150, j * 150, (1 - j - i) * 150)

# Verify the solution
e1, e2 = equations(solution)
# print(f"Verification: e1 = {e1:.6e}, e2 = {e2:.6e}")

A = [12, 12, 12, 112]
B = [108]
C = [5, 105]
D = [0, 0, 150]
P_o = P_bond(D, 0.10)
P_a = P_bond(A, 0.10)
P_b = P_bond(B, 0.10)
P_c = P_bond(C, 0.10)


print(P_o - (i * P_a + j * P_b + (1 - i - j) * P_c))
