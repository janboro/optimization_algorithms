import numpy as np


def constrained_sphere(x: np.ndarray):
    constraint_1 = x[1] > 3.2 and x[1] < 6.4
    constraint_2 = (x[0] ** 2 + x[1] ** 2) < 14.0
    constraint_3 = x[0] != x[1]

    if constraint_1 and constraint_2 and constraint_3:
        # Solution is feasible
        return np.sum(np.power(x, 2))
    else:
        # Solution is NOT feasible
        return np.sum(np.power(x, 2)) + 10000


def sphere(values: np.ndarray):
    return np.sum(np.power(values, 2))
