import numpy as np


def sphere(values: np.ndarray):
    return -np.sum(np.power(values, 2))


def sin(values: np.ndarray):
    cost = (
        10
        * -np.cos(2 * (values[0] ** 2 + values[1] ** 2) ** 0.5)
        * np.exp(-0.5 * ((values[0] + 1) ** 2 + (values[1] - 1) ** 2) ** 0.5)
    )
    return cost
