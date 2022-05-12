from typing import Callable
import numpy as np
from data_type.search_agent import SearchAgent


def get_neighbours(A, step_size, lower_bound, upper_bound, cost_function: Callable):
    n_variables = len(A.position)
    neighbours = []
    for i in range(n_variables * 2):

        if i == 0:
            step = np.array([step_size[0], 0])
        elif i == 1:
            step = np.array([0, step_size[1]])
        elif i == 2:
            step = np.array([-step_size[0], 0])
        elif i == 3:
            step = np.array([0, -step_size[1]])

        position = []
        for value, LB, UB in zip(A.position + step, lower_bound, upper_bound):
            if value < LB:
                value = LB
            elif value > UB:
                value = UB
            position.append(value)
        position = np.array(position)

        cost = cost_function(position)
        neighbour = SearchAgent(position=position, cost=cost)
        neighbours.append(neighbour)
    return neighbours
