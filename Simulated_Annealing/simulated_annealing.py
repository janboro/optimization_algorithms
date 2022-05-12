import numpy as np
from cost_functions import sphere, sin
from data_type.search_agent import SearchAgent
from get_neighbouring_solution import get_neighbours
import matplotlib.pyplot as plt

test_funcion_no = 1

objective_function = sin

initial_position = np.array([0.8, -0.5])
initial_cost = objective_function(initial_position)

upper_bound = np.array([1, 1])
lower_bound = np.array([-1, -1])

step_size = np.array([0.05, 0.05])
n_variables = len(initial_position)

# Simulated annealing params
T = 1
cooling_rate = 0.99
max_iter = 500


best_solution = []
best_position = []
A = SearchAgent(position=initial_position, cost=initial_cost)
best_solution.append(A.cost)
best_position.append(A.position)

for i in range(max_iter):
    neighbours = get_neighbours(
        A=A, step_size=step_size, lower_bound=lower_bound, upper_bound=upper_bound, cost_function=objective_function
    )

    B = np.random.choice(neighbours)
    delta = A.cost - B.cost
    if delta < 0:
        A = B
        best_solution.append(B.cost)
        best_position.append(B.position)
    else:
        p = np.random.rand()
        if p < np.exp(-delta / T):
            A = B
            best_solution.append(B.cost)
            best_position.append(B.position)
    T *= cooling_rate


print(A)

for val in best_position:
    print(val)
plt.plot(best_solution)
plt.show()
