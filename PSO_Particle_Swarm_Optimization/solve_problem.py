import matplotlib.pyplot as plt

from data_type.problem import Problem, Bounds
from data_type.PSO_params import PSOParams
from PSO import PSO
from cost_functions import sphere

pso_params = PSOParams(iterations=50, swarm_size=50)
bounds = Bounds(upper=10, lower=-10)
problem = Problem(cost_function=sphere, variables=5, bounds=bounds)
pso = PSO(problem=problem, pso_params=pso_params)
best_result = pso.run()

plt.figure()
plt.plot(pso.best_costs)

plt.figure()
plt.semilogy(pso.best_costs)

plt.show()
