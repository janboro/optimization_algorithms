import matplotlib.pyplot as plt
import numpy as np
from cost_functions import sphere
from data_type.particle import Particle
from data_type.best_solution import BestSolution

# Define details of objective function
variables_number = 10
upper_bound = np.ones(variables_number)
lower_bound = np.zeros(variables_number)
max_velocity = upper_bound - lower_bound * 0.2
min_velocity = -max_velocity
objective_function = sphere


# PSO Parameters
swarm_size = 50
max_iterations = 500
initial_weight = 0.9
final_weight = 0.4
inertia_dampening = (initial_weight - final_weight) / max_iterations
c1 = 1.5
c2 = 1.5

# PSO Algorithm

# Initialize swarm
global_best_costs = []
global_best = BestSolution(position=np.zeros(variables_number), cost=float("inf"))

swarm = []
positions = np.random.randint(low=lower_bound, high=upper_bound + 1, size=(swarm_size, variables_number))
velocities = np.zeros(shape=(swarm_size, variables_number))
for position, velocity in zip(positions, velocities):
    particle_cost = objective_function(position)
    swarm.append(
        Particle(
            position=position,
            velocity=velocity,
            cost=particle_cost,
            personal_best=BestSolution(position=position, cost=particle_cost),
        )
    )
    if particle_cost < global_best.cost:
        global_best = BestSolution(position=position, cost=particle_cost)
global_best_costs.append(global_best.cost)
# Main loop
inertia_weight = initial_weight
for _ in range(max_iterations):
    # Update velocities
    for particle in swarm:
        particle.velocity = (
            inertia_weight * particle.velocity
            + c1 * np.random.uniform() * (particle.personal_best.position - particle.position)
            + c2 * np.random.uniform() * (global_best.position - particle.position)
        )
        # Applying velocity bounds
        particle.velocity = np.maximum(particle.velocity, min_velocity)
        particle.velocity = np.minimum(particle.velocity, max_velocity)

        # Sigmoid transfer function
        T = 1 / (1 + np.exp(-particle.velocity))
        r = np.random.uniform(size=variables_number)

        # Update position
        particle.position = np.where(r < T, 0, 1)

        # particle.position = particle.position + particle.velocity
        # # Applying position bounds
        # particle.position = np.maximum(particle.position, lower_bound)
        # particle.position = np.minimum(particle.position, upper_bound)

        particle.cost = objective_function(particle.position)

        if particle.cost < particle.personal_best.cost:
            particle.personal_best.position = particle.position
            particle.personal_best.cost = particle.cost
            if particle.personal_best.cost < global_best.cost:
                global_best = particle.personal_best
    inertia_weight *= inertia_dampening
    global_best_costs.append(global_best.cost)
print(global_best)

plt.figure()
plt.plot(global_best_costs)
plt.figure()
plt.semilogy(global_best_costs)
plt.show()
