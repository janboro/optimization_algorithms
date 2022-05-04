import numpy as np
from data_type.PSO_params import PSOParams
from data_type.problem import Problem
from data_type.particle import Particle
from data_type.problem import Problem
from data_type.best_solution import BestSolution


class PSO:
    def __init__(self, problem: Problem, pso_params: PSOParams = PSOParams()):
        self.problem = problem
        self.pso_params = pso_params
        self.global_best = BestSolution(cost=float("inf"))
        self.swarm = self.initialize_swarm()
        self.best_costs = []  # used to plot efficiency over time

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.pso_params.swarm_size):
            initial_position = np.random.uniform(
                low=self.problem.bounds.lower, high=self.problem.bounds.upper, size=self.problem.variables
            )
            initial_velocity = np.zeros(self.problem.variables)
            cost = self.problem.cost_function(initial_position)

            particle = Particle(
                position=initial_position,
                velocity=initial_velocity,
                cost=cost,
                personal_best=BestSolution(position=initial_position, cost=cost),
            )
            swarm.append(particle)
            if particle.personal_best.cost < self.global_best.cost:
                self.global_best = particle.personal_best
        return swarm

    def get_new_velocity(self, particle: Particle):
        inertia_term = self.pso_params.inertia * particle.velocity
        cognitive_component = (
            self.pso_params.cognitive_acceleration
            * np.random.uniform(size=self.problem.variables)
            * (particle.personal_best.position - particle.position)
        )
        social_component = (
            self.pso_params.social_acceleration
            * np.random.uniform(size=self.problem.variables)
            * (self.global_best.position - particle.position)
        )

        velocity = inertia_term + cognitive_component + social_component
        return velocity

    def apply_position_bounds(self, particle: Particle):
        particle.position = np.maximum(particle.position, self.problem.bounds.lower)
        particle.position = np.minimum(particle.position, self.problem.bounds.upper)

    def apply_vector_bounds(self, particle: Particle):
        particle.velocity = np.maximum(particle.velocity, self.problem.bounds.min_velocity)
        particle.velocity = np.minimum(particle.velocity, self.problem.bounds.max_velocity)

    def apply_bounds(self, particle: Particle):
        self.apply_position_bounds(particle=particle)
        if self.problem.bounds.constrain_velocity:
            self.apply_vector_bounds(particle=particle)

    def update_global_best_solution(self, particle: Particle):
        if particle.personal_best.cost < self.global_best.cost:
            self.global_best = particle.personal_best

    def update_particle_best_solution(self, particle: Particle):
        if particle.cost < particle.personal_best.cost:
            particle.personal_best.position = particle.position
            particle.personal_best.cost = particle.cost
            self.update_global_best_solution(particle=particle)

    def run(self):
        for i in range(self.pso_params.iterations):
            print(f"Iteration: {i}")
            for particle in self.swarm:
                velocity = self.get_new_velocity(particle=particle)

                particle.velocity = velocity
                particle.position = particle.position + particle.velocity
                particle.cost = self.problem.cost_function(particle.position)

                self.apply_bounds(particle=particle)
                self.update_particle_best_solution(particle=particle)

            self.pso_params.inertia *= self.pso_params.inertia_dampening
            self.best_costs.append(self.global_best.cost)

        return self.global_best
