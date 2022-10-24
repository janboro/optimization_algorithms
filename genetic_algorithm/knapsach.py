from random import random, choices, randint, randrange
from typing import List, Callable, Tuple
from collections import namedtuple
from functools import partial

Genome = List[int]
Population = List[Genome]
Item = namedtuple("Item", ["name", "value", "weight"])
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


items = [
    Item("Laptop", 500, 2200),
    Item("Headphones", 150, 160),
    Item("Coffee Mug", 60, 350),
    Item("Notepad", 40, 333),
    Item("Water Bottle", 30, 192),
]

lot_of_items = [
    Item("Mints", 5, 25),
    Item("Socks", 10, 38),
    Item("Tissues", 15, 80),
    Item("Phone", 50, 200),
    Item("Hat", 100, 70),
] + items


def generate_genome(genome_length: int) -> Genome:
    return choices([0, 1], k=genome_length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length=genome_length) for _ in range(size)]


def fitness(genome: Genome, items: List[Item], weight_limit: int) -> int:
    if len(genome) != len(items):
        raise ValueError("The genome and items need to be of the same length")

    weight = 0
    value = 0

    for i, item in enumerate(items):
        if genome[i] == 1:
            weight += item.weight
            value += item.value
            if weight > weight_limit:
                return 0

    return value


def selection_pair(population: Population, fitness_function: FitnessFunc) -> Tuple[Genome, Genome]:
    # the weight param allows us to pick the genome with a better fitness value with a higher probability
    # the k=2 draws two choices from the pool of genomes in the population
    return tuple(choices(population=population, weights=[fitness_function(genome) for genome in population], k=2))


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes must be of the same length")

    length = len(a)
    if length < 2:
        return a, b

    crossover_point = randint(1, length - 1)
    return a[0:crossover_point] + b[crossover_point:], b[0:crossover_point] + a[crossover_point:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100,
) -> Tuple[Population, int]:
    population = populate_func()

    i = 0
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)
    return population, i


def genome_to_items(genome: Genome, items: List[Item]) -> List[Item]:
    result = []
    for i, item in enumerate(items):
        if genome[i] == 1:
            result.append(item.name)
    return result


population, generations = run_evolution(
    populate_func=partial(generate_population, size=10, genome_length=len(items)),
    fitness_func=partial(fitness, items=items, weight_limit=3000),
    fitness_limit=740,
    generation_limit=100,
)


print(f"Number of generations: {generations}")
print(f"Best solution: {genome_to_items(population[0], items)}")
