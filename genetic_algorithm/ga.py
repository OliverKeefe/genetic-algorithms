from random import choices, randint, randrange, random
from typing import List, Callable, Tuple, Optional

"""

Simple Genetic Algorithm.

Using SGA to solve the Knapsack Problem.

Scenario: Lets say you've flown to Barcelona on a cheap Vueling flight.
You only have an allocation of 15 KG and you want to bring back as many of your
impulse purchases from your impromptu shopping trip in Plaça de Catalunya. How do you 
compute the most optimal items.
"""

Genome = List[int]
Population = List[Genome]
FitnessFunction = Callable[[Genome], int]
PopulateFunction = Callable[[], Population]
SelectionFunction = Callable[[Population, FitnessFunction], Tuple[Genome, Genome]]
CrossoverFunction = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunction = Callable[[Genome], Genome]
PrinterFunction = Callable[[Population, int, FitnessFunction], None]

# Items (weight, value) and capacity
items: list[tuple[int, int]] = [
    (12, 4), # Bottle of wine
    (2, 2), # Jamon Iberico
    (1, 2), # A nice t-shirt
    (1, 1), # Slightly less nice t-shirt
    (4, 10) # High-end 8k camera
]
capacity: int = 15


def generate_genome(length: int) -> Genome:
    """
    Generates a genome based on specified length, genomes are a binary list
    that look like this: [1, 0, 0, 1, 1] - In the case of the items above,
    the length is 5.

    :param length: int representing the length of the genome (number of items in the knapsack).
    :return: Genome
    """
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    """
    Generates a population of genomes.
    :param size:
    :param genome_length:
    :return:
    """
    return [generate_genome(genome_length) for _ in range(size)]

#TODO: Finish documentation.
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")

    length = len(a)

    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population: Population, fitness_function: FitnessFunction) -> int:
    return sum(fitness_function(genome) for genome in population)


def selection_pair(population: Population, fitness_function: FitnessFunction) -> Tuple[Genome, Genome]:
    return tuple(choices(
        population=population,
        weights=[fitness_function(gene) for gene in population],
        k=2
    ))


def sort_population(population: Population, fitness_function: FitnessFunction) -> Population:
    return sorted(population, key=fitness_function, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(
        population: Population,
        generation_id: int,
        fitness_function: FitnessFunction
) -> Genome:
    print(f"GENERATION {generation_id:02d}")
    print("-----------------------")
    print("Population:", ", ".join(genome_to_string(gene) for gene in population))
    avg_fitness = population_fitness(population, fitness_function) / len(population)
    print(f"Avg. Fitness: {avg_fitness:.2f}")

    sorted_population = sort_population(population, fitness_function)
    best_genome = sorted_population[0]
    worst_genome = sorted_population[-1]

    print(f"Best: {genome_to_string(best_genome)} ({fitness_function(best_genome)})")
    print(f"Worst: {genome_to_string(worst_genome)} ({fitness_function(worst_genome)})\n")

    return best_genome


def run_evolution(
        populate_function: PopulateFunction,
        fitness_function: FitnessFunction,
        fitness_limit: int,
        selection_function: SelectionFunction = selection_pair,
        crossover_function: CrossoverFunction = single_point_crossover,
        mutation_function: MutationFunction = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunction] = None
) -> Tuple[Population, int]:

    population = populate_function()

    for i in range(generation_limit):
        population = sorted(population, key=fitness_function, reverse=True)

        if printer is not None:
            printer(population, i, fitness_function)

        if fitness_function(population[0]) >= fitness_limit:
            break

        # Preserves best 2 individuals
        next_generation = population[:2]

        for _ in range(len(population) // 2 - 1):
            parents = selection_function(population, fitness_function)
            offspring_a, offspring_b = crossover_function(parents[0], parents[1])
            offspring_a = mutation_function(offspring_a)
            offspring_b = mutation_function(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i

def knapsack_fitness(genome: Genome) -> int:
    total_weight = 0
    total_value = 0
    for gene, (item_weight, item_value) in zip(genome, items):
        if gene == 1:
            total_weight += item_weight
            total_value += item_value

        # Penalize solutions that exceed the capacity
        if total_weight > capacity:
            return 0

    return total_value

def main() -> None:
    population_size = 10
    genome_length = len(items)
    fitness_limit = 15
    generation_limit = 100

    best_population, generation = run_evolution(
        populate_function=lambda: generate_population(population_size, genome_length),
        fitness_function=knapsack_fitness,
        fitness_limit=fitness_limit,
        generation_limit=generation_limit,
        printer=print_stats
    )

    best_genome = best_population[0]

    print("Best solution found:")
    print("Genome:", genome_to_string(best_genome))
    print("Total Value:", knapsack_fitness(best_genome))
    print("Items selected:")
    for i, gene in enumerate(best_genome):
        if gene == 1:
            print(f"Item {i + 1}: Weight={items[i][0]}, Value={items[i][1]}")


if __name__ == "__main__":
    main()
