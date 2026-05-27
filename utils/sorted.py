from mealpy.utils.agent import Agent


def sorted_population(population: list[Agent], minmax: str = "min") -> list[Agent]:
    sorted(population, key=lambda p: p.target.fitness)

    if minmax == "max":
        population = population[::-1]

    return population
