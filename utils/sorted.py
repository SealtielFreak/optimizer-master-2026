from mealpy.utils.agent import Agent


def sorted_population(population: list[Agent] | None, minmax: str = "min") -> list[Agent]:
    if population is None:
        raise ValueError("Population is not initialized")

    sorted(population, key=lambda p: p.target.fitness)

    if minmax == "max":
        population = population[::-1]

    return population
