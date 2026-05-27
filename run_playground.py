from mealpy import FloatVar

from collection.b2026.deal import DELU
from bench.func import get_default


problem = get_default(0, ndim=50)
model = DELU(epoch=350, pop_size=120, stats_mode='mode')

problem_dict = {
    "obj_func": problem.evaluate,
    "bounds": FloatVar(problem.lb, problem.ub),
    "minmax": "min",
}

result = model.solve(problem_dict)
print(result)
