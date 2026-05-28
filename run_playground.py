from mealpy import FloatVar

from bench.func import get_default
from collection.b2026.de import DEAL

problem = get_default(0, ndim=30)
model = DEAL(epoch=2000, pop_size=100, stats_mode='mode', n_layers=5)

problem_dict = {
    "obj_func": problem.evaluate,
    "bounds": FloatVar(problem.lb, problem.ub),
    "minmax": "min",
}

result = model.solve(problem_dict)
print(result)
