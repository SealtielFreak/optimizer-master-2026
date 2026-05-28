from mealpy import FloatVar

from bench.func import get_default
from collection.b2026.de import MG_L_SHADE

problem = get_default(0, ndim=30)
model = MG_L_SHADE(epoch=1300, pop_size=150, stats_mode='mode', n_layers=3)

problem_dict = {
    "obj_func": problem.evaluate,
    "bounds": FloatVar(problem.lb, problem.ub),
    "minmax": "min",
}

result = model.solve(problem_dict)
print(result)
