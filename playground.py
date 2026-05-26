from mealpy import FloatVar

from collection.b2026.deal import DEAL
from bench.func import get_default

problem = get_default(0, ndim=20)
model = DEAL(epoch=250, pop_size=75)

problem_dict = {
    "obj_func": problem.evaluate,
    "bounds": FloatVar(problem.lb, problem.ub),
    "minmax": "min",
}

result = model.solve(problem_dict)
print(result)
