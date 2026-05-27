from mealpy import FloatVar

from collection.b2026.deal import DEAL
from bench.func import get_default

from scipy.stats import wilcoxon, norm, mode

problem = get_default(0, ndim=50)
model = DEAL(epoch=350, pop_size=120)

problem_dict = {
    "obj_func": problem.evaluate,
    "bounds": FloatVar(problem.lb, problem.ub),
    "minmax": "min",
}

result = model.solve(problem_dict)
print(result)

# print(mode([p.target.fitness for p in model.history_best_pop], keepdims=True)[0])
# print(mode([p.target.fitness for p in model.history_worst_pop], keepdims=True)[0])
