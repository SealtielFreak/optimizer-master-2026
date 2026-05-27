import math

import numpy as np
from mealpy.optimizer.classic import ClassicOptimizer
from mealpy.utils.agent import Agent
from scipy.stats import wilcoxon


class DELU(ClassicOptimizer):
    """
        Differential Evolutive Lineal Uniform
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, minimum_pop: int = 4, uf: float = 0.1, cr: float = 0.9,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.history_best_pop = []
        self.history_worst_pop = []

        self.counter_pop_tolerance = 10
        self.counter_pop = 0

        self.b_stats = 1
        self.w_stats = 1
        self.p = 1

        self.g_best_history = None
        self.g_worst_history = None

        self.minimum_pop = self.validator.check_int("minimum_pop", minimum_pop, [4, 100000])
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.uf = self.validator.check_float("uf", uf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))

        self.set_parameters(["epoch", "pop_size", "uf", "cr", "minimum_pop"])
        self.sort_flag = False
        self.is_parallelizable = False

        self.default_pop_size = self.pop_size

        self.v_max = 0
        self.v_min = 0

    def mutation(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.correct_solution(pos_new)

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()

        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()

        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)

        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        if len(self.history_best_pop) > 1:
            self.g_best_history = self.get_sorted_population(self.history_best_pop, self.problem.minmax)[0]

            best_pop_fitness = [p.target.fitness for p in self.history_best_pop]
            self.b_stats = wilcoxon(best_pop_fitness).pvalue
        else:
            self.g_best_history = self.g_best

        if len(self.history_worst_pop) > 1:
            self.g_worst_history = self.get_sorted_population(self.history_worst_pop, self.problem.minmax)[-1]

            wors_pop_fitness = [p.target.fitness for p in self.history_worst_pop]
            self.w_stats = wilcoxon(wors_pop_fitness).pvalue
        else:
            self.g_worst_history = self.g_worst

        current_pop_len = len(self.pop)

        if self.b_stats < 0.05 and current_pop_len > self.minimum_pop and epoch > self.epoch // 3:
            self.pop.pop(0)
            self.pop_size -= 1
            self.counter_pop += 1
        elif current_pop_len < self.default_pop_size:
            self.pop += [self.generate_agent()]
            self.pop_size += 1

        diff = math.fabs(self.w_stats - self.b_stats)
        self.p = 1 if diff == 0 else diff

        pos_new_solutions = []

        for idx in range(0, self.pop_size):
            self.pop[idx].velocity *= self.p

            idx_0, idx_1 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = (self.pop[idx].solution + self.uf * (
                    self.g_best_history.solution - self.pop[idx].solution) + self.uf * (
                               self.pop[idx_0].solution - self.pop[idx_1].solution)) + self.pop[idx].velocity

            pos_new = self.mutation(self.pop[idx].solution, pos_new)

            pos_new = self.correct_solution(pos_new)
            pos_new_solutions.append(pos_new)

        new_targets = []

        for idx, pos_new in enumerate(pos_new_solutions):
            target = self.get_target(pos_new)
            new_targets.append((pos_new, target))

        for idx, (pos_new, target) in enumerate(new_targets):
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(
                    solution=pos_new.copy(),
                    target=target.copy()
                )

            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(
                    local_solution=pos_new.copy(),
                    local_target=target.copy()
                )

        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)

        self.history_best_pop += [self.pop[0]]
        self.history_worst_pop += [self.pop[-1]]

        self.g_best = self.get_sorted_population(self.pop, self.problem.minmax)[0]
