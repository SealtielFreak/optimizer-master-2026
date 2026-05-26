import numpy as np

from mealpy.optimizer.classic import ClassicOptimizer
from mealpy.utils.agent import Agent

from scipy.special import gamma
from scipy.stats import wilcoxon


class DEAL(ClassicOptimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
        """

        super().__init__(**kwargs)

        self.historial_best_pop = []
        self.w = 1
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w"])
        self.sort_flag = False
        self.is_parallelizable = False

        self.v_max = 0
        self.v_min = 0

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
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        # Update weight after each move count  (weight down)

        pos_new_solutions = []

        if len(self.historial_best_pop) > 2:
            self.w = wilcoxon([p.target.fitness for p in self.historial_best_pop]).pvalue

        for idx in range(0, self.pop_size):
            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)

            social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)

            self.pop[idx].velocity = self.w * self.pop[idx].velocity + cognitive + social

            # pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.pop[idx].solution + self.pop[idx].velocity

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

        gbest = self.get_best_agent(self.pop, self.problem.minmax)
        self.historial_best_pop.append(gbest)
        self.gbest = sorted(self.historial_best_pop, key=lambda p: p.target.fitness)[0]
