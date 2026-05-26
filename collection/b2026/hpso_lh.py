import numpy as np

from mealpy.optimizer.classic import ClassicOptimizer
from mealpy.utils.agent import Agent

from scipy.special import gamma


class HPSO_LH2026A(ClassicOptimizer):
    def __init__(self, epoch: int = 1000, pop_size: int = 50,
                 c1: float = 1.49, c2: float = 1.49,
                 beta: float = 1.5, alpha: float = 0.01,
                 s_max: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.beta = self.validator.check_float("beta", beta, (0.1, 2.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0.0, 1.0))
        self.s_max = self.validator.check_int("s_max", s_max, [1, 100])
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "beta", "alpha", "s_max"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.center = 0.5 * (self.problem.lb + self.problem.ub)
        self.R0 = 0.5 * np.linalg.norm(self.problem.ub - self.problem.lb)
        num = gamma(1.0 + self.beta) * np.sin(np.pi * self.beta / 2.0)
        den = gamma((1.0 + self.beta) / 2.0) * self.beta * (2.0 ** ((self.beta - 1.0) / 2.0))
        self.sigma_u = (num / den) ** (1.0 / self.beta)
        self.stagnation = np.zeros(self.pop_size, dtype=int)

    def _sample_hypersphere(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, 1.0, d)
        u_hat = u / (np.linalg.norm(u) + 1e-12)
        r = self.R0 * self.generator.uniform(0.0, 1.0) ** (1.0 / d)
        return self.center + r * u_hat

    def _levy_step(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, self.sigma_u, d)
        v = self.generator.normal(0.0, 1.0, d)
        return u / (np.abs(v) ** (1.0 / self.beta) + 1e-12)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self._sample_hypersphere()
            solution = self.correct_solution(solution)
        velocity = np.zeros(self.problem.n_dims)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.problem.lb, self.problem.ub)

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            cognitive = self.c1 * r1 * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * r2 * (self.g_best.solution - self.pop[idx].solution)
            self.pop[idx].velocity = cognitive + social
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            self.pop[idx].solution = pos_new
            self.pop[idx].target = target

            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].local_solution = pos_new.copy()
                self.pop[idx].local_target = target.copy()
                self.stagnation[idx] = 0
            else:
                self.stagnation[idx] += 1

            if self.stagnation[idx] >= self.s_max:
                step = self._levy_step()
                pos_levy = self.pop[idx].solution + self.alpha * step * (self.pop[idx].solution - self.g_best.solution)
                pos_levy = self.correct_solution(pos_levy)
                target_levy = self.get_target(pos_levy)
                self.pop[idx].solution = pos_levy
                self.pop[idx].target = target_levy
                self.pop[idx].velocity = np.zeros(self.problem.n_dims)
                if self.compare_target(target_levy, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].local_solution = pos_levy.copy()
                    self.pop[idx].local_target = target_levy.copy()
                self.stagnation[idx] = 0


class HPSO_LH2026B(ClassicOptimizer):
    def __init__(self, epoch: int = 1000, pop_size: int = 50,
                 c1: float = 2.05, c2: float = 2.05,
                 beta: float = 1.5, alpha: float = 0.05,
                 s_max: int = 3, archive_size: int = 20,
                 ref_patience: int = 5, tol: float = 1e-6,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.beta = self.validator.check_float("beta", beta, (0.1, 2.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0.0, 1.0))
        self.s_max = self.validator.check_int("s_max", s_max, [1, 100])
        self.archive_size = self.validator.check_int("archive_size", archive_size, [2, 500])
        self.ref_patience = self.validator.check_int("ref_patience", ref_patience, [1, 1000])
        self.tol = self.validator.check_float("tol", tol, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "beta", "alpha",
                             "s_max", "archive_size", "ref_patience", "tol"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.center = 0.5 * (self.problem.lb + self.problem.ub)
        self.R0 = 0.5 * np.linalg.norm(self.problem.ub - self.problem.lb)
        num = gamma(1.0 + self.beta) * np.sin(np.pi * self.beta / 2.0)
        den = gamma((1.0 + self.beta) / 2.0) * self.beta * (2.0 ** ((self.beta - 1.0) / 2.0))
        self.sigma_u = (num / den) ** (1.0 / self.beta)
        phi = self.c1 + self.c2
        if phi <= 4.0:
            self.chi = 1.0
        else:
            self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        self.stagnation = np.zeros(self.pop_size, dtype=int)
        self.archive_sols = np.empty((0, self.problem.n_dims))
        self.archive_fits = np.empty((0,))
        self.ref_fitness = None
        self.ref_solution = None
        self.ref_stagnation = 0

    def _sample_hypersphere(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, 1.0, d)
        u_hat = u / (np.linalg.norm(u) + 1e-12)
        r = self.R0 * self.generator.uniform(0.0, 1.0) ** (1.0 / d)
        return self.center + r * u_hat

    def _levy_step(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, self.sigma_u, d)
        v = self.generator.normal(0.0, 1.0, d)
        return u / (np.abs(v) ** (1.0 / self.beta) + 1e-12)

    def _is_better(self, fit_a: float, fit_b: float) -> bool:
        if self.problem.minmax == "min":
            return fit_a < fit_b
        return fit_a > fit_b

    def _update_archive(self, solution: np.ndarray, fitness: float) -> None:
        if self.archive_sols.shape[0] < self.archive_size:
            self.archive_sols = np.vstack([self.archive_sols, solution[None, :]])
            self.archive_fits = np.append(self.archive_fits, fitness)
            return
        if self.problem.minmax == "min":
            worst_idx = int(np.argmax(self.archive_fits))
            if fitness < self.archive_fits[worst_idx]:
                self.archive_sols[worst_idx] = solution
                self.archive_fits[worst_idx] = fitness
        else:
            worst_idx = int(np.argmin(self.archive_fits))
            if fitness > self.archive_fits[worst_idx]:
                self.archive_sols[worst_idx] = solution
                self.archive_fits[worst_idx] = fitness

    def _archive_anchor(self) -> np.ndarray:
        if self.archive_sols.shape[0] == 0:
            return self.g_best.solution.copy()
        order = np.argsort(self.archive_fits) if self.problem.minmax == "min" else np.argsort(-self.archive_fits)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        weights = (len(order) - ranks).astype(float)
        weights = weights / weights.sum()
        idx = self.generator.choice(len(order), p=weights)
        return self.archive_sols[idx].copy()

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self._sample_hypersphere()
            solution = self.correct_solution(solution)
        velocity = np.zeros(self.problem.n_dims)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.problem.lb, self.problem.ub)

    def after_initialization(self):
        super().after_initialization()
        for agent in self.pop:
            self._update_archive(agent.local_solution.copy(), agent.local_target.fitness)
        self.ref_fitness = self.g_best.target.fitness
        self.ref_solution = self.g_best.solution.copy()

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            cognitive = self.c1 * r1 * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * r2 * (self.g_best.solution - self.pop[idx].solution)
            self.pop[idx].velocity = self.chi * (cognitive + social)
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            self.pop[idx].solution = pos_new
            self.pop[idx].target = target

            prev_local_fit = self.pop[idx].local_target.fitness
            improved_local = self.compare_target(target, self.pop[idx].local_target, self.problem.minmax)
            if improved_local:
                self.pop[idx].local_solution = pos_new.copy()
                self.pop[idx].local_target = target.copy()
                self._update_archive(pos_new.copy(), target.fitness)
            meaningful = improved_local and abs(prev_local_fit - target.fitness) > self.tol
            if meaningful:
                self.stagnation[idx] = 0
            else:
                self.stagnation[idx] += 1

            swarm_stuck = self.ref_stagnation >= self.ref_patience
            if self.stagnation[idx] >= self.s_max or swarm_stuck:
                anchor = self._archive_anchor()
                step = self._levy_step()
                scale = self.alpha * (self.problem.ub - self.problem.lb)
                pos_mut = anchor + step * scale
                pos_mut = self.correct_solution(pos_mut)
                target_mut = self.get_target(pos_mut)

                if self.compare_target(target_mut, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx].solution = pos_mut
                    self.pop[idx].target = target_mut
                    self.pop[idx].velocity = np.zeros(self.problem.n_dims)
                    if self.compare_target(target_mut, self.pop[idx].local_target, self.problem.minmax):
                        self.pop[idx].local_solution = pos_mut.copy()
                        self.pop[idx].local_target = target_mut.copy()
                        self._update_archive(pos_mut.copy(), target_mut.fitness)
                self.stagnation[idx] = 0

        fits = np.array([a.target.fitness for a in self.pop])
        epoch_best_idx = int(np.argmin(fits)) if self.problem.minmax == "min" else int(np.argmax(fits))
        epoch_best_fit = self.pop[epoch_best_idx].target.fitness

        if self.ref_fitness is None or (self._is_better(epoch_best_fit, self.ref_fitness)
                                        and abs(epoch_best_fit - self.ref_fitness) > self.tol):
            self.ref_fitness = epoch_best_fit
            self.ref_solution = self.pop[epoch_best_idx].solution.copy()
            self.ref_stagnation = 0
        else:
            if self._is_better(epoch_best_fit, self.ref_fitness):
                self.ref_fitness = epoch_best_fit
                self.ref_solution = self.pop[epoch_best_idx].solution.copy()
            self.ref_stagnation += 1


class HPSO_LH2026C(ClassicOptimizer):
    def __init__(self, epoch: int = 1000, pop_size: int = 50,
                 c1: float = 2.05, c2: float = 2.05,
                 beta: float = 1.5,
                 s_max: int = 3, archive_size: int = 20,
                 ref_patience: int = 5, tol: float = 1e-6,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.beta = self.validator.check_float("beta", beta, (0.1, 2.0))
        self.s_max = self.validator.check_int("s_max", s_max, [1, 100])
        self.archive_size = self.validator.check_int("archive_size", archive_size, [2, 500])
        self.ref_patience = self.validator.check_int("ref_patience", ref_patience, [1, 1000])
        self.tol = self.validator.check_float("tol", tol, (0.0, 1.0))

        self.set_parameters(["epoch", "pop_size", "c1", "c2", "beta",
                             "s_max", "archive_size", "ref_patience", "tol"])

        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.center = 0.5 * (self.problem.lb + self.problem.ub)
        self.diameter = np.linalg.norm(self.problem.ub - self.problem.lb)
        self.R = 0.5 * self.diameter
        self.R_min = 1e-4 * self.diameter
        self.R_max = 0.5 * self.diameter
        self.c_adapt = 0.817
        self.mut_success = 0
        self.mut_trials = 0

        num = gamma(1.0 + self.beta) * np.sin(np.pi * self.beta / 2.0)
        den = gamma((1.0 + self.beta) / 2.0) * self.beta * (2.0 ** ((self.beta - 1.0) / 2.0))

        self.sigma_u = (num / den) ** (1.0 / self.beta)

        phi = self.c1 + self.c2
        if phi <= 4.0:
            self.chi = 1.0
        else:
            self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))

        self.stagnation = np.zeros(self.pop_size, dtype=int)
        self.archive_sols = np.empty((0, self.problem.n_dims))
        self.archive_fits = np.empty((0,))
        self.ref_fitness = None
        self.ref_solution = None
        self.ref_stagnation = 0

    def _sample_hypersphere(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, 1.0, d)
        u_hat = u / (np.linalg.norm(u) + 1e-12)
        r = self.R * self.generator.uniform(0.0, 1.0) ** (1.0 / d)

        return self.center + r * u_hat

    def _levy_step(self) -> np.ndarray:
        d = self.problem.n_dims
        u = self.generator.normal(0.0, self.sigma_u, d)
        v = self.generator.normal(0.0, 1.0, d)

        return u / (np.abs(v) ** (1.0 / self.beta) + 1e-12)

    def _is_better(self, fit_a: float, fit_b: float) -> bool:
        if self.problem.minmax == "min":
            return fit_a < fit_b

        return fit_a > fit_b

    def _update_archive(self, solution: np.ndarray, fitness: float) -> None:
        if self.archive_sols.shape[0] < self.archive_size:
            self.archive_sols = np.vstack([self.archive_sols, solution[None, :]])
            self.archive_fits = np.append(self.archive_fits, fitness)
            return

        if self.problem.minmax == "min":
            worst_idx = int(np.argmax(self.archive_fits))
            if fitness < self.archive_fits[worst_idx]:
                self.archive_sols[worst_idx] = solution
                self.archive_fits[worst_idx] = fitness
        else:
            worst_idx = int(np.argmin(self.archive_fits))

            if fitness > self.archive_fits[worst_idx]:
                self.archive_sols[worst_idx] = solution
                self.archive_fits[worst_idx] = fitness

    def _archive_anchor(self) -> np.ndarray:
        if self.archive_sols.shape[0] == 0:
            return self.g_best.solution.copy()

        order = np.argsort(self.archive_fits) if self.problem.minmax == "min" else np.argsort(-self.archive_fits)
        ranks = np.empty_like(order)

        ranks[order] = np.arange(len(order))
        weights = (len(order) - ranks).astype(float)
        weights = weights / weights.sum()

        idx = self.generator.choice(len(order), p=weights)

        return self.archive_sols[idx].copy()

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self._sample_hypersphere()
            solution = self.correct_solution(solution)

        velocity = np.zeros(self.problem.n_dims)
        local_pos = solution.copy()

        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()

        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.problem.lb, self.problem.ub)

    def after_initialization(self):
        super().after_initialization()

        for agent in self.pop:
            self._update_archive(agent.local_solution.copy(), agent.local_target.fitness)

        self.ref_fitness = self.g_best.target.fitness
        self.ref_solution = self.g_best.solution.copy()

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)

            cognitive = self.c1 * r1 * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * r2 * (self.g_best.solution - self.pop[idx].solution)

            self.pop[idx].velocity = self.chi * (cognitive + social)

            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)

            self.pop[idx].solution = pos_new
            self.pop[idx].target = target

            prev_local_fit = self.pop[idx].local_target.fitness
            improved_local = self.compare_target(target, self.pop[idx].local_target, self.problem.minmax)

            if improved_local:
                self.pop[idx].local_solution = pos_new.copy()
                self.pop[idx].local_target = target.copy()
                self._update_archive(pos_new.copy(), target.fitness)

            meaningful = improved_local and abs(prev_local_fit - target.fitness) > self.tol

            if meaningful:
                self.stagnation[idx] = 0
            else:
                self.stagnation[idx] += 1

            swarm_stuck = self.ref_stagnation >= self.ref_patience
            if self.stagnation[idx] >= self.s_max or swarm_stuck:
                anchor = self._archive_anchor()
                step = self._levy_step()
                pos_mut = anchor + step * self.R
                pos_mut = self.correct_solution(pos_mut)
                target_mut = self.get_target(pos_mut)

                self.mut_trials += 1

                if self.compare_target(target_mut, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx].solution = pos_mut
                    self.pop[idx].target = target_mut
                    self.pop[idx].velocity = np.zeros(self.problem.n_dims)
                    self.mut_success += 1

                    if self.compare_target(target_mut, self.pop[idx].local_target, self.problem.minmax):
                        self.pop[idx].local_solution = pos_mut.copy()
                        self.pop[idx].local_target = target_mut.copy()
                        self._update_archive(pos_mut.copy(), target_mut.fitness)

                self.stagnation[idx] = 0

        fits = np.array([a.target.fitness for a in self.pop])
        epoch_best_idx = int(np.argmin(fits)) if self.problem.minmax == "min" else int(np.argmax(fits))
        epoch_best_fit = self.pop[epoch_best_idx].target.fitness

        if self.ref_fitness is None or (self._is_better(epoch_best_fit, self.ref_fitness)
                                        and abs(epoch_best_fit - self.ref_fitness) > self.tol):
            self.ref_fitness = epoch_best_fit
            self.ref_solution = self.pop[epoch_best_idx].solution.copy()
            self.ref_stagnation = 0
        else:
            if self._is_better(epoch_best_fit, self.ref_fitness):
                self.ref_fitness = epoch_best_fit
                self.ref_solution = self.pop[epoch_best_idx].solution.copy()

            self.ref_stagnation += 1

        if self.mut_trials > 0:
            rate = self.mut_success / self.mut_trials

            if rate > 0.2:
                self.R = min(self.R / self.c_adapt, self.R_max)
            elif rate < 0.2:
                self.R = max(self.R * self.c_adapt, self.R_min)

            self.mut_success = 0
            self.mut_trials = 0

        if self.ref_stagnation >= 2 * self.ref_patience:
            self.R = min(self.R / (self.c_adapt ** 5), self.R_max)
            self.ref_stagnation = 0
