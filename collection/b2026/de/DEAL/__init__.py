import dataclasses
import math

import numpy as np
from mealpy.optimizer.classic import ClassicOptimizer
from mealpy.utils.agent import Agent
from scipy.stats import cauchy

from utils.sorted import sorted_population

_STATS_MODES = 'mode', 'sorted', 'median', 'mean'


@dataclasses.dataclass
class Layer:
    local_pop: list[Agent]
    dyn_pop_archive: list[Agent]
    dyn_miu_f: list[float]
    dyn_miu_cr: list[float]
    g_best: Agent

    _max_pop_size: int

    dyn_pop_size: int = 0
    k_counter: int = 0

    @property
    def pop_size(self):
        return self._max_pop_size

    @property
    def n_min(self):
        return math.ceil(self.pop_size / 5)

    def __float__(self):
        return self.g_best.target.fitness


class DEAL(ClassicOptimizer):
    """
        Differential Evolution Adaptative-Layers: Based in DE, JADE, L-SHADED
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            minimum_pop: int = 4,
            miu_f: float = 0.5,
            miu_cr: float = 0.5,
            cr: float = 0.9,
            ap: float = 0.1,
            n_layers: int = 5,
            c1: float = 2.05,
            c2: float = 2.05,
            w: float = 0.4,
            stats_mode: str = "sorted",
            **kwargs: object
    ) -> None:
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.ap = self.validator.check_float("ap", ap, (0, 1.0))
        self.minimum_pop = self.validator.check_int("minimum_pop", minimum_pop, [4, 100000])
        self.n_layers = self.validator.check_int("n_layers", n_layers, [1, 100000])
        self.stats_mode = self.validator.check_str("stats_mode", stats_mode, _STATS_MODES)

        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w = self.validator.check_float("w", w, (0, 1.0))

        self.set_parameters(
            ["epoch", "pop_size", "miu_f", "miu_cr", "cr", "minimum_pop", "n_layers", "stats_mode", "c1", "c2", "w"]
        )

        self.all_global_history_best_pop = []
        self.layer_all_g_best = []
        self.default_pop_size = self.pop_size

        self.dyn_miu_cr = 0
        self.dyn_miu_f = 0

        self.sort_flag = False
        self.is_parallelizable = False

        self.all_g_best_layers = []

        self.all_layers = []
        self.layers_size = 0

    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)

        return up / down if down != 0 else 0.5

    def initialize_variables(self):
        self.layers_size = self.pop_size // self.n_layers

        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def initialization(self) -> None:
        super().initialization()

        self.pop = sorted_population(self.pop)
        self.g_best = self.pop[0].copy()

        for i in range(0, self.n_layers):
            a, b = (
                self.layers_size * i,
                self.layers_size * (i + 1)
            )

            local_pop = [*self.pop[a:b]]
            local_pop = sorted_population(local_pop, self.problem.minmax)
            max_pop_size = len(local_pop)
            g_best = local_pop[0]

            layer = Layer(
                local_pop=local_pop,
                dyn_pop_archive=[],
                dyn_miu_f=self.miu_f * np.ones(max_pop_size),
                dyn_miu_cr=self.miu_cr * np.ones(max_pop_size),
                g_best=g_best,
                dyn_pop_size=max_pop_size,

                _max_pop_size=max_pop_size
            )

            self.all_layers.append(layer)

    def mutation(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)

        return self.correct_solution(pos_new)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        local_pos = solution.copy()
        velocity = self.generator.uniform(self.v_min, self.v_max)

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

    def evolve(self, epoch: int) -> None:
        def evolve_layer(n: int, epoch: int, layer: Layer):
            local_pop = layer.local_pop

            pop_old = [agent.copy() for agent in local_pop]
            pop_sorted = sorted_population(local_pop, self.problem.minmax)

            list_f = []
            list_cr = []
            list_f_index = []
            list_cr_index = []

            list_f_new = np.ones(len(local_pop))
            list_cr_new = np.ones(len(local_pop))

            pop = []

            for idx in range(0, len(local_pop)):
                idx_rand = self.generator.integers(0, layer.pop_size)
                cr = self.generator.normal(layer.dyn_miu_cr[idx_rand], 0.1)
                cr = np.clip(cr, 0, 1)
                while True:
                    f = cauchy.rvs(layer.dyn_miu_f[idx_rand], 0.1)
                    if f < 0:
                        continue
                    elif f > 1:
                        f = 1
                    break

                list_cr_new[idx] = cr
                list_f_new[idx] = f

                r1_idx = self.generator.choice(list(set(range(0, len(local_pop))) - {idx}))
                x_r1 = local_pop[r1_idx]

                p = self.generator.uniform(0.15, 0.2)
                top = int(np.ceil(layer.dyn_pop_size * p))
                g_best = pop_sorted[self.generator.integers(0, top)]

                new_pop = self.pop + layer.dyn_pop_archive
                r2_idx = self.generator.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))
                x_r2 = new_pop[r2_idx]

                x_new = (
                    (local_pop[idx].solution + f * (g_best.solution - local_pop[idx].solution) + f * (
                            x_r1.solution - x_r2.solution))
                )
                pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, local_pop[idx].solution)
                j_rand = self.generator.integers(0, self.problem.n_dims)

                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)

                pop.append(agent)
                pop[-1].target = self.get_target(pos_new)

            for idx in range(0, len(local_pop)):
                if self.compare_target(pop[idx].target, local_pop[idx].target, self.problem.minmax):
                    list_cr.append(list_cr_new[idx])
                    list_f.append(list_f_new[idx])
                    list_f_index.append(idx)
                    list_cr_index.append(idx)
                    local_pop[idx] = pop[idx].copy()
                    layer.dyn_pop_archive.append(self.pop[idx].copy())

            temp = len(layer.dyn_pop_archive) - self.pop_size
            if temp > 0:
                idx_list = self.generator.choice(range(0, len(layer.dyn_pop_archive)), temp, replace=False)
                archive_pop_new = []

                for idx, agent in enumerate(layer.dyn_pop_archive):
                    if idx not in idx_list:
                        archive_pop_new.append(agent.copy())

                layer.dyn_pop_archive = archive_pop_new

            if len(list_f) != 0 and len(list_cr) != 0:
                list_fit_old = np.ones(len(list_cr_index))
                list_fit_new = np.ones(len(list_cr_index))
                idx_increase = 0

                for idx in range(0, layer.dyn_pop_size):
                    if idx in list_cr_index:
                        list_fit_old[idx_increase] = pop_old[idx].target.fitness
                        list_fit_new[idx_increase] = local_pop[idx].target.fitness
                        idx_increase += 1

                total_fit = np.sum(np.abs(list_fit_new - list_fit_old))
                list_weights = 0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit

                layer.dyn_miu_cr[layer.k_counter] = np.sum(list_weights * np.array(list_cr))
                layer.dyn_miu_f[layer.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)

                layer.k_counter += 1

                if layer.k_counter >= layer.dyn_pop_size:
                    layer.k_counter = 0

            local_pop = sorted_population(local_pop, self.problem.minmax)
            g_best = local_pop[0].copy()

            layer.dyn_pop_size = round(layer.pop_size + epoch * ((layer.n_min - layer.pop_size) / self.epoch))
            layer.local_pop = local_pop
            layer.g_best = g_best

            return g_best

        pop = []

        sorted(
            self.all_layers, key=lambda l: l.g_best.target.fitness
        )

        for idx, layer in enumerate(self.all_layers[1:]):
            g_best = layer.g_best

            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (g_best.local_solution - g_best.solution)
            social = self.c2 * self.generator.random(self.problem.n_dims) * (g_best.solution - g_best.solution)

            g_best.velocity = self.w * g_best.velocity - cognitive - social

            pos_new = g_best.solution + g_best.velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)

            if self.compare_target(target, g_best.target, self.problem.minmax):
                g_best.update(solution=pos_new.copy(), target=target.copy())

            if self.compare_target(target, g_best.local_target, self.problem.minmax):
                g_best.update(local_solution=pos_new.copy(), local_target=target.copy())

        self.layer_all_g_best = []

        for n, layer in enumerate(self.all_layers):
            layer_g_best = evolve_layer(n, epoch, layer)
            self.layer_all_g_best += [layer_g_best]
            pop += layer.local_pop

        self.pop = pop

        self.layer_all_g_best = sorted_population(self.layer_all_g_best, self.problem.minmax)

        self.all_global_history_best_pop += self.layer_all_g_best
        self.all_global_history_best_pop = sorted_population(self.all_global_history_best_pop, self.problem.minmax)

        self.g_best = self.all_global_history_best_pop[0].copy()
