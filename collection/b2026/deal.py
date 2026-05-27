import dataclasses

import numpy as np

from mealpy.optimizer.classic import ClassicOptimizer
from mealpy.utils.agent import Agent

from scipy.stats import wilcoxon, cauchy


_STATS_MODES = 'mode', 'sorted', 'median', 'mean'

@dataclasses.dataclass
class LayersPopulation:
    pop: list[Agent]
    pop_size: int
    minimum_pop: int
    default_pop_size: int

    ap: float
    dyn_miu_cr: float
    dyn_miu_f: float

    n_epoch: int

    all_local_history_best_pop: list[Agent] = dataclasses.field(default_factory=list)
    all_local_history_wort_pop: list[Agent] = dataclasses.field(default_factory=list)

    g_best: Agent | None = dataclasses.field(default_factory=Agent)
    g_best_history: Agent | None = dataclasses.field(default_factory=Agent)

    generator: np.random.Generator = dataclasses.field(default_factory=np.random.default_rng)

    mutation = lambda a, b: a
    correct_solution = lambda pos: pos

    def evolve(self, epoch, minmax, target, mode: str = "sorted") -> Agent | None:
        """
        Evolve method for layer execute (compatible with)
        """
        b_stats = 1

        list_f = []
        list_cr = []
        temp_f = []
        temp_cr = []

        current_history_best_pop = []

        self.g_best_history = self.g_best

        if len(self.all_local_history_best_pop) > 5:
            current_history_best_pop = self.sorted_best(minmax)
            self.g_best_history = current_history_best_pop[0].copy()

            best_pop_fitness = [p.target.fitness for p in self.all_local_history_best_pop]
            b_stats = wilcoxon(best_pop_fitness).pvalue

        current_pop_len = len(self.pop)
        g_best_mode_solution = self.g_best_history.solution

        if mode in _STATS_MODES and len(current_history_best_pop) > (self.default_pop_size / 3):
            g_best_mode_solution = stats_solution(
                mode,
                np.array([p.solution for p in current_history_best_pop])
            )

        self.pop = self.sorted(minmax)

        if b_stats < 0.05 and current_pop_len > self.minimum_pop and epoch > self.n_epoch // 3:
            self.all_local_history_best_pop += [self.pop[-1].copy()]
            self.pop_size -= 1
            self.pop.pop(-1)
        elif current_pop_len < self.default_pop_size:
            p = self.all_local_history_wort_pop[0].copy()
            self.all_local_history_wort_pop.pop(0)
            self.pop += [p]
            self.pop_size += 1

        pos_new_solutions = []
        for idx in range(0, self.pop_size):
            cr = self.generator.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)

            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)

                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break

            temp_f += [f]
            temp_cr += [cr]

            r1_idx, r2_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            x_new = (
                (self.pop[idx].solution + f * (g_best_mode_solution - self.pop[idx].solution) + f * (
                            self.pop[r1_idx].solution - self.pop[r2_idx].solution))
            )

            pos_new = self.mutation(self.pop[idx].solution, x_new)
            pos_new = self.correct_solution(pos_new)
            pos_new_solutions.append(pos_new)

        new_targets = []
        for idx, pos_new in enumerate(pos_new_solutions):
            target = target(pos_new)
            new_targets.append((pos_new, target))

        for idx, (pos_new, target) in enumerate(new_targets):
            if compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(
                    solution=pos_new.copy(),
                    target=target.copy()
                )

                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])

        self.dyn_miu_cr, self.dyn_miu_f = update_history(
            self.dyn_miu_cr, self.dyn_miu_f, self.ap, list_cr, list_f,
        )

        self.pop = self.sorted(self.problem.minmax)
        self.g_best = self.pop[0].copy()
        self.all_local_history_best_pop += [self.g_best.copy()]
        self.pop = self.sorted(minmax)
        self.g_best = self.pop[0].copy()
        self.all_local_history_best_pop += [self.g_best.copy()]

        return self.g_best

    def sorted(self, minmax: str = "min") -> list[Agent]:
        sorted(self.pop, key=lambda p: p.target.fitness)

        if minmax == "max":
            self.pop = self.pop[::-1]

        return self.pop

    def sorted_best(self, minmax: str = "min") -> list[Agent]:
        sorted(self.all_local_history_best_pop, key=lambda p: p.target.fitness)

        if minmax == "max":
            self.all_local_history_best_pop = self.all_local_history_best_pop[::-1]

        return self.all_local_history_best_pop

    def __iter__(self):
        return iter(self.pop)


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
            cluster: int = 5,
            stats_mode: str = "sorted",
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """

        super().__init__(**kwargs)

        self.all_global_history_best_pop = []
        self.all_global_history_wort_pop = []

        self.counter_pop_tolerance = 10
        self.counter_pop = 0

        self.b_stats = 1
        self.w_stats = 1
        self.impulse = 1

        self.g_best_history = None

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.ap = self.validator.check_float("ap", ap, (0, 1.0))
        self.minimum_pop = self.validator.check_int("minimum_pop", minimum_pop, [4, 100000])
        self.n_layers = self.validator.check_int("n_layers", cluster, [1, 100000])
        self.stats_mode = self.validator.check_str("stats_mode", stats_mode, _STATS_MODES)

        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "cr", "minimum_pop", "n_layers", "stats_mode"])

        self.default_pop_size = self.pop_size

        self.v_max = 0
        self.v_min = 0

        self.dyn_miu_cr = 0
        self.dyn_miu_f = 0

        self.dyn_pop_archive = []
        self.current_history_best_pop = []

        self.sort_flag = False
        self.is_parallelizable = False

        self.all_layers = []
        self.layers_size = 0

    def initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = []

        self.layers_size = self.pop_size // self.n_layers

    def initialization(self) -> None:
        super().initialization()

        for i in range(0, self.n_layers):
            a, b = (
                self.layers_size * i,
                self.layers_size * (i + 1)
            )

            new_pop = self.pop[a:b]
            layer = LayersPopulation(
                pop=new_pop,
                pop_size=len(new_pop),
                minimum_pop=self.minimum_pop,
                default_pop_size=self.default_pop_size,
                ap=self.ap,
                dyn_miu_cr=self.dyn_miu_cr,
                dyn_miu_f=self.dyn_miu_f,
                n_epoch=self.epoch,
            )

            layer.mutation = self.mutation
            layer.amend_solution = self.amend_solution
            layer.generator = self.generator

            self.all_layers += [layer]

        print(self.n_layers, len(self.all_layers))

    def mutation(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)

        return self.correct_solution(pos_new)

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

    def evolve(self, epoch: int) -> None:
        for layer in self.all_layers:
            layer.evolve(epoch, self.problem.minmax, self.get_target)

    def evolve_old(self, epoch):
        list_f = []
        list_cr = []
        temp_f = []
        temp_cr = []

        self.current_history_best_pop = []
        self.g_best_history = self.g_best

        if len(self.all_global_history_best_pop) > 5:
            self.current_history_best_pop = self.get_sorted_population(
                self.all_global_history_best_pop, self.problem.minmax
            )
            self.g_best_history = self.current_history_best_pop[0].copy()

            best_pop_fitness = [p.target.fitness for p in self.all_global_history_best_pop]
            self.b_stats = wilcoxon(best_pop_fitness).pvalue

        current_pop_len = len(self.pop)
        g_best_mode_solution = self.g_best_history.solution

        if self.mode in _STATS_MODES and len(self.current_history_best_pop) > (self.default_pop_size / 3):
            g_best_mode_solution = stats_solution(
                self.mode,
                np.array([p.solution for p in self.current_history_best_pop])
            )

        self.pop = self.get_sorted_population(
            self.pop, self.problem.minmax
        )

        if self.b_stats < 0.05 and current_pop_len > self.minimum_pop and epoch > self.epoch // 3:
            self.all_global_history_wort_pop += [self.pop[-1].copy()]
            self.pop_size -= 1
            self.counter_pop += 1
            self.pop.pop(-1)
        elif current_pop_len < self.default_pop_size:
            p = self.all_global_history_wort_pop[0].copy()
            self.all_global_history_wort_pop.pop(0)
            self.pop += [p]
            self.pop_size += 1

        pos_new_solutions = []
        for idx in range(0, self.pop_size):
            cr = self.generator.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)

            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)

                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break

            temp_f += [f]
            temp_cr += [cr]

            r1_idx, r2_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            x_new = (
                (self.pop[idx].solution + f * (g_best_mode_solution - self.pop[idx].solution) + f * (
                            self.pop[r1_idx].solution - self.pop[r2_idx].solution))
            )

            pos_new = self.mutation(self.pop[idx].solution, x_new)
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

                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])

        self.dyn_miu_cr, self.dyn_miu_f = update_history(
            self.dyn_miu_cr, self.dyn_miu_f, self.ap, list_cr, list_f,
        )

        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
        self.g_best = self.pop[0].copy()
        self.all_global_history_best_pop += [self.g_best.copy()]


def stats_solution(mode, arr):
    all_modes = {
        'mode': lambda s: mode(s, keepdims=True)[0][0],
        'median': lambda s: np.median(s, axis=0),
        'mean': lambda s: mode(s, axis=0),
    }

    if f_mode := all_modes.get(mode):
        return f_mode(arr)

    raise ValueError("Invalid mode")


def lehmer_mean(list_objects):
    temp = np.sum(list_objects)
    return 0 if temp == 0 else np.sum(list_objects ** 2) / temp


def update_history(dyn_miu_cr, dyn_miu_f, ap, list_cr, list_f):
    if len(list_cr) == 0:
        dyn_miu_cr = (1 - ap) * dyn_miu_cr + ap * 0.5
    else:
        dyn_miu_cr = (1 - ap) * dyn_miu_cr + ap * np.mean(np.array(list_cr))

    if len(list_f) == 0:
        dyn_miu_f = (1 - ap) * dyn_miu_f + ap * 0.5
    else:
        dyn_miu_f = (1 - ap) * dyn_miu_f + ap * lehmer_mean(np.array(list_f))

    return dyn_miu_cr, dyn_miu_cr

def compare_target(target_x, target_y, minmax: str = "min") -> bool:
    if minmax == "min":
        return True if target_x.fitness < target_y.fitness else False
    else:
        return False if target_x.fitness < target_y.fitness else True
