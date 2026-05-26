import numpy as np

from mealpy.optimizer.classic import ClassicOptimizer


class OptimizerK(ClassicOptimizer):
    """
    Optimizer K
    """

    def __init__(self, epoch=2000, pop_size=100, h_memory=6, p_best=0.11,
                 elite_frac=0.1, restart_patience=200, ls_budget=60, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 1000000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.h_memory = self.validator.check_int("h_memory", h_memory, [2, 50])
        self.p_best = self.validator.check_float("p_best", p_best, (0.0, 1.0))
        self.elite_frac = self.validator.check_float("elite_frac", elite_frac, (0.0, 0.5))
        self.restart_patience = self.validator.check_int("restart_patience", restart_patience, [10, 100000])
        self.ls_budget = self.validator.check_int("ls_budget", ls_budget, [0, 10000])
        self.set_parameters(["epoch", "pop_size", "h_memory", "p_best",
                             "elite_frac", "restart_patience", "ls_budget"])
        self.sort_flag = False

    def initialize_variables(self):
        self.lo = self.problem.lb
        self.hi = self.problem.ub
        self.ndim = self.problem.n_dims
        self.pop_min = max(8, self.pop_size // 5)
        self.mF = np.full(self.h_memory, 0.5)
        self.mCR = np.full(self.h_memory, 0.5)
        self.mem_idx = 0
        self.archive = np.empty((0, self.ndim))
        self.stag = 0
        self.last_best = np.inf

    def initialization(self):
        if self.pop is None:
            n = self.pop_size
            cut = np.linspace(0, 1, n + 1)
            rd = cut[:n, None] + np.random.rand(n, self.ndim) * (cut[1:, None] - cut[:n, None])

            for j in range(self.ndim):
                rd[:, j] = np.random.permutation(rd[:, j])

            positions = self.lo + rd * (self.hi - self.lo)

            self.pop = [self.generate_agent(positions[i]) for i in range(n)]

    def evolve(self, epoch):
        progress = (epoch - 1) / max(self.epoch - 1, 1)
        n_target = max(self.pop_min, int(round(self.pop_size - (self.pop_size - self.pop_min) * progress)))
        if len(self.pop) > n_target:
            self.pop = self.get_sorted_population(self.pop, self.problem.minmax)[:n_target]

        n = len(self.pop)
        pos = np.array([a.solution for a in self.pop])
        fit = np.array([a.target.fitness for a in self.pop])

        idx = np.random.randint(0, self.h_memory, n)
        F = self.mF[idx] + 0.1 * np.tan(np.pi * (np.random.rand(n) - 0.5))

        for _ in range(8):
            bad = F <= 0

            if not bad.any():
                break

            F[bad] = self.mF[idx[bad]] + 0.1 * np.tan(np.pi * (np.random.rand(bad.sum()) - 0.5))

        F = np.clip(F, 1e-6, 1.0)
        CR = np.clip(np.random.normal(self.mCR[idx], 0.1), 0, 1)

        order = np.argsort(fit)
        p_top = max(2, int(round(self.p_best * n)))
        pbest = order[:p_top][np.random.randint(0, p_top, n)]
        union = pos if self.archive.shape[0] == 0 else np.vstack([pos, self.archive])

        r1 = np.random.randint(0, n, n)
        r2 = np.random.randint(0, union.shape[0], n)

        for _ in range(5):
            bad = r1 == np.arange(n)

            if bad.any():
                r1[bad] = np.random.randint(0, n, bad.sum())

            bad = (r2 == np.arange(n)) | (r2 == r1)

            if not bad.any():
                break

            r2[bad] = np.random.randint(0, union.shape[0], bad.sum())

        mutant = pos + F[:, None] * (pos[pbest] - pos + pos[r1] - union[r2])
        mutant = np.where(mutant < self.lo, (pos + self.lo) / 2, mutant)
        mutant = np.where(mutant > self.hi, (pos + self.hi) / 2, mutant)
        mask = np.random.rand(n, self.ndim) < CR[:, None]
        mask[np.arange(n), np.random.randint(0, self.ndim, n)] = True
        trial = np.where(mask, mutant, pos)

        new_pop = [self.generate_agent(self.correct_solution(trial[i])) for i in range(n)]

        improved = np.zeros(n, dtype=bool)
        delta = np.zeros(n)

        for i in range(n):
            if self.compare_target(new_pop[i].target, self.pop[i].target, self.problem.minmax):
                improved[i] = True
                delta[i] = abs(self.pop[i].target.fitness - new_pop[i].target.fitness)

        if improved.any():
            self.archive = np.vstack([self.archive, pos[improved]]) if self.archive.size else pos[improved].copy()

            if self.archive.shape[0] > n:
                self.archive = self.archive[np.random.choice(self.archive.shape[0], n, replace=False)]

            if delta.sum() > 0:
                w = delta[improved] / delta[improved].sum()
                sF, sCR = F[improved], CR[improved]
                denom = np.sum(w * sF)

                if denom > 1e-30:
                    self.mF[self.mem_idx] = float(np.sum(w * sF * sF) / denom)
                    self.mCR[self.mem_idx] = float(np.sum(w * sCR * sCR) / max(np.sum(w * sCR), 1e-30))
                    self.mem_idx = (self.mem_idx + 1) % self.h_memory

        self.pop = [new_pop[i] if improved[i] else self.pop[i] for i in range(n)]

        best_fit = min(a.target.fitness for a in self.pop)

        if abs(best_fit - self.last_best) < 1e-12:
            self.stag += 1
        else:
            self.stag = 0

        self.last_best = best_fit

        if self.stag >= self.restart_patience:
            sorted_pop = self.get_sorted_population(self.pop, self.problem.minmax)
            keep = max(2, n // 10)
            elite = sorted_pop[:keep]
            fresh = [self.generate_agent() for _ in range(n - keep)]

            self.pop = elite + fresh
            self.archive = np.empty((0, self.ndim))
            self.stag = 0

    def track_optimize_process(self):
        if self.ls_budget > 0:
            best_x = self.g_best.solution.copy()
            step = 0.05 * (self.hi - self.lo)
            no_imp = 0

            for _ in range(self.ls_budget):
                imp = False

                for j in np.random.permutation(self.ndim):
                    for sign in (-1.0, 1.0):
                        t = best_x.copy()
                        t[j] = np.clip(best_x[j] + sign * step[j], self.lo[j], self.hi[j])
                        agent = self.generate_agent(self.correct_solution(t))

                        if self.compare_target(agent.target, self.g_best.target, self.problem.minmax):
                            best_x = t
                            self.g_best = agent
                            imp = True

                            break

                step *= 1.2 if imp else 0.5
                no_imp = 0 if imp else no_imp + 1

                if no_imp >= 3 and np.all(step < 1e-12 * (self.hi - self.lo + 1e-30)):
                    break

        super().track_optimize_process()
