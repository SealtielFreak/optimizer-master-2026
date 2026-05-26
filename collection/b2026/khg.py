import math

import numpy as np
from mealpy.optimizer.classic import ClassicOptimizer


class OptimizerKHG(ClassicOptimizer):
    """
    Optimizer K
    """

    def __init__(self, epoch=60, pop_size=100, pop_min=20,
                 de_iters=60, pso_epochs=20, h_memory=6, p_best=0.11,
                 elite_frac=0.1, restart_patience=12, ls_budget=60, **kwargs):
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 1000000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pop_min = self.validator.check_int("pop_min", pop_min, [4, 10000])
        self.de_iters = self.validator.check_int("de_iters", de_iters, [1, 1000])
        self.pso_epochs = self.validator.check_int("pso_epochs", pso_epochs, [1, 1000])
        self.h_memory = self.validator.check_int("h_memory", h_memory, [2, 50])
        self.p_best = self.validator.check_float("p_best", p_best, (0.0, 1.0))
        self.elite_frac = self.validator.check_float("elite_frac", elite_frac, (0.0, 0.5))
        self.restart_patience = self.validator.check_int("restart_patience", restart_patience, [3, 100000])
        self.ls_budget = self.validator.check_int("ls_budget", ls_budget, [0, 10000])

        self.set_parameters(["epoch", "pop_size", "pop_min", "de_iters", "pso_epochs",
                             "h_memory", "p_best", "elite_frac", "restart_patience", "ls_budget"])
        self.sort_flag = False

    def _hypersphere_sample(self, n, radius, thickness, centroid):
        directions = np.random.normal(0, 1, (n, self.ndim))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

        r_min = max(radius - thickness, 0.0)

        u = np.random.rand(n, 1)
        r = (u * (radius ** self.ndim - r_min ** self.ndim) + r_min ** self.ndim) ** (1.0 / self.ndim)

        return directions * r + centroid

    def _auto_radius(self, positions, centroid, progress=0.0):
        if positions.shape[0] < 2:
            return 0.5 * self.diag, 0.16 * self.diag

        sigma = float(np.mean(np.std(positions, axis=0)))

        d_from_center = np.linalg.norm(positions - centroid, axis=1)
        d_avg = float(np.mean(d_from_center))

        radius = max(sigma * math.sqrt(self.ndim), d_avg)
        floor = (0.5 - 0.49 * progress) * self.diag

        radius = max(radius, floor)
        radius = min(radius, self.diag)

        thickness = radius / 3.0

        return radius, thickness

    def initialize_variables(self):
        self.lo = self.problem.lb
        self.hi = self.problem.ub
        self.ndim = self.problem.n_dims
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        self.mF = np.full(self.h_memory, 0.5)
        self.mCR = np.full(self.h_memory, 0.5)
        self.mem_idx = 0
        self.archive = np.empty((0, self.ndim))
        self.stag = 0
        self.last_best = np.inf

    def initialization(self):
        if self.pop is None:
            n = self.pop_size
            domain_center = 0.5 * (self.lo + self.hi)

            init_radius = 0.5 * self.diag
            init_thickness = init_radius / 3.0

            positions = self._hypersphere_sample(n, init_radius, init_thickness, domain_center)
            positions = np.clip(positions, self.lo, self.hi)

            self.pop = [self.generate_agent(positions[i]) for i in range(n)]

    def evolve(self, epoch):
        progress = (epoch - 1) / max(self.epoch - 1, 1)
        n_target = max(self.pop_min, int(round(self.pop_size - (self.pop_size - self.pop_min) * progress)))

        if len(self.pop) > n_target:
            self.pop = self.get_sorted_population(self.pop, self.problem.minmax)[:n_target]

            if self.archive.shape[0] > n_target:
                self.archive = self.archive[np.random.choice(self.archive.shape[0], n_target, replace=False)]

        for _ in range(self.de_iters):
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

            new_agents = [self.generate_agent(self.correct_solution(trial[i])) for i in range(n)]

            improved = np.zeros(n, dtype=bool)
            delta = np.zeros(n)

            for i in range(n):
                if self.compare_target(new_agents[i].target, self.pop[i].target, self.problem.minmax):
                    improved[i] = True
                    delta[i] = abs(self.pop[i].target.fitness - new_agents[i].target.fitness)

            if improved.any():
                self.archive = np.vstack([self.archive, pos[improved]]) if self.archive.size else pos[improved].copy()
                if self.archive.shape[0] > n:
                    self.archive = self.archive[np.random.choice(self.archive.shape[0], n, replace=False)]

                if delta.sum() > 0:
                    w = delta[improved] / delta[improved].sum()
                    sF, sCR = F[improved], CR[improved]
                    denom_f = np.sum(w * sF)

                    if denom_f > 1e-30:
                        self.mF[self.mem_idx] = float(np.sum(w * sF * sF) / denom_f)
                        self.mCR[self.mem_idx] = float(np.sum(w * sCR * sCR) / max(np.sum(w * sCR), 1e-30))
                        self.mem_idx = (self.mem_idx + 1) % self.h_memory

            self.pop = [new_agents[i] if improved[i] else self.pop[i] for i in range(n)]

        n = len(self.pop)
        elite_n = max(2, int(self.elite_frac * n))
        sorted_pop = self.get_sorted_population(self.pop, self.problem.minmax)
        elite = sorted_pop[:elite_n]
        centroid = np.array(sorted_pop[0].solution)

        current_positions = np.array([a.solution for a in self.pop])
        radius, thickness = self._auto_radius(current_positions, centroid, progress)

        n_explore = n - elite_n
        explorers = self._hypersphere_sample(n_explore, radius, thickness, centroid)
        explorers = np.clip(explorers, self.lo, self.hi)

        pso_pop = list(elite) + [self.generate_agent(self.correct_solution(explorers[i])) for i in range(n_explore)]
        pso_pos = np.array([a.solution for a in pso_pop])
        pso_fit = np.array([a.target.fitness for a in pso_pop])

        v_max = 0.5 * (self.hi - self.lo)
        speed = np.random.uniform(-1, 1, (n, self.ndim)) * v_max
        pbest_pos = pso_pos.copy()
        pbest_fit = pso_fit.copy()
        pbest_agents = list(pso_pop)

        gbest_agent = self.get_best_agent(pso_pop, self.problem.minmax)
        gbest_pos = np.array(gbest_agent.solution)
        gbest_fit = gbest_agent.target.fitness

        pso_stag = 0
        prev_gbest = gbest_fit

        for t in range(self.pso_epochs):
            r = t / max(self.pso_epochs - 1, 1)
            c1 = 2.5 - 2.0 * r
            c2 = 0.5 + 2.0 * r

            speed = (c1 * np.random.rand(n, self.ndim) * (pbest_pos - pso_pos)
                     + c2 * np.random.rand(n, self.ndim) * (gbest_pos - pso_pos))
            speed = np.clip(speed, -v_max, v_max)

            new_pos = pso_pos + speed
            out = (new_pos < self.lo) | (new_pos > self.hi)
            new_pos = np.clip(new_pos, self.lo, self.hi)
            speed = np.where(out, -0.5 * speed, speed)

            new_agents = [self.generate_agent(self.correct_solution(new_pos[i])) for i in range(n)]
            pso_pos = new_pos
            pso_fit = np.array([a.target.fitness for a in new_agents])

            for i in range(n):
                if self.compare_target(new_agents[i].target, pbest_agents[i].target, self.problem.minmax):
                    pbest_agents[i] = new_agents[i]
                    pbest_pos[i] = new_pos[i]
                    pbest_fit[i] = pso_fit[i]
                    if self.compare_target(new_agents[i].target, gbest_agent.target, self.problem.minmax):
                        gbest_agent = new_agents[i]
                        gbest_pos = new_pos[i].copy()
                        gbest_fit = pso_fit[i]

            if abs(prev_gbest - gbest_fit) < 1e-12:
                pso_stag += 1
            else:
                pso_stag = 0

            prev_gbest = gbest_fit

            if pso_stag >= max(3, self.pso_epochs // 5):
                worst_n = max(1, n // 4)
                worst_idx = np.argsort(pbest_fit)[-worst_n:]

                beta = 1.5
                sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
                           / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

                u_lev = np.random.normal(0, sigma_u, (worst_n, self.ndim))
                v_lev = np.random.normal(0, 1, (worst_n, self.ndim))

                step = 0.1 * (self.hi - self.lo) * u_lev / (np.abs(v_lev) ** (1 / beta) + 1e-12)

                new_jumped = np.clip(gbest_pos + step, self.lo, self.hi)

                for k, idx_w in enumerate(worst_idx):
                    pso_pos[idx_w] = new_jumped[k]
                    speed[idx_w] = np.random.uniform(-1, 1, self.ndim) * v_max

                pso_stag = 0

        de_order = np.argsort([a.target.fitness for a in self.pop])
        pso_order = np.argsort(pbest_fit)
        keep_de = max(elite_n, n // 2)

        self.pop = ([self.pop[i] for i in de_order[:keep_de]]
                    + [pbest_agents[i] for i in pso_order[:n - keep_de]])

        best_now = self.get_best_agent(self.pop, self.problem.minmax).target.fitness

        if best_now < self.last_best - 1e-12:
            self.stag = 0
        else:
            self.stag += 1

        self.last_best = best_now

        if self.stag >= self.restart_patience:
            sorted_pop = self.get_sorted_population(self.pop, self.problem.minmax)
            keep = max(2, n // 10)

            elite_seeds = sorted_pop[:keep]
            best_pos = np.array(elite_seeds[0].solution)

            current_positions = np.array([a.solution for a in self.pop])
            radius, thickness = self._auto_radius(current_positions, best_pos, progress=0.0)
            thickness = radius / 3.0

            fresh_pos = self._hypersphere_sample(n - keep, radius, thickness, best_pos)
            fresh_pos = np.clip(fresh_pos, self.lo, self.hi)

            fresh = [self.generate_agent(self.correct_solution(fresh_pos[i])) for i in range(n - keep)]

            self.pop = list(elite_seeds) + fresh
            self.archive = np.empty((0, self.ndim))
            self.stag = 0

        if epoch == self.epoch and self.ls_budget > 0:
            best_x = np.array(self.g_best.solution, dtype=float).copy()
            best_agent = self.g_best
            step = 0.05 * (self.hi - self.lo)
            no_imp = 0

            for _ in range(self.ls_budget):
                imp = False

                for j in np.random.permutation(self.ndim):
                    for sign in (-1.0, 1.0):
                        trial = best_x.copy()
                        trial[j] = np.clip(best_x[j] + sign * step[j], self.lo[j], self.hi[j])

                        cand = self.generate_agent(self.correct_solution(trial))

                        if self.compare_target(cand.target, best_agent.target, self.problem.minmax):
                            best_x = trial
                            best_agent = cand
                            worst_idx = int(np.argmax([a.target.fitness for a in self.pop]))

                            self.pop[worst_idx] = cand
                            imp = True

                            break
                    if imp:
                        break

                step *= 1.2 if imp else 0.5
                no_imp = 0 if imp else no_imp + 1

                if no_imp >= 3 and np.all(step < 1e-12 * (self.hi - self.lo + 1e-30)):
                    break
