import math

import numpy as np
from mealpy.optimizer.classic import ClassicOptimizer


class OptimizerKu(ClassicOptimizer):
    def __init__(self, epoch=60, pop_size=100, pop_min=20, pop_max=150,
                 de_iters=60, pso_epochs=20, h_memory=6, p_best=0.11,
                 elite_frac=0.1, restart_patience=12, ls_budget=60,
                 success_high=0.20, success_low=0.05, grow_factor=1.2,
                 shrink_factor=0.85, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 1000000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pop_min = self.validator.check_int("pop_min", pop_min, [4, 10000])
        self.pop_max = self.validator.check_int("pop_max", pop_max, [10, 50000])
        self.de_iters = self.validator.check_int("de_iters", de_iters, [1, 1000])
        self.pso_epochs = self.validator.check_int("pso_epochs", pso_epochs, [1, 1000])
        self.h_memory = self.validator.check_int("h_memory", h_memory, [2, 50])
        self.p_best = self.validator.check_float("p_best", p_best, (0.0, 1.0))
        self.elite_frac = self.validator.check_float("elite_frac", elite_frac, (0.0, 0.5))
        self.restart_patience = self.validator.check_int("restart_patience", restart_patience, [3, 100000])
        self.ls_budget = self.validator.check_int("ls_budget", ls_budget, [0, 10000])
        self.success_high = self.validator.check_float("success_high", success_high, (0.0, 1.0))
        self.success_low = self.validator.check_float("success_low", success_low, (0.0, 1.0))
        self.grow_factor = self.validator.check_float("grow_factor", grow_factor, (1.0, 5.0))
        self.shrink_factor = self.validator.check_float("shrink_factor", shrink_factor, (0.1, 1.0))
        self.set_parameters(["epoch", "pop_size", "pop_min", "pop_max", "de_iters", "pso_epochs",
                             "h_memory", "p_best", "elite_frac", "restart_patience", "ls_budget",
                             "success_high", "success_low", "grow_factor", "shrink_factor"])
        self.sort_flag = False

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
        self.rad_factor = 0.5
        self.fit_cache = None

    def initialization(self):
        if self.pop is None:
            n = self.pop_size
            cut = np.linspace(0, 1, n + 1)
            rd = cut[:n, None] + np.random.rand(n, self.ndim) * (cut[1:, None] - cut[:n, None])
            for j in range(self.ndim):
                rd[:, j] = np.random.permutation(rd[:, j])
            positions = self.lo + rd * (self.hi - self.lo)
            self.pop = [self.generate_agent(positions[i]) for i in range(n)]

    def _build_pop_from_arrays(self, positions, fitnesses):
        agents = []

        for i in range(positions.shape[0]):
            ag = self.generate_empty_agent(positions[i])
            ag.target = self.problem.get_target(ag.solution)
            agents.append(ag)

        return agents

    def evolve(self, epoch):
        progress = (epoch - 1) / max(self.epoch - 1, 1)

        pos = np.array([a.solution for a in self.pop])
        fit = np.array([a.target.fitness for a in self.pop])

        total_trials = 0
        total_success = 0

        for _ in range(self.de_iters):
            n = pos.shape[0]

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

            same = np.all(trial == pos, axis=1)
            to_eval = ~same
            trial_fit = np.copy(fit)
            if to_eval.any():
                trials_to_eval = trial[to_eval]
                for k, t_pos in enumerate(trials_to_eval):
                    corrected = self.correct_solution(t_pos)
                    trial_fit_val = self.get_target(corrected).fitness
                    trials_to_eval[k] = corrected
                    trial_fit[np.where(to_eval)[0][k]] = trial_fit_val
                trial[to_eval] = trials_to_eval

            improved = trial_fit < fit
            total_trials += n
            total_success += int(improved.sum())

            delta = np.where(improved, fit - trial_fit, 0.0)

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

            pos = np.where(improved[:, None], trial, pos)
            fit = np.where(improved, trial_fit, fit)

        success_rate = total_success / max(total_trials, 1)
        current_n = pos.shape[0]
        if success_rate > self.success_high:
            new_n = max(self.pop_min, int(round(current_n * self.shrink_factor)))
        elif success_rate < self.success_low:
            new_n = min(self.pop_max, int(round(current_n * self.grow_factor)))
        else:
            new_n = current_n

        if new_n < current_n:
            order = np.argsort(fit)
            pos = pos[order[:new_n]]
            fit = fit[order[:new_n]]
            if self.archive.shape[0] > new_n:
                self.archive = self.archive[np.random.choice(self.archive.shape[0], new_n, replace=False)]
        elif new_n > current_n:
            extra = new_n - current_n
            best_idx = int(np.argmin(fit))
            center = pos[best_idx]
            sigma = np.std(pos, axis=0) + 1e-12
            new_pos = center + np.random.normal(0, 1, (extra, self.ndim)) * sigma
            new_pos = np.clip(new_pos, self.lo, self.hi)
            new_fit = np.array([self.get_target(self.correct_solution(p)).fitness for p in new_pos])
            pos = np.vstack([pos, new_pos])
            fit = np.concatenate([fit, new_fit])

        n = pos.shape[0]
        elite_n = max(2, int(self.elite_frac * n))
        order = np.argsort(fit)
        elite_pos = pos[order[:elite_n]]
        elite_fit = fit[order[:elite_n]]
        centroid = elite_pos[0].copy()

        radius = max(self.rad_factor * self.diag * (1 - 0.7 * progress), 1e-6 * self.diag)
        thickness = radius * 0.5
        r_min = max(radius - thickness, 0.0)

        n_explore = n - elite_n
        directions = np.random.normal(0, 1, (n_explore, self.ndim))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        u = np.random.rand(n_explore, 1)
        if radius > 0:
            ratio_d = (r_min / radius) ** self.ndim
            r = radius * (u * (1.0 - ratio_d) + ratio_d) ** (1.0 / self.ndim)
        else:
            r = np.zeros((n_explore, 1))
        explorers = directions * r + centroid
        explorers = np.clip(explorers, self.lo, self.hi)
        explorers_fit = np.array([self.get_target(self.correct_solution(p)).fitness for p in explorers])

        pso_pos = np.vstack([elite_pos, explorers])
        pso_fit = np.concatenate([elite_fit, explorers_fit])

        v_max = 0.5 * (self.hi - self.lo)
        pbest_pos = pso_pos.copy()
        pbest_fit = pso_fit.copy()

        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        pso_stag = 0
        prev_gbest = gbest_fit

        for t in range(self.pso_epochs):
            r_t = t / max(self.pso_epochs - 1, 1)
            c1 = 2.5 - 2.0 * r_t
            c2 = 0.5 + 2.0 * r_t

            speed = (c1 * np.random.rand(n, self.ndim) * (pbest_pos - pso_pos)
                     + c2 * np.random.rand(n, self.ndim) * (gbest_pos - pso_pos))
            speed = np.clip(speed, -v_max, v_max)

            new_pos = pso_pos + speed
            new_pos = np.clip(new_pos, self.lo, self.hi)

            new_fit = np.array([self.get_target(self.correct_solution(p)).fitness for p in new_pos])
            pso_pos = new_pos
            pso_fit = new_fit

            improved = pso_fit < pbest_fit
            pbest_fit = np.where(improved, pso_fit, pbest_fit)
            pbest_pos = np.where(improved[:, None], pso_pos, pbest_pos)

            gbest_idx = int(np.argmin(pbest_fit))
            if pbest_fit[gbest_idx] < gbest_fit:
                gbest_fit = float(pbest_fit[gbest_idx])
                gbest_pos = pbest_pos[gbest_idx].copy()

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
                new_jumped_fit = np.array([self.get_target(self.correct_solution(p)).fitness for p in new_jumped])
                for k, idx_w in enumerate(worst_idx):
                    pso_pos[idx_w] = new_jumped[k]
                    pso_fit[idx_w] = new_jumped_fit[k]
                pso_stag = 0

        de_order = np.argsort(fit)
        pso_order = np.argsort(pbest_fit)
        keep_de = max(elite_n, n // 2)
        pos = np.vstack([pos[de_order[:keep_de]], pbest_pos[pso_order[:n - keep_de]]])
        fit = np.concatenate([fit[de_order[:keep_de]], pbest_fit[pso_order[:n - keep_de]]])

        self.pop = self._build_pop_from_arrays(pos, fit)
        for i, ag in enumerate(self.pop):
            ag.target = self.problem.get_target(pos[i])

        best_now = float(np.min(fit))
        if best_now < self.last_best - 1e-12:
            self.stag = 0
            self.rad_factor = min(0.7, self.rad_factor * 1.05)
        else:
            self.stag += 1
            self.rad_factor *= 0.85
        self.last_best = best_now

        if self.stag >= self.restart_patience:
            order = np.argsort(fit)
            keep = max(2, n // 10)
            kept_pos = pos[order[:keep]]
            kept_fit = fit[order[:keep]]
            fresh_n = max(self.pop_min, n) - keep
            fresh_pos = np.random.uniform(self.lo, self.hi, (fresh_n, self.ndim))
            fresh_fit = np.array([self.get_target(self.correct_solution(p)).fitness for p in fresh_pos])
            pos = np.vstack([kept_pos, fresh_pos])
            fit = np.concatenate([kept_fit, fresh_fit])
            self.pop = [self.generate_empty_agent(pos[i]) for i in range(pos.shape[0])]
            for i in range(pos.shape[0]):
                self.pop[i].target = self.problem.get_target(pos[i])
            self.archive = np.empty((0, self.ndim))
            self.stag = 0
            self.rad_factor = 0.5

        if epoch == self.epoch and self.ls_budget > 0:
            best_x = np.array(self.g_best.solution, dtype=float).copy()
            best_s = float(self.g_best.target.fitness)
            step_axial = 0.05 * (self.hi - self.lo)
            no_imp = 0
            for _ in range(self.ls_budget):
                imp = False
                for j in np.random.permutation(self.ndim):
                    for sign in (-1.0, 1.0):
                        trial = best_x.copy()
                        trial[j] = np.clip(best_x[j] + sign * step_axial[j], self.lo[j], self.hi[j])
                        ts = self.get_target(self.correct_solution(trial)).fitness
                        if ts < best_s:
                            best_s = ts
                            best_x = trial
                            new_ag = self.generate_empty_agent(best_x)
                            new_ag.target = self.problem.get_target(best_x)
                            worst_idx = int(np.argmax([a.target.fitness for a in self.pop]))
                            self.pop[worst_idx] = new_ag
                            imp = True
                            break
                    if imp:
                        break
                step_axial *= 1.2 if imp else 0.5
                no_imp = 0 if imp else no_imp + 1
                if no_imp >= 3 and np.all(step_axial < 1e-12 * (self.hi - self.lo + 1e-30)):
                    break
