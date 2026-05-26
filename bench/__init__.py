import multiprocessing as mp

import dataclasses
import itertools
import time
import typing

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mealpy import FloatVar
from scipy.stats import wilcoxon


_NAN_PENALTY = 1e18
_ALPHA = 0.05

C = typing.TypeVar("C")


@dataclasses.dataclass
class FunctionSpec:
    name: str
    func: typing.Callable
    bounds: tuple[float, float]
    ndim: int
    f_global: float


class ClassFunctionSpec(typing.Generic[C]):
    def __init__(self, cls: C, ndim: int) -> None:
        self.__obj_f = cls()

        self.name: str = self.__obj_f.name
        self.func: typing.Callable = self.__obj_f.evaluate
        self.bounds: tuple[float, float] = self.__obj_f.lb[0], self.__obj_f.ub[0]
        self.ndim: int = ndim
        self.f_global: float = self.__obj_f.f_global


@dataclasses.dataclass
class AlgorithmSpec:
    name: str
    cls: type
    kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RunResult:
    function: str
    ndim: int
    algorithm: str
    f_global: float
    run_fitnesses: list[float]
    run_times: list[float]
    mean_diversity: float
    mean_exploration_pct: float
    mean_exploitation_pct: float
    final_diversity: float
    nan_penalty_used: bool


def _tracked_func_factory(func: Callable) -> tuple[Callable, list]:
    nan_hits = [0]

    def tracked(x: np.ndarray) -> float:
        try:
            raw = func(x)
        except Exception:
            nan_hits[0] += 1
            return _NAN_PENALTY
        try:
            val = float(raw)
        except (TypeError, ValueError):
            nan_hits[0] += 1
            return _NAN_PENALTY
        if np.isnan(val) or np.isinf(val):
            nan_hits[0] += 1
            return _NAN_PENALTY
        return val

    return tracked, nan_hits


def _run_single(task: tuple) -> RunResult:
    func_spec, alg_spec, epoch, pop_size, n_runs = task

    run_fitnesses = []
    run_times = []
    diversities = []
    explorations = []
    exploitations = []
    final_divs = []

    tracked_func, nan_hits = _tracked_func_factory(func_spec.func)

    problem_def = {
        "obj_func": tracked_func,
        "bounds": FloatVar(
            lb=(func_spec.bounds[0],) * func_spec.ndim,
            ub=(func_spec.bounds[1],) * func_spec.ndim,
        ),
        "minmax": "min",
        "log_to": None,
    }

    for _ in range(n_runs):
        model = alg_spec.cls(epoch=epoch, pop_size=pop_size, **alg_spec.kwargs)
        t0 = time.perf_counter()
        model.solve(problem_def)
        elapsed = time.perf_counter() - t0

        raw_fitness = model.g_best.target.fitness
        fitness = float(raw_fitness) if not np.isnan(raw_fitness) else np.nan
        run_fitnesses.append(fitness)
        run_times.append(elapsed)

        hist = model.history

        raw_div = hist.diversity
        clean_div = [d for d in raw_div if not np.isnan(d)]
        diversities.extend(clean_div if clean_div else [np.nan])

        raw_exp = hist.exploration
        raw_exp = raw_exp.tolist() if hasattr(raw_exp, "tolist") else list(raw_exp)
        clean_exp = [v for v in raw_exp if not np.isnan(v)]
        explorations.extend(clean_exp if clean_exp else [np.nan])

        raw_oit = hist.exploitation
        raw_oit = raw_oit.tolist() if hasattr(raw_oit, "tolist") else list(raw_oit)
        clean_oit = [v for v in raw_oit if not np.isnan(v)]
        exploitations.extend(clean_oit if clean_oit else [np.nan])

        last_div = raw_div[-1] if raw_div else np.nan
        final_divs.append(float(last_div) if not np.isnan(last_div) else np.nan)

    fitnesses_arr = np.array(run_fitnesses, dtype=float)
    fitnesses_arr[fitnesses_arr >= _NAN_PENALTY] = np.nan

    return RunResult(
        function=func_spec.name,
        ndim=func_spec.ndim,
        algorithm=alg_spec.name,
        f_global=func_spec.f_global,
        run_fitnesses=fitnesses_arr.tolist(),
        run_times=run_times,
        mean_diversity=float(np.nanmean(diversities)) if any(not np.isnan(d) for d in diversities) else np.nan,
        mean_exploration_pct=float(np.nanmean(explorations)) if any(not np.isnan(v) for v in explorations) else np.nan,
        mean_exploitation_pct=float(np.nanmean(exploitations)) if any(
            not np.isnan(v) for v in exploitations) else np.nan,
        final_diversity=float(np.nanmean(final_divs)) if any(not np.isnan(d) for d in final_divs) else np.nan,
        nan_penalty_used=nan_hits[0] > 0,
    )


def _wilcoxon_pairwise(
        results: list[RunResult],
        alpha: float = _ALPHA,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test (paired, two-sided) comparing every pair of
    algorithms across all functions simultaneously.

    Data used: per-run errors  |fitness - f_global|  for each (function, algorithm).
    Pairing: errors from the same run index on the same function are treated as
    paired observations (both algorithms faced the same problem instance).

    Result symbols per pair (A vs B):
        +  A is significantly better than B  (p < alpha, median_A < median_B)
        -  A is significantly worse than B   (p < alpha, median_A > median_B)
        =  No significant difference          (p >= alpha)
        ?  Test could not be computed         (all-NaN, constant differences, etc.)
    """
    alg_names = sorted({r.algorithm for r in results})
    func_names = sorted({r.function for r in results})

    errors_index: dict[tuple[str, str], np.ndarray] = {}
    for r in results:
        errs = np.abs(np.array(r.run_fitnesses, dtype=float) - r.f_global)
        errors_index[(r.function, r.algorithm)] = errs

    rows = []
    for alg_a, alg_b in itertools.combinations(alg_names, 2):
        paired_a: list[float] = []
        paired_b: list[float] = []

        for fn in func_names:
            errs_a = errors_index.get((fn, alg_a), np.array([np.nan]))
            errs_b = errors_index.get((fn, alg_b), np.array([np.nan]))

            n = min(len(errs_a), len(errs_b))
            for i in range(n):
                va, vb = errs_a[i], errs_b[i]
                if not (np.isnan(va) or np.isnan(vb)):
                    paired_a.append(va)
                    paired_b.append(vb)

        pa = np.array(paired_a)
        pb = np.array(paired_b)
        diffs = pa - pb

        stat, p_value, symbol = _run_wilcoxon(pa, pb, diffs, alpha)

        rows.append({
            "algorithm_A": alg_a,
            "algorithm_B": alg_b,
            "n_pairs": len(paired_a),
            "median_error_A": float(np.median(pa)) if pa.size else np.nan,
            "median_error_B": float(np.median(pb)) if pb.size else np.nan,
            "statistic": stat,
            "p_value": p_value,
            "alpha": alpha,
            "result": symbol,
            "interpretation": _interpret(symbol, alg_a, alg_b),
        })

    return pd.DataFrame(rows)


def _wilcoxon_per_function(
        results: list[RunResult],
        alpha: float = _ALPHA,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test for every (function, algorithm_A, algorithm_B) triple.
    Paired observations are the per-run errors from each independent run.
    Requires n_runs >= 2 to produce meaningful p-values.
    """
    alg_names = sorted({r.algorithm for r in results})
    func_names = sorted({r.function for r in results})

    errors_index: dict[tuple[str, str], np.ndarray] = {}
    for r in results:
        errs = np.abs(np.array(r.run_fitnesses, dtype=float) - r.f_global)
        errors_index[(r.function, r.algorithm)] = errs

    rows = []
    for fn in func_names:
        for alg_a, alg_b in itertools.combinations(alg_names, 2):
            errs_a = errors_index.get((fn, alg_a), np.array([np.nan]))
            errs_b = errors_index.get((fn, alg_b), np.array([np.nan]))

            n = min(len(errs_a), len(errs_b))
            mask = ~(np.isnan(errs_a[:n]) | np.isnan(errs_b[:n]))
            pa = errs_a[:n][mask]
            pb = errs_b[:n][mask]
            diffs = pa - pb

            stat, p_value, symbol = _run_wilcoxon(pa, pb, diffs, alpha)

            rows.append({
                "function": fn,
                "algorithm_A": alg_a,
                "algorithm_B": alg_b,
                "n_pairs": int(mask.sum()),
                "median_error_A": float(np.median(pa)) if pa.size else np.nan,
                "median_error_B": float(np.median(pb)) if pb.size else np.nan,
                "statistic": stat,
                "p_value": p_value,
                "alpha": alpha,
                "result": symbol,
                "interpretation": _interpret(symbol, alg_a, alg_b),
            })

    return pd.DataFrame(rows)


def _run_wilcoxon(
        pa: np.ndarray,
        pb: np.ndarray,
        diffs: np.ndarray,
        alpha: float,
) -> tuple[float | None, float | None, str]:
    if pa.size < 2:
        return None, None, "?"
    if np.all(diffs == 0):
        return 0.0, 1.0, "="
    try:
        res = wilcoxon(pa, pb, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic)
        p_value = float(res.pvalue)
        if p_value < alpha:
            symbol = "+" if np.median(pa) < np.median(pb) else "-"
        else:
            symbol = "="
        return stat, p_value, symbol
    except Exception:
        return None, None, "?"


def _interpret(symbol: str, alg_a: str, alg_b: str) -> str:
    if symbol == "+":
        return f"{alg_a} significantly better than {alg_b}"
    if symbol == "-":
        return f"{alg_a} significantly worse than {alg_b}"
    if symbol == "=":
        return "No significant difference"
    return "Test not applicable"


def _wilcoxon_scoreboard(wilcoxon_per_fn: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate wins (+), ties (=), losses (-), and undecided (?) per algorithm
    across all functions and pairwise comparisons.
    """
    alg_names = sorted(
        set(wilcoxon_per_fn["algorithm_A"]) | set(wilcoxon_per_fn["algorithm_B"])
    )
    records: dict[str, dict] = {
        a: {"algorithm": a, "wins": 0, "ties": 0, "losses": 0, "undecided": 0}
        for a in alg_names
    }

    for _, row in wilcoxon_per_fn.iterrows():
        a, b, sym = row["algorithm_A"], row["algorithm_B"], row["result"]
        if sym == "+":
            records[a]["wins"] += 1
            records[b]["losses"] += 1
        elif sym == "-":
            records[a]["losses"] += 1
            records[b]["wins"] += 1
        elif sym == "=":
            records[a]["ties"] += 1
            records[b]["ties"] += 1
        else:
            records[a]["undecided"] += 1
            records[b]["undecided"] += 1

    df = pd.DataFrame(records.values())
    df = df.sort_values("wins", ascending=False).reset_index(drop=True)
    return df


class BenchmarkSuite:
    def __init__(
            self,
            functions: list[FunctionSpec],
            algorithms: list[AlgorithmSpec],
            epoch: int = 100,
            pop_size: int = 50,
            n_runs: int = 5,
            n_workers: int | None = None,
            alpha: float = _ALPHA,
    ):
        self.functions = functions
        self.algorithms = algorithms
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_runs = n_runs
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.alpha = alpha

        self._results: list[RunResult] = []
        self._summary_df: pd.DataFrame | None = None
        self._metrics_df: pd.DataFrame | None = None
        self._winners_df: pd.DataFrame | None = None
        self._wilcoxon_pairwise_df: pd.DataFrame | None = None
        self._wilcoxon_per_fn_df: pd.DataFrame | None = None
        self._wilcoxon_scoreboard_df: pd.DataFrame | None = None

    def run(self) -> None:
        tasks = [
            (fs, alg, self.epoch, self.pop_size, self.n_runs)
            for fs in self.functions
            for alg in self.algorithms
        ]

        if self.n_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.n_workers) as pool:
                self._results = pool.map(_run_single, tasks)
        else:
            self._results = [_run_single(t) for t in tasks]

        self._build_dataframes()

    def _build_dataframes(self) -> None:
        summary_rows = []
        metrics_rows = []

        for r in self._results:
            fitnesses = np.array(r.run_fitnesses, dtype=float)
            valid = fitnesses[~np.isnan(fitnesses)]
            errors = np.abs(fitnesses - r.f_global)

            summary_rows.append({
                "function": r.function,
                "ndim": r.ndim,
                "algorithm": r.algorithm,
                "f_global": r.f_global,
                "best": float(np.nanmin(fitnesses)) if valid.size else np.nan,
                "mean": float(np.nanmean(fitnesses)) if valid.size else np.nan,
                "median": float(np.nanmedian(fitnesses)) if valid.size else np.nan,
                "std": float(np.nanstd(fitnesses)) if valid.size else np.nan,
                "worst": float(np.nanmax(fitnesses)) if valid.size else np.nan,
                "mean_error": float(np.nanmean(errors)) if valid.size else np.nan,
                "mean_time_s": float(np.mean(r.run_times)),
                "nan_penalty_used": r.nan_penalty_used,
            })

            metrics_rows.append({
                "function": r.function,
                "algorithm": r.algorithm,
                "mean_diversity": r.mean_diversity,
                "mean_exploration_pct": r.mean_exploration_pct,
                "mean_exploitation_pct": r.mean_exploitation_pct,
                "final_diversity": r.final_diversity,
            })

        self._summary_df = pd.DataFrame(summary_rows)
        self._metrics_df = pd.DataFrame(metrics_rows)

        def _safe_idxmin(s):
            s_valid = s.dropna()
            return s_valid.idxmin() if not s_valid.empty else None

        winner_idx = self._summary_df.groupby("function")["mean_error"].apply(_safe_idxmin)
        valid_winners = winner_idx.dropna()
        winners = self._summary_df.loc[valid_winners, ["function", "algorithm", "mean_error"]].copy()
        winners = winners.rename(columns={"algorithm": "winner_algorithm", "mean_error": "best_mean_error"})
        winners = winners.reset_index(drop=True)
        self._winners_df = winners

        self._wilcoxon_per_fn_df = _wilcoxon_per_function(self._results, self.alpha)
        self._wilcoxon_pairwise_df = _wilcoxon_pairwise(self._results, self.alpha)
        self._wilcoxon_scoreboard_df = _wilcoxon_scoreboard(self._wilcoxon_per_fn_df)

    @property
    def summary(self) -> pd.DataFrame:
        return self._summary_df

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics_df

    @property
    def winners(self) -> pd.DataFrame:
        return self._winners_df

    @property
    def wilcoxon_pairwise(self) -> pd.DataFrame:
        return self._wilcoxon_pairwise_df

    @property
    def wilcoxon_per_function(self) -> pd.DataFrame:
        return self._wilcoxon_per_fn_df

    @property
    def wilcoxon_scoreboard(self) -> pd.DataFrame:
        return self._wilcoxon_scoreboard_df

    def print_results(self) -> None:
        _section("SUMMARY")
        print(self._summary_df.to_string(index=False))

        _section("EXPLORATION / EXPLOITATION METRICS")
        print(self._metrics_df.to_string(index=False))

        _section("WINNERS (lowest mean error per function)")
        print(self._winners_df.to_string(index=False))

        _section(f"WILCOXON SIGNED-RANK — PER FUNCTION  (alpha={self.alpha})")
        _print_wilcoxon_legend()
        print(self._wilcoxon_per_fn_df.to_string(index=False))

        _section(f"WILCOXON SIGNED-RANK — GLOBAL PAIRWISE  (alpha={self.alpha})")
        _print_wilcoxon_legend()
        print(self._wilcoxon_pairwise_df.to_string(index=False))

        _section("WILCOXON SCOREBOARD  (wins / ties / losses across all functions)")
        print(self._wilcoxon_scoreboard_df.to_string(index=False))

    def save(self, output_dir: str | Path = ".") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        with pd.ExcelWriter(output_dir / f"benchmark-{ts}.xlsx") as writer:
            self._summary_df.to_excel(writer, sheet_name="summary", index=False)
            self._metrics_df.to_excel(writer, sheet_name="metrics", index=False)
            self._winners_df.to_excel(writer, sheet_name="winners", index=False)
            self._wilcoxon_per_fn_df.to_excel(writer, sheet_name="wilcoxon_per_function", index=False)
            self._wilcoxon_pairwise_df.to_excel(writer, sheet_name="wilcoxon_pairwise", index=False)
            self._wilcoxon_scoreboard_df.to_excel(writer, sheet_name="wilcoxon_scoreboard", index=False)

        for name, df in [
            ("benchmark", self._summary_df),
            ("benchmark-metrics", self._metrics_df),
            ("benchmark-winners", self._winners_df),
            ("benchmark-wilcoxon-per-fn", self._wilcoxon_per_fn_df),
            ("benchmark-wilcoxon-pairwise", self._wilcoxon_pairwise_df),
            ("benchmark-wilcoxon-scoreboard", self._wilcoxon_scoreboard_df),
        ]:
            df.to_parquet(output_dir / f"{name}-{ts}.parquet", index=False)

        xlsx_path = output_dir / f"benchmark-{ts}.xlsx"
        print(f"\nResults saved to: {output_dir.resolve()}")
        print(f"  all sheets -> benchmark-{ts}.xlsx")
        print(f"  individual parquet files with prefix benchmark-*-{ts}.parquet")

        return xlsx_path


def _print_wilcoxon_legend() -> None:
    print(
        "  Legend:  + = A significantly better  |  - = A significantly worse  |  = = no difference  |  ? = not applicable")


def _section(title: str) -> None:
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
