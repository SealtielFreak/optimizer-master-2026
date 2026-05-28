import numpy as np
from scipy.stats import cauchy


def stats_solution(mode, arr):
    all_modes = {
        'mode': lambda s: mode(s, keepdims=True)[0][0],
        'median': lambda s: np.median(s, axis=0),
        'mean': lambda s: mode(s, axis=0),
    }

    if f_mode := all_modes.get(mode):
        return f_mode(arr)

    raise ValueError("Invalid mode")


def lehmer_mean(x):
    temp = np.sum(x)
    return 0 if temp == 0 else np.sum(x ** 2) / temp


def generate_cauchy(
        generator,
        dyn_miu_cr,
        dyn_miu_f: int | float | list,
        idx: int | None = None,
) -> tuple[float, float]:
    cr = generator.normal(dyn_miu_cr, 0.1)
    cr = np.clip(cr, 0, 1)


    while True:
        f = cauchy.rvs(dyn_miu_f, 0.1) if idx is None else cauchy.rvs(dyn_miu_f[idx], 0.1)

        if f < 0:
            continue
        elif f > 1:
            f = 1

        break

    return f, cr
