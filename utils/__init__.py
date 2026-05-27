import numpy as np


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
