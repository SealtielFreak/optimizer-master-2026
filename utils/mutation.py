import numpy as np

from utils import lehmer_mean


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
