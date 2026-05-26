import opfunu

_ALL_DEFAULT_PROBLEMS = (
    opfunu.name_based.Ackley01,
    opfunu.name_based.Alpine01,
    opfunu.name_based.Alpine02,
    opfunu.name_based.ChungReynolds,
    opfunu.name_based.Cigar,
    opfunu.name_based.Csendes,
    opfunu.name_based.Deceptive,
    opfunu.name_based.DixonPrice,
    opfunu.name_based.EggHolder,
    opfunu.name_based.Exponential,
    opfunu.name_based.Griewank,
    opfunu.name_based.Infinity,
    opfunu.name_based.Levy03,
    opfunu.name_based.Mishra01,
    opfunu.name_based.Mishra02,
    opfunu.name_based.Mishra11,
    opfunu.name_based.MultiModal,
    opfunu.name_based.NeedleEye,
    opfunu.name_based.Parsopoulos,
    # opfunu.name_based.Qing,

    opfunu.name_based.Quintic,
    opfunu.name_based.Salomon,
    opfunu.name_based.XinSheYang01,
    opfunu.name_based.YaoLiu04,

    # opfunu.name_based.Zacharov
)


def get_all_default_problems():
    return _ALL_DEFAULT_PROBLEMS


def get_default(n: int, *args, **kwargs):
    return _ALL_DEFAULT_PROBLEMS[n](*args, **kwargs)
