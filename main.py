from mealpy import PSO, DE, GWO, WOA, GA

from bench import AlgorithmSpec, BenchmarkSuite, ClassFunctionSpec
from collection.b2026.hpso_lh import HPSO_LH2026A, HPSO_LH2026B, HPSO_LH2026C

from collection.b2026.k import OptimizerK
from collection.b2026.kf import OptimizerKF
from collection.b2026.khg import OptimizerKHG
from collection.b2026.ku import OptimizerKu

import opfunu


ALL_DEFAULT_PROBLEMS = (
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
    # opfunu.name_based.XinSheYang01,
    # opfunu.name_based.YaoLiu04,
    # opfunu.name_based.Zacharov
)

ALGORITHMS = [
    AlgorithmSpec(name="PSO", cls=PSO.OriginalPSO),
    # AlgorithmSpec(name="DE", cls=DE.OriginalDE),
    # AlgorithmSpec(name="GWO", cls=GWO.OriginalGWO),
    # AlgorithmSpec(name="WOA", cls=WOA.OriginalWOA),
    # AlgorithmSpec(name="GA", cls=GA.BaseGA),

    AlgorithmSpec(name="K", cls=OptimizerK),
    # AlgorithmSpec(name="KF", cls=OptimizerKF),
    # AlgorithmSpec(name="KHG", cls=OptimizerKHG),
    # AlgorithmSpec(name="KU", cls=OptimizerKu),
    AlgorithmSpec(name="HPSOA", cls=HPSO_LH2026A, kwargs=dict(c1=1.49, c2=1.49, beta=1.5, alpha=0.01, s_max=3)),
    AlgorithmSpec(name="HPSOB", cls=HPSO_LH2026B, kwargs=dict(c1=1.49, c2=1.49, beta=1.5, alpha=0.01, s_max=3)),
    AlgorithmSpec(name="HPSOC", cls=HPSO_LH2026C, kwargs=dict(c1=1.49, c2=1.49, beta=1.5, alpha=0.01, s_max=3))
]

FUNCTIONS = [
    ClassFunctionSpec(p, ndim=20) for p in ALL_DEFAULT_PROBLEMS
]

if __name__ == "__main__":
    suite = BenchmarkSuite(
        functions=FUNCTIONS,
        algorithms=ALGORITHMS,
        epoch=200,
        pop_size=75,
        n_runs=3,
        n_workers=15,
    )

    print("Running benchmark...")
    suite.run()

    suite.print_results()
    suite.save(output_dir="results")
