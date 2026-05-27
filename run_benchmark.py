from mealpy import PSO, DE, GWO, WOA, GA, SHADE, DE, EP

from bench import AlgorithmSpec, BenchmarkSuite, ClassFunctionSpec
from bench.func import get_all_default_problems

from collection.b2026.deal import DEALL

ALGORITHMS = [
    AlgorithmSpec(name="PSO", cls=PSO.OriginalPSO),
    # AlgorithmSpec(name="GWO", cls=GWO.OriginalGWO),

    AlgorithmSpec(name="SHADE", cls=SHADE.OriginalSHADE),
    AlgorithmSpec(name="LSHADE", cls=SHADE.L_SHADE),
    AlgorithmSpec(name="JADE", cls=DE.JADE),
    AlgorithmSpec(name="DE", cls=DE.OriginalDE),

    AlgorithmSpec(name="EP", cls=EP.LevyEP),

    # AlgorithmSpec(name="WOA", cls=WOA.OriginalWOA),
    # AlgorithmSpec(name="GA", cls=GA.BaseGA),

    AlgorithmSpec(name="DEAL_MODE", cls=DEALL, kwargs=dict(stats_mode='mode')),
    AlgorithmSpec(name="DEAL_MEDIAN", cls=DEALL, kwargs=dict(stats_mode='median')),
    AlgorithmSpec(name="DEAL_MEAN", cls=DEALL, kwargs=dict(stats_mode='mean')),
    AlgorithmSpec(name="DEAL_S", cls=DEALL, kwargs=dict(stats_mode='sorted')),
]

FUNCTIONS = [
    ClassFunctionSpec(p, ndim=50) for p in get_all_default_problems()
]

if __name__ == "__main__":
    suite = BenchmarkSuite(
        functions=FUNCTIONS,
        algorithms=ALGORITHMS,
        epoch=250,
        pop_size=75,
        n_runs=3,
        n_workers=15,
    )

    print("Running benchmark...")
    suite.run()

    suite.print_results()
    suite.save(output_dir="results")
