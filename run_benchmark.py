from mealpy import PSO, DE, GWO, WOA, GA, SHADE, DE, EP

from bench import AlgorithmSpec, BenchmarkSuite, ClassFunctionSpec
from bench.func import get_all_default_problems

from collection.b2026.deal import DEAL

ALGORITHMS = [
    AlgorithmSpec(name="PSO", cls=PSO.OriginalPSO),
    # AlgorithmSpec(name="GWO", cls=GWO.OriginalGWO),

    # AlgorithmSpec(name="SHADE", cls=SHADE.OriginalSHADE),
    # AlgorithmSpec(name="LSHADE", cls=SHADE.L_SHADE),
    # AlgorithmSpec(name="JADE", cls=DE.JADE),
    # AlgorithmSpec(name="DE", cls=DE.OriginalDE),

    AlgorithmSpec(name="EP", cls=EP.LevyEP),

    # AlgorithmSpec(name="WOA", cls=WOA.OriginalWOA),
    # AlgorithmSpec(name="GA", cls=GA.BaseGA),

    # AlgorithmSpec(name="HPSOA", cls=HPSO_LH2026A),
    # AlgorithmSpec(name="HPSOB", cls=HPSO_LH2026B),
    # AlgorithmSpec(name="HPSOC", cls=HPSO_LH2026C),

    AlgorithmSpec(name="DEAL", cls=DEAL),
]

FUNCTIONS = [
    ClassFunctionSpec(p, ndim=20) for p in get_all_default_problems()
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
