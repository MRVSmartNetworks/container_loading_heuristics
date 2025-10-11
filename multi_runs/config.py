import os
from pathlib import Path

from benchmark import ExactSolver, SolverACO, MasterAco, SolverORTools
from benchmark.aco.solver_ACO import ACOException

start_ch = "A"
end_ch = "J"


datasets = [f"dataset{chr(x)}" for x in range(ord(start_ch), ord(end_ch) + 1)]
mod_datasets = [f"MODdataset{chr(x)}" for x in range(ord(start_ch), ord(end_ch) + 1)]
ivancic_datasets = [f"thpack9_{x}" for x in range(1, 48)]
beng_datasets = [f"BENG0{x}" for x in range(1, 9)]
exact_datasets = [f"test_exact_{x}" for x in range(1, 10)]

DATASETS = {
    "realistic-ds": datasets,
    "mod-ds": mod_datasets,
    "ivancic-ds": ivancic_datasets,
    "beng-ds": beng_datasets,
    "exact-ds": exact_datasets,
}

app = []
# Configuration:
RUNS = {
    "exact-solver": {
        "solver": ExactSolver,
        "default_datasets": (exact_datasets,),
    },
    "master-aco": {"solver": MasterAco, "default_datasets": (datasets, app)},
    "or-tools": {
        "solver": SolverORTools,
        "default_datasets": (mod_datasets, app),
    },
}

project_root = Path(os.path.dirname(__file__)).parent

N_ITER = 5
CHECKPOINT_PATH = project_root / "results/checkpoints/"
SUMMARY_PATH = project_root / "results/summaries/"
ONLY_STATS = False
