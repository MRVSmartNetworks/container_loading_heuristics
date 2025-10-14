#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import random
from typing import List

import pandas as pd

from benchmark import ExactSolver, MasterAco, SolverACO, SolverORTools
from benchmark.aco.solver_ACO import ACOException
from multi_runs.config import (
    CHECKPOINT_PATH,
    DATASETS,
    N_ITER,
    ONLY_STATS,
    RUNS,
    SUMMARY_PATH,
)
from multi_runs.profiler import profile_usage
from multi_runs.utils import eval_cost, stats_properties
from sol_representation import sol_check

docstring = """
Test a specific solver for the "3D Bin Packing with Stackable Items" problem.

The available solvers are:
- ACO: column generation based on the Ant Colony Generation (ACO)
- Exact Solver: exact solution of the 3D Bin Packing problem (1 truck only)
- OR Tools: solves the problem through the exact solution of each single truck
"""

all_solvers = (ExactSolver, SolverACO, MasterAco, SolverORTools)


@profile_usage(interval=0.5)
def single_run(
    solver_name: str,
    dataset_name: str,
    run_number: int,
):
    """
    Performs individual run for one solver, using a specific instance of a dataset.
    This is meant to run `N_ITER` times.
    """
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )
    if "thpack" in dataset_name:
        df_vehicles = df_vehicles.iloc[1].to_frame().T
    sol_file_name = f"{solver_name}_{dataset_name}_sol.csv"

    try:
        solver = RUNS[solver_name]["solver"]()
        extra_res = {}  # Paceholder kwarg
        # Common solver API:
        #  time, cost = solver.solve(items, vehicles, out_filename, time_limit, ...)
        t, solver_cost = solver.solve(
            df_items,
            df_vehicles,
            sol_file_name=sol_file_name,
            time_limit=300,
            pass_t_aco=extra_res,
        )
        # Read dataframe solution
        if os.path.exists(os.path.join("results", sol_file_name)):
            df_sol = pd.read_csv(
                os.path.join("results", sol_file_name),
            )
            os.makedirs(
                os.path.join(".", "results", dataset_name),
                exist_ok=True,
            )
            df_sol.to_csv(f"./results/{dataset_name}/{random.randint(0, 100)}_{sol_file_name}")
            # Check if solution is correct
            try:
                of = sol_check(df_sol, df_vehicles, df_items)
            except Exception as e:
                of = f"{e}"
                print(of)
                return
            # Evaluate the total cost
            cost = eval_cost(df_sol, df_vehicles)
            print(f"\nIteration {run_number}: cost={cost}, time={t}\n")

            # save checkpoint
            f_checkp = open(
                CHECKPOINT_PATH / f"{solver_name}_{dataset_name}_checkpoint.csv",
                "a",
            )
            if os.stat(CHECKPOINT_PATH / f"{solver_name}_{dataset_name}_checkpoint.csv").st_size == 0:
                f_checkp.write(f"cost,time,solver_cost,ACO_time\n")
            f_checkp.write(f"{cost},{t},{solver_cost},{-1 if 'tACO' not in extra_res else extra_res['tACO']}\n")

            f_checkp.close()
        else:
            print("The solver did not generate a solution CSV.\nThis is to be expected if using `--exact`")
    except ACOException:
        print(f"\nItems cannot be stored in the proposed trucks for {dataset_name}.\n")


def run_solver_datasets(solver_name: str, datasets: List[List[str]]):
    for ds_list in datasets:
        for i, dataset_name in enumerate(ds_list):
            print(f"{dataset_name}\n============================================\n")
            if ONLY_STATS:
                continue

            for i in range(N_ITER):
                print(f"++++++++++++++++++ Iteration {i + 1} ++++++++++++++++++")
                single_run(solver_name, dataset_name, i)

            stats_properties(
                CHECKPOINT_PATH / f"{solver_name}_{dataset_name}_checkpoint.csv",
                SUMMARY_PATH / f"{solver_name}_summary.csv",
                dataset_name,
            )


def main(args):
    CHECKPOINT_PATH.mkdir(exist_ok=True)
    SUMMARY_PATH.mkdir(exist_ok=True)

    used_solvers = args.solver

    used_ds = {}
    if args.dataset is not None and args.dataset != []:
        for solv in used_solvers:
            used_ds[solv] = (DATASETS[ds] for ds in args.dataset)
    else:
        for solv in used_solvers:
            used_ds[solv] = RUNS[solv]["default_datasets"]

    for solver in used_solvers:
        run_solver_datasets(solver, used_ds[solver])


if __name__ == "__main__":
    supported_solvers = list(RUNS.keys())
    supported_datasets = ("realistic-ds", "mod-ds", "ivancic-ds", "beng-ds", "exact-ds")
    parser = argparse.ArgumentParser(description=docstring)
    parser.add_argument(
        "--solver",
        metavar="SOLVER",
        required=True,
        type=str,
        choices=supported_solvers,
        nargs="+",
        help=f"solver(s) to be used; supports: {supported_solvers}",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        required=False,
        type=str,
        choices=supported_datasets,
        nargs="+",
        help=f"dataset(s) to be used - if not specified, will use the default ones for the solver(s); supports: {supported_datasets}",
    )
    args = parser.parse_args()

    main(args)
