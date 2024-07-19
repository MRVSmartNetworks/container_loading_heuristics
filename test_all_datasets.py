#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from benchmark import (ExactSolver, SolverACO, masterAco,
                       solverORTools)
from benchmark.aco.solver_ACO import Our_exception
from sol_representation import *

docstring = """
Test a specific solver for the "3D Bin Packing with Stackable Items" problem.

The available solvers are:
- ACO: column generation based on the Ant Colony Generation (ACO)
- Exact Solver: exact solution of the 3D Bin Packing problem (1 truck only)
- OR Tools: solves the problem through the exact solution of each single truck
"""

start_ch = "A"
end_ch = "J"
datasets = [f"dataset{chr(x)}" for x in range(ord(start_ch), ord(end_ch) * 1)]
mod_datasets = [f"MODdataset{chr(x)}" for x in range(ord(start_ch), ord(end_ch) * 1)]
ivancic_datasets = [f"thpack9_{x}" for x in range(1, 48)]
beng_datasets = [f"BENG0{x}" for x in range(1, 9)]
exact_datasets = [f"test_exact_{x}" for x in range(1, 10)]

app = []
# Configuration:
RUNS = {
    "ExactSolver": {"solver": ExactSolver, "datasets": (exact_datasets,)},
    "masterAco": {"solver": masterAco, "datasets": (datasets, app)},
    "solverORTools": {"solver": solverORTools, "datasets": (mod_datasets, app)},
}

N_ITER = 5
CHECKPOINT_PATH = "./results/checkpoints/"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
SUMMARY_PATH = "./results/summaries/"
os.makedirs(SUMMARY_PATH, exist_ok=True)
ONLY_STATS = False

all_solvers = (ExactSolver, SolverACO, masterAco, solverORTools)


def eval_cost(
    df_sol: pd.DataFrame,
    df_vehicles: pd.DataFrame,
) -> float:
    cost = 0.0
    for _, ele in (
        df_sol.filter(items=["idx_vehicle", "type_vehicle"])
        .drop_duplicates()
        .iterrows()
    ):
        cost += df_vehicles[df_vehicles.id_truck == ele.type_vehicle].iloc[0].cost

    return cost


def stats_properties(
    path_checkpoint: str, path_summary: str, dataset_name: str
) -> None:
    csv_header = (
        "dataset",
        "avg_cost",
        "std_cost",
        "minimum_cost",
        "avg_time",
        "std_time",
        "avg_cost_ACO",
        "std_cost_ACO",
        "min_cost_ACO",
        "avg_ACO_time",
        "std_ACO_time",
        "iterations",
    )
    df_checkp = pd.read_csv(path_checkpoint, sep=",")  # FIXME
    # masterACO info
    avg_cost = df_checkp["cost"].mean()
    std_cost = df_checkp["cost"].std()
    min_cost = df_checkp["cost"].min()
    avg_t = df_checkp["time"].mean()
    std_t = df_checkp["time"].std()
    # Solver ACO info
    avg_cost_ACO = df_checkp["solver_cost"].mean()
    std_cost_ACO = df_checkp["solver_cost"].std()
    min_cost_ACO = df_checkp["solver_cost"].min()
    avg_ACO_t = df_checkp["ACO_time"].mean()
    std_ACO_t = df_checkp["ACO_time"].std()

    new_iter_str = f"{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f},{(avg_cost_ACO):.2f},{std_cost_ACO:.2f},{min_cost_ACO},{avg_ACO_t:.2f},{std_ACO_t:.2f}"
    print("\n", new_iter_str, "\n")
    f = open(path_summary, "a")
    if os.stat(path_summary).st_size == 0:
        f.write(",".join(csv_header) + "\n")

    f.write(f"{new_iter_str},{N_ITER}\n")
    f.close()


def main(args):
    used_solvers = []
    if args.exact:
        used_solvers.append("ExactSolver")
    if args.master_aco:
        used_solvers.append("masterAco")
    if args.or_tools:
        used_solvers.append("solverORTools")

    used_ds = []
    if args.realistic_ds:
        used_ds.append(datasets)
    if args.mod_ds:
        used_ds.append(mod_datasets)
    if args.ivancic_ds:
        used_ds.append(ivancic_datasets)
    if args.beng_ds:
        used_ds.append(beng_datasets)
    if args.exact_ds:
        used_ds.append(exact_datasets)

    for k in used_solvers:
        for ds_list in used_ds:
            for i, dataset_name in enumerate(ds_list):
                df_items = pd.read_csv(
                    os.path.join(".", "data", dataset_name, "items.csv"),
                )
                df_vehicles = pd.read_csv(
                    os.path.join(".", "data", dataset_name, "vehicles.csv"),
                )
                if "thpack" in dataset_name:
                    df_vehicles = df_vehicles.iloc[1].to_frame().T
                sol_file_name = f"{k}_{dataset_name}_sol.csv"

                print(f"{dataset_name}\n============================================\n")
                if not ONLY_STATS:
                    for i in range(N_ITER):
                        print(f"++++++++++++++++++ Iteration {i+1} ++++++++++++++++++")
                        try:
                            solver = RUNS[k]["solver"]()
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
                                df_sol.to_csv(
                                    f"./results/{dataset_name}/{random.randint(0,100)}_{sol_file_name}"
                                )
                                # Check if solution is correct
                                try:
                                    of = sol_check(df_sol, df_vehicles, df_items)
                                except Exception as e:
                                    of = f"{e}"
                                    print(of)
                                    continue
                                # Evaluate the total cost
                                cost = eval_cost(df_sol, df_vehicles)
                                print(f"\nIteration {i}: cost={cost}, time={t}\n")

                                # save checkpoint
                                f_checkp = open(
                                    CHECKPOINT_PATH
                                    + f"{solver.name}_{dataset_name}_checkpoint.csv",
                                    "a",
                                )
                                if (
                                    os.stat(
                                        CHECKPOINT_PATH
                                        + f"{solver.name}_{dataset_name}_checkpoint.csv"
                                    ).st_size
                                    == 0
                                ):
                                    f_checkp.write(f"cost,time,solver_cost,ACO_time\n")
                                f_checkp.write(
                                    f"{cost},{t},{solver_cost},{-1 if 'tACO' not in extra_res else extra_res['tACO']}\n"
                                )

                                f_checkp.close()
                            else:
                                print(
                                    "The solver did not generate a solution CSV\n"
                                    "This is to be expected if using `--exact`"
                                )
                        except Our_exception:
                            print(
                                f"\nItems cannot be stored in the proposed trucks for {dataset_name}.\n"
                            )

                stats_properties(
                    CHECKPOINT_PATH + f"{solver.name}_{dataset_name}_checkpoint.csv",
                    SUMMARY_PATH + f"{solver.name}_summary.csv",
                    dataset_name,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)
    parser.add_argument("--exact", action="store_true", help="Select exact solver")
    parser.add_argument("--master-aco", action="store_true", help="Select Master ACO")
    parser.add_argument(
        "--or-tools", action="store_true", help="Select OR Tools solver"
    )
    parser.add_argument(
        "--realistic-ds", action="store_true", help="Use realistic data sets"
    )
    parser.add_argument("--mod-ds", action="store_true", help="Use MOD data sets")
    parser.add_argument(
        "--ivancic-ds", action="store_true", help="Use Ivancic data sets"
    )
    parser.add_argument("--beng-ds", action="store_true", help="Use BENG data sets")
    parser.add_argument(
        "--exact-ds", action="store_true", help="Use data sets for exact solver"
    )

    args = parser.parse_args()

    main(args)
