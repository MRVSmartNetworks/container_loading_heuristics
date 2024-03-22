#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from benchmark import (ExactSolver, SolverACO, columnGeneration, masterAco,
                       solverORTools)
from benchmark.aco.solver_ACO import Our_exception
from sol_representation import *

"""
Example usage:
      python3 test_all_datasets.py --master-aco --or-tools
"""

datasets = [
    "dataset_small",
    "datasetA",
    "datasetB",
    "datasetC",
    "datasetD",
    "datasetE",
    "datasetF",
    "datasetG",
    "datasetH",
    "datasetI",
    "datasetJ",
    "easy_dataset",
    "MODdatasetG",
]

ivancic_datasets = [
    "thpack9_1",
    "thpack9_2",
    "thpack9_3",
    "thpack9_4",
    "thpack9_5",
    "thpack9_6",
    "thpack9_7",
    "thpack9_8",
    "thpack9_9",
    "thpack9_10",
    "thpack9_11",
    "thpack9_12",
    "thpack9_13",
    "thpack9_14",
    "thpack9_15",
    "thpack9_16",
    "thpack9_17",
    "thpack9_18",
    "thpack9_19",
    "thpack9_20",
    "thpack9_21",
    "thpack9_22",
    "thpack9_23",
    "thpack9_24",
    "thpack9_25",
    "thpack9_26",
    "thpack9_27",
    "thpack9_28",
    "thpack9_29",
    "thpack9_30",
    "thpack9_31",
    "thpack9_32",
    "thpack9_33",
    "thpack9_34",
    "thpack9_35",
    "thpack9_36",
    "thpack9_37",
    "thpack9_38",
    "thpack9_39",
    "thpack9_40",
    "thpack9_41",
    "thpack9_42",
    "thpack9_43",
    "thpack9_44",
    "thpack9_45",
    "thpack9_46",
    "thpack9_47",
]

beng_datasets = [
    "BENG01",
    "BENG02",
    "BENG03",
    "BENG04",
    "BENG05",
    "BENG06",
    "BENG07",
    "BENG08",
    "BENG09",
    "BENG10",
]

# Configuration:
RUNS = {
    "ExactSolver": {
        "solver": ExactSolver,  # Solver class
        "datasets": (("easy_dataset")),  # Tuple containing the datasets
    },
    # "SolverACO":
    # "columnGeneration": (),
    "masterAco": {
        "solver": masterAco,
        "datasets": (datasets, ivancic_datasets, beng_datasets),
    },
    "solverORTools": {
        "solver": solverORTools,
        "datasets": (),
    },
}

N_ITER = 10
CHECKPOINT_PATH = "./results/checkpoints/"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
SUMMARY_PATH = "./results/summaries/"
os.makedirs(SUMMARY_PATH, exist_ok=True)
ONLY_STATS = False
IVANCIC = False

all_solvers = (ExactSolver, SolverACO, columnGeneration, masterAco, solverORTools)


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
    df_checkp = pd.read_csv(path_checkpoint, sep=",")
    avg_cost = df_checkp["cost"].mean()
    std_cost = df_checkp["cost"].std()
    min_cost = df_checkp["cost"].min()
    avg_cost_ACO = df_checkp["solver_cost"].mean()
    std_cost_ACO = df_checkp["solver_cost"].std()
    min_cost_ACO = df_checkp["solver_cost"].min()
    avg_t = df_checkp["time"].mean()
    std_t = df_checkp["time"].std()

    print(
        f"\n{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{(avg_cost_ACO):.2f},{std_cost_ACO:.2f},{min_cost_ACO},{avg_t:.2f},{std_t:.2f}\n"
    )
    f = open(path_summary, "a")
    if os.stat(path_summary).st_size == 0:
        f.write(
            f"dataset,avg_cost,std_cost,minimum_cost,avg_cost_ACO,std_cost_ACO,min_cost_ACO,avg_time,std_time,iterations\n"
        )

    f.write(
        f"{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{(avg_cost_ACO):.2f},{std_cost_ACO:.2f},{min_cost_ACO},{avg_t:.2f},{std_t:.2f},{N_ITER}\n"
    )
    f.close()


parser = argparse.ArgumentParser()
parser.add_argument("--exact", action="store_true", help="Select exact solver")
# parser.add_argument("--solver-aco", action="store_true", help="Select solver ACO")
# parser.add_argument(
#     "--column-generation", action="store_true", help="Select Column Generation"
# )
parser.add_argument("--master-aco", action="store_true", help="Select Master ACO")
parser.add_argument("--or-tools", action="store_true", help="Select OR Tools solver")

if __name__ == "__main__":
    args = parser.parse_args()

    used_solvers = []
    if args.exact:
        used_solvers.append("ExactSolver")
    if args.master_aco:
        used_solvers.append("masterAco")
    if args.or_tools:
        used_solvers.append("solverORTools")

    for k in used_solvers:
        for ds_list in RUNS[k]["datasets"]:
            for i, dataset_name in enumerate(ds_list):
                df_items = pd.read_csv(
                    os.path.join(".", "data", dataset_name, "items.csv"),
                )
                df_vehicles = pd.read_csv(
                    os.path.join(".", "data", dataset_name, "vehicles.csv"),
                )
                if ds_list == ivancic_datasets:
                    df_vehicles = df_vehicles.iloc[1].to_frame().T
                sol_file_name = f"{k}_{dataset_name}_sol.csv"
                # measure execution time

                print(f"{dataset_name}\n============================================\n")
                if not ONLY_STATS:
                    for i in range(N_ITER):
                        print(f"++++++++++++++++++ Iteration {i+1} ++++++++++++++++++")
                        try:
                            solver = RUNS[k]["solver"]()
                            # Need common API:
                            #  time, cost = solve(items, vehicles, out_filename, time_limit)

                            t, solver_cost = solver.solve(
                                df_items,
                                df_vehicles,
                                sol_file_name=sol_file_name,
                                time_limit=300,
                            )
                            # read dataframe solution
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
                                f_checkp.write(f"cost,time,solver_cost\n")
                            f_checkp.write(f"{cost},{t},{solver_cost}\n")

                            f_checkp.close()
                        except Our_exception:
                            print(
                                f"\nItems cannot be stored in the proposed trucks for {dataset_name}.\n"
                            )

                stats_properties(
                    CHECKPOINT_PATH + f"{solver.name}_{dataset_name}_checkpoint.csv",
                    SUMMARY_PATH + f"{solver.name}_summary.csv",
                    dataset_name,
                )
