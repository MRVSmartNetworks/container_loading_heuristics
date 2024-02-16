#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from benchmark import SolverACO, columnGeneration, masterAco, solverORTools, ExactSolver
from sol_representation import *
import numpy as np
import random

N_ITER = 10
CHECKPOINT_PATH = "./results/checkpoints/"
SUMMARY_PATH = "./results/summaries/"
ONLY_STATS = False


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
    avg_t = df_checkp["time"].mean()
    std_t = df_checkp["time"].std()

    print(
        f"\n{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f}\n"
    )
    f = open(path_summary, "a")
    if os.stat(path_summary).st_size == 0:
        f.write(
            f"dataset,avg_cost,std_cost,minimum_cost,avg_time,std_time,iterations\n"
        )

    f.write(
        f"{dataset},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f},{N_ITER}\n"
    )
    f.close()


if __name__ == "__main__":
    datasets = [
        # "datasetA",
        # "datasetB",
        # "datasetC",
        "datasetD",
        "datasetE",
        "datasetF",
        # "datasetG",
        # "datasetH",
        # "datasetI",
        # "datasetJ",
        # "dataset_small",
        # "easy_dataset",
        # "MODdatasetG"
    ]
    solver = masterAco()
    for d, dataset in enumerate(datasets):
        df_items = pd.read_csv(
            os.path.join(".", "data", dataset, "items.csv"),
        )
        df_vehicles = pd.read_csv(
            os.path.join(".", "data", dataset, "vehicles.csv"),
        )
        sol_file_name = f"{solver.name}_{dataset}_sol.csv"
        # measure execution time

        print(f"{dataset}\n============================================\n")
        if not ONLY_STATS:
            for i in range(N_ITER):
                print(f"++++++++++++++++++ Iteration {i} ++++++++++++++++++")
                solver = masterAco()

                t = solver.solve(df_items, df_vehicles, sol_file_name, time_limit=300)
                # read dataframe solution
                df_sol = pd.read_csv(
                    os.path.join("results", sol_file_name),
                )
                df_sol.to_csv(
                    f"./results/{dataset}/{sol_file_name}_{random.randint(0,100)}"
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
                    CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv", "a"
                )
                if (
                    os.stat(
                        CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv"
                    ).st_size
                    == 0
                ):
                    f_checkp.write(f"cost,time\n")
                f_checkp.write(f"{cost},{t}\n")

                f_checkp.close()

        stats_properties(
            CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv",
            SUMMARY_PATH + f"{solver.name}_summary.csv",
            dataset,
        )
