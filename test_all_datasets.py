#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from benchmark import SolverACO, columnGeneration, masterAco, solverORTools, ExactSolver
from sol_representation import *
import numpy as np
import sys

N_ITER = 2


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


if __name__ == "__main__":
    datasets = [
        # 'datasetA',
        # 'datasetB',
        # 'datasetC',
        # 'datasetD',
        #'datasetE',
        # "datasetF",
        #'datasetG',
        #'datasetH',
        # 'datasetI',
        #'datasetJ',
        "dataset_small",
    ]
    solver = masterAco()
    f = open(os.path.join(".", "results", f"{solver.name}_summary.csv"), "a")
    f.write(f"dataset,avg_cost,std_cost,minimum_cost,avg_time,std_time\n")
    for d, dataset in enumerate(datasets):
        df_items = pd.read_csv(
            os.path.join(".", "data", dataset, "items.csv"),
        )
        df_vehicles = pd.read_csv(
            os.path.join(".", "data", dataset, "vehicles.csv"),
        )
        sol_file_name = f"{solver.name}_{dataset}_sol.csv"
        # measure execution time
        cost_lst = []
        t_lst = []
        print(f"============================================\n{dataset}\n")
        for i in range(N_ITER):
            solver = masterAco()

            sys.stdout = open(os.devnull, "w")  # Disable printing of solver
            t_lst.append(
                solver.solve(df_items, df_vehicles, sol_file_name, time_limit=300)
            )
            sys.stdout = sys.__stdout__
            # read dataframe solution
            df_sol = pd.read_csv(
                os.path.join("results", sol_file_name),
            )
            # Check if solution is correct
            try:
                of = sol_check(df_sol, df_vehicles, df_items)
            except Exception as e:
                of = f"{e}"
                print(of)
                continue
            # Evaluate the total cost
            cost_lst.append(eval_cost(df_sol, df_vehicles))
            print(f"Iteration {i}: cost={cost_lst[-1]}, time={t_lst[-1]}")

        avg_cost = np.average(cost_lst)
        std_cost = np.std(cost_lst)
        min_cost = min(cost_lst)
        avg_t = np.average(t_lst)
        std_t = np.std(t_lst)

        print(
            f"\n{dataset},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f}\n"
        )

        f.write(
            f"{dataset},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f}\n"
        )
    f.close()
