#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import pandas as pd

from benchmark import ExactSolver, SolverACO, columnGeneration, masterAco, solverORTools

# from solver import Solver23
from sol_representation import *

SAVE_COMPARISON = True
TRUCK_TO_PRINT = 10

if __name__ == "__main__":
    dataset_name = "thpack9_47"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )

    # Impose constraints on dataset
    # df_items = df_items.sample(frac=1)
    # df_vehicles = df_vehicles.iloc[[0, 1]]
    # df_vehicles["height"] = 1000
    # df_vehicles["max_weight"] = 100000
    # df_vehicles["max_weight_stack"] = 100000
    # df_items["height"] = 1000
    # df_items = df_items.sample(n=10)

    solver = SolverACO()
    sol_file_name = f"{solver.name}_{dataset_name}_sol.csv"
    print(f"\n{solver.name}\n==========================================")
    t = solver.solve(df_items, df_vehicles, sol_file_name=sol_file_name)

    # read dataframe solution
    df_sol = pd.read_csv(
        os.path.join("results", sol_file_name),
    )
    for i in range(TRUCK_TO_PRINT):
        try:
            # orthogonal_plane(df_items, df_vehicles, df_sol, idx_vehicle=i)
            stack_3D(df_items, df_vehicles, df_sol, idx_vehicle=i)
        except:
            break

    # Save solution information in benchmark.csv
    if SAVE_COMPARISON:
        if os.stat("results/benchmark.csv").st_size == 0:
            df_bench = pd.DataFrame()
        else:
            df_bench = pd.read_csv(os.path.join("results", "benchmark.csv"))
        new_row = create_comparison(df_sol, df_vehicles, dataset_name, solver.name, t)
        df_bench = pd.concat([pd.DataFrame(new_row), df_bench.loc[:]])
        df_bench.to_csv(os.path.join("results", "benchmark.csv"), index=False)

    of = sol_check(df_sol, df_vehicles, df_items)
    print(of)
