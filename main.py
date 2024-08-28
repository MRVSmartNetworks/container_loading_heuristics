#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import pandas as pd

from benchmark import ExactSolver, SolverACO, columnGeneration, masterAco, solverORTools

# from solver import Solver23
from sol_representation import *

SAVE_COMPARISON = True
TRUCK_TO_PRINT = 10

def create_comparison(
    df_sol: pd.DataFrame,
    df_vehicles: pd.DataFrame,
    dataset_name: str,
    solver_name: str,
    t: float,
) -> dict:
    n_vehicles = df_sol["idx_vehicle"].iloc[-1] + 1
    vehicles_type = pd.unique(df_sol["type_vehicle"])
    cost = 0.0
    for _, ele in (
        df_sol.filter(items=["idx_vehicle", "type_vehicle"])
        .drop_duplicates()
        .iterrows()
    ):
        cost += df_vehicles[df_vehicles.id_truck == ele.type_vehicle].iloc[0].cost

    new_comp = {
        "solver": [solver_name],
        "dataset": [dataset_name],
        "n_vehicles": [n_vehicles],
        "vehicles": [vehicles_type],
        "cost": round(cost, 2),
        "time": t,
    }
    return new_comp

if __name__ == "__main__":
    dataset_name = "thpack9_47"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )
    
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
