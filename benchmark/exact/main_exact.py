#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
from exact import *
from benchmark_aco import *
from sol_representation import *

EXACT = False
COMPARE_ACO = True

if __name__ == "__main__":
    script_folder = os.path.dirname(__file__)
    dataset_folder = os.path.join(script_folder, "exact", "test_")

    dataset_name = "test_8"

    dataset = os.path.join(dataset_folder, dataset_name)

    df_items = pd.read_csv(
        os.path.join("paper-tests", "test_", dataset, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join("paper-tests", "test_", dataset, "vehicles.csv"),
    )

    if EXACT:
        model_path = os.path.join(script_folder, "logs")
        output_variables_path = os.path.join(
            script_folder, "exact", "test_", dataset_name, "out_data.p"
        )

        exact_solver = ExactSolver(df_items, df_vehicles, model_path)
        exact_solver.solve()

    # Use ACO solver
    if COMPARE_ACO:
        np.random.seed(27839)
        sol_file_name = f"aco_out_{dataset_name}.csv"

        aco_solver = SolverACO()
        aco_solver.solve(df_items, df_vehicles, sol_file_name)

        # Print results
        df_sol = pd.read_csv(
            os.path.join("results", sol_file_name),
        )
        # orthogonal_plane(df_items, df_vehicles, df_sol, idx_vehicle=0)
        stack_3D(df_items, df_vehicles, df_sol, idx_vehicle=0)

        print(sol_check(df_sol, df_vehicles, df_items))
