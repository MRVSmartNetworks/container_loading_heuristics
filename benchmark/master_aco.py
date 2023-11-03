#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from masterProblem import MasterProblem
from solver_ACO import SolverACO

if __name__ == "__main__":
    dataset_name = "datasetA"
    sol_file_name = f"{dataset_name}_"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )

    # df_vehicles = df_vehicles.iloc[[1, 6]]
    sol_aco = SolverACO()
    df_sol, totCost, pattern_list, time_spent = sol_aco.solver(df_items, df_vehicles)
    master = MasterProblem()
