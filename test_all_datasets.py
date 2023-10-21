#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from solver import *
from sol_representation import *

if __name__ == "__main__":
    datasets = [
        # 'datasetA',
        # 'datasetB',
        # 'datasetC',
        # 'datasetD',
        'datasetE',
        'datasetF',
        'datasetG',
        'datasetH',
        # 'datasetI',
        'datasetJ',
        # 'dataset_small'
    ]
    solver = Solver10()
    f = open(
        os.path.join(".", "results", f"{solver.name}_summary.csv"),
        "a"
    )
    for dataset in datasets:
        df_items = pd.read_csv(
            os.path.join(".", "data", dataset, "items.csv"),
        )
        df_vehicles = pd.read_csv(
            os.path.join(".", "data", dataset, "vehicles.csv"),
        )
        sol_file_name = f"{solver.name}_{dataset}_sol.csv"
        # measure execution time
        start = time.time()
        solver.solve(df_items, df_vehicles, sol_file_name, time_limit=300)
        stop = time.time()
        # read dataframe solution
        df_sol = pd.read_csv(
            os.path.join("results", sol_file_name),
        )
        try:
            of = sol_check(df_sol, df_vehicles, df_items)
        except Exception as e:
            of = f"{e}"
        f.write(f"{dataset},{(stop - start):.2f},{of}\n")
    f.close()
