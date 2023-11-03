#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from columnGeneration import columnGeneration

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
    model_cg = columnGeneration()
    bestAnts = model_cg.solve(df_items, df_vehicles)
