#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from columnGeneration import columnGeneration

if __name__ == "__main__":
    dataset_name = "dataset_small"
    sol_file_name = f"{dataset_name}_"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )
    
    model_cg = columnGeneration(df_vehicles, df_items)
    bestAnts = model_cg.solve()
