# -*- coding: utf-8 -*-
import os
import pandas as pd
from exact.check.check_sol import check_3D, check_above
import numpy as np

if __name__ == "__main__":
    df_items = pd.read_csv(
        os.path.join(".", "data", "test_real_1672_AS", "items.csv"),
    )

    df_vehicles = pd.read_csv(
        os.path.join(".", "data", "test_real_1672_AS", "vehicles.csv"),
    )

    df_sol = pd.read_csv(
        os.path.join("tmp_sol.csv"),
    )

    # Define sets
    n_items = len(df_items)
    items = range(n_items)
    n_vehicles = len(df_vehicles)
    vehicles = range(n_vehicles)

    df_vehicles = df_vehicles[["length", "width", "height"]].to_numpy()
    dimensions_names = ["length", "width", "height"]

    idx_vehicle = 0
    vehicle_load = df_sol.query(f"n_vehicle == {idx_vehicle}")

    orient = vehicle_load["orient"]
    coordinates = df_sol[["x_origin", "y_origin", "z_origin"]].to_numpy()
    sizes = np.zeros((n_items, 3))
    for i in range(len(vehicle_load["orient"])):
        for d in range(2):
            if orient[i] == "w":
                sizes[i, d] = df_items.iloc[i][dimensions_names[d]]
            else:
                sizes[i, d] = df_items.iloc[i][dimensions_names[1 - d]]

        sizes[i, 2] = df_items.iloc[i][dimensions_names[2]]

    # SECONDO ME TROPPI BIN, se stampassimo gli stack?
    # check_3D(
    #     coordinates=coordinates,
    #     sizes=sizes,
    #     n_items=n_items,
    #     df_vehicles=df_vehicles
    # )

    # check_above(
    #     n_stacks=n_items,
    #     x_o=,
    #     y_o=,
    #     x_e=,
    #     y_e=,
    #     df_vehicles=df_vehicles
    # )
