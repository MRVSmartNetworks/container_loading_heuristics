# -*- coding: utf-8 -*-
import numpy as np
import warnings

# Remotion of the warnings as output
warnings.simplefilter(action="ignore", category=FutureWarning)


def sol_check(df_sol, df_vehicles, df_items):
    # Make sure each item has been used exactly once
    # (assuming the dataset does not contain duplicate items)
    df_items_index = df_items.id_item
    used_items, count_used_items = np.unique(df_sol.id_item, return_counts=True)

    # All items have been used exactly once in the solution (no duplicates)
    assert len(used_items) == len(
        df_sol.id_item
    ), f"The solution contains {len(df_sol.id_item)} items, but {len(used_items)} distinct items have been used!"
    # All items in the dataset have been used exactly once
    assert len(used_items) == len(
        df_items_index
    ), f"The number of items in the dataset is {len(df_items_index)}, but the solution contains {len(used_items)} distinct items!"

    id_vehicles = df_sol.idx_vehicle.unique()
    for id in id_vehicles:
        df_sol_single_vehicle = df_sol[df_sol["idx_vehicle"] == id]
        vehicle_type = df_sol_single_vehicle.type_vehicle.unique()[0]
        dimensions_names = ["length", "width", "height"]
        tot_weight = 0
        tot_surface = 0
        data_vehicle = df_vehicles.query(f"id_truck=='{vehicle_type}'")
        df_sol_single_vehicle = df_sol_single_vehicle.query(
            f"type_vehicle == '{vehicle_type}'"
        )
        n_items_vehicle = len(df_sol_single_vehicle)
        coordinates = df_sol_single_vehicle[
            ["x_origin", "y_origin", "z_origin"]
        ].to_numpy()
        sizes = np.zeros((n_items_vehicle, 3))
        density = np.zeros(n_items_vehicle)
        for i in range(n_items_vehicle):
            data_item = df_items.query(
                f"id_item=='{df_sol_single_vehicle.iloc[i].id_item}'"
            )
            for d in range(2):
                if df_sol_single_vehicle["orient"].iloc[i] == "l":
                    sizes[i, d] = data_item[dimensions_names[d]]
                else:
                    sizes[i, d] = data_item[dimensions_names[1 - d]]

            sizes[i, 2] = data_item[dimensions_names[2]]

            tot_weight += data_item.iloc[0].weight

            # If the item is placed at z=0, update the used surface
            if coordinates[i, 2] == 0:
                tot_surface += sizes[i, 0] * sizes[i, 1]

            for d in range(3):
                assert (
                    coordinates[i, d] + sizes[i, d]
                    <= data_vehicle.iloc[0][dimensions_names[d]]
                ), f"Item {data_item.iloc[0].id_item} is out of bound for vehicle {vehicle_type} ({dimensions_names[d]} not respected)"

        tot_dens = tot_weight / tot_surface

        assert (
            tot_dens <= data_vehicle.iloc[0].max_density
        ), f"Density is out of bound for vehicle {id} of type {vehicle_type}"
        assert (
            tot_weight <= data_vehicle.iloc[0].max_weight
        ), f"Weight is out of bound for vehicle {id} of type {vehicle_type}"

        idx_stack = df_sol_single_vehicle.id_stack.unique()
        for stack in idx_stack:
            tot_weight_stack = 0
            df_sol_stack = df_sol_single_vehicle.query(f"id_stack == '{stack}'")
            n_items_stack = len(df_sol_stack)
            for i in range(n_items_stack):
                data_item_stack = df_items.query(
                    f"id_item=='{df_sol_stack.iloc[i].id_item}'"
                )
                tot_weight_stack += data_item_stack.iloc[0].weight
            assert (
                tot_weight_stack <= data_vehicle.iloc[0].max_weight_stack
            ), f"Stack {stack} weight is out of bound for vehicle {vehicle_type}"
    of = 0.0
    for _, ele in (
        df_sol.filter(items=["idx_vehicle", "type_vehicle"])
        .drop_duplicates()
        .iterrows()
    ):
        of += df_vehicles[df_vehicles.id_truck == ele.type_vehicle].iloc[0].cost
    print("The solution is correct.")
    return of
