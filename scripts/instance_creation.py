# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd


def create_instance(data_folder, df_items, df_vehicles):
    if not os.path.exists(os.path.join(".", "data", data_folder)):
        os.mkdir(os.path.join(".", "data", data_folder))
    df_items.to_csv(os.path.join("data", data_folder, "items.csv"), index=False)
    df_vehicles.to_csv(os.path.join("data", data_folder, "vehicles.csv"), index=False)


if __name__ == "__main__":
    n_items_to_gen = 5000
    bins_data = [
        {
            "length": 1500,
            "width": 800,
            "height": 100,
            "nesting_height": 110,
            "portata": 1500,
            "max_stackability": 4,
        },
        {
            "length": 1500,
            "width": 800,
            "height": 100,
            "nesting_height": 110,
            "portata": 2000,
            "max_stackability": 4,
        },
        {
            "length": 1500,
            "width": 1000,
            "height": 100,
            "nesting_height": 110,
            "portata": 1500,
            "max_stackability": 4,
        },
        {
            "length": 1500,
            "width": 1000,
            "height": 100,
            "nesting_height": 110,
            "portata": 2000,
            "max_stackability": 4,
        },
    ]
    a = 2
    b = 5
    perc_void = 0.2
    data_items = {
        "id_item": [],
        "length": [],
        "width": [],
        "height": [],
        "weight": [],
        "nesting_height": [],
        "stackability_code": [],
        "forced_orientation": [],
        "max_stackability": [],
    }
    n_digits_idx = int(math.log10(n_items_to_gen)) + 1
    for count in range(n_items_to_gen):
        sel_pos = np.random.randint(0, len(bins_data))
        sel_item = bins_data[sel_pos]
        data_items["id_item"].append(f"I{str(count).zfill(n_digits_idx)}")
        data_items["length"].append(sel_item["length"])
        data_items["width"].append(sel_item["width"])
        data_items["height"].append(sel_item["height"])
        if np.random.uniform() < perc_void:
            weight = 0
        else:
            # different materials may have different weights
            weight = sel_item["portata"] * 0.3 + sel_item[
                "portata"
            ] * 0.5 * np.random.beta(a, b)

        data_items["weight"].append(np.round(weight, 0))
        data_items["nesting_height"].append(sel_item["nesting_height"])
        if sel_pos in [0, 1]:
            data_items["stackability_code"].append(0)
        else:
            data_items["stackability_code"].append(1)
        data_items["forced_orientation"].append("n")
        data_items["max_stackability"].append(sel_item["max_stackability"])

    df_items = pd.DataFrame.from_dict(data_items)
    data_vehicles = {
        "id_truck": ["V0"],
        "length": [10000],
        "width": [3000],
        "height": [100],
        "max_weight": [100000],
        "max_weight_stack": [100000],
        "cost": [1],
        "max_density": [100000],
    }
    df_vehicles = pd.DataFrame.from_dict(data_vehicles)
    # print on file
    create_instance("./easy_dataset/", df_items, df_vehicles)
