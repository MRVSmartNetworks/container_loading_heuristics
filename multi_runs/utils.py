import os
from pathlib import Path
from typing import Union

import pandas as pd

from .config import N_ITER

PathLike = Union[Path, str]


def eval_cost(
    df_sol: pd.DataFrame,
    df_vehicles: pd.DataFrame,
) -> float:
    cost = 0.0
    for _, ele in (
        df_sol.filter(items=["idx_vehicle", "type_vehicle"])
        .drop_duplicates()
        .iterrows()
    ):
        cost += df_vehicles[df_vehicles.id_truck == ele.type_vehicle].iloc[0].cost

    return cost


def stats_properties(
    path_checkpoint: PathLike, path_summary: PathLike, dataset_name: PathLike
) -> None:
    csv_header = (
        "dataset",
        "avg_cost",
        "std_cost",
        "minimum_cost",
        "avg_time",
        "std_time",
        "avg_cost_ACO",
        "std_cost_ACO",
        "min_cost_ACO",
        "avg_ACO_time",
        "std_ACO_time",
        "iterations",
    )
    df_checkp = pd.read_csv(path_checkpoint, sep=",")  # FIXME
    # masterACO info
    avg_cost = df_checkp["cost"].mean()
    std_cost = df_checkp["cost"].std()
    min_cost = df_checkp["cost"].min()
    avg_t = df_checkp["time"].mean()
    std_t = df_checkp["time"].std()
    # Solver ACO info
    avg_cost_ACO = df_checkp["solver_cost"].mean()
    std_cost_ACO = df_checkp["solver_cost"].std()
    min_cost_ACO = df_checkp["solver_cost"].min()
    avg_ACO_t = df_checkp["ACO_time"].mean()
    std_ACO_t = df_checkp["ACO_time"].std()

    new_iter_str = f"{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f},{(avg_cost_ACO):.2f},{std_cost_ACO:.2f},{min_cost_ACO},{avg_ACO_t:.2f},{std_ACO_t:.2f}"
    print("\n", new_iter_str, "\n")
    f = open(path_summary, "a")
    if os.stat(path_summary).st_size == 0:
        f.write(",".join(csv_header) + "\n")

    f.write(f"{new_iter_str},{N_ITER}\n")
    f.close()
