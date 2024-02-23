#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from benchmark import SolverACO, columnGeneration, masterAco, solverORTools, ExactSolver
from sol_representation import *
import numpy as np
import random
from benchmark.aco.solver_ACO import Our_exception

N_ITER = 1
CHECKPOINT_PATH = "./results/checkpoints/"
SUMMARY_PATH = "./results/summaries/"
ONLY_STATS = False
IVANCIC = True


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
    path_checkpoint: str, path_summary: str, dataset_name: str
) -> None:
    df_checkp = pd.read_csv(path_checkpoint, sep=",")
    avg_cost = df_checkp["cost"].mean()
    std_cost = df_checkp["cost"].std()
    min_cost = df_checkp["cost"].min()
    avg_t = df_checkp["time"].mean()
    std_t = df_checkp["time"].std()

    print(
        f"\n{dataset_name},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f}\n"
    )
    f = open(path_summary, "a")
    if os.stat(path_summary).st_size == 0:
        f.write(
            f"dataset,avg_cost,std_cost,minimum_cost,avg_time,std_time,iterations\n"
        )

    f.write(
        f"{dataset},{(avg_cost):.2f},{std_cost:.2f},{min_cost},{avg_t:.2f},{std_t:.2f},{N_ITER}\n"
    )
    f.close()


if __name__ == "__main__":
    datasets = [
        # "datasetA",
        # "datasetB",
        # "datasetC",
        "datasetD",
        "datasetE",
        "datasetF",
        # "datasetG",
        # "datasetH",
        # "datasetI",
        # "datasetJ",
        # "dataset_small",
        # "easy_dataset",
        # "MODdatasetG"
    ]

    ivancic_datasets = [
        # "thpack9_1",
        # "thpack9_2",
        # "thpack9_3",
        # "thpack9_4",
        # "thpack9_5",
        # "thpack9_6",
        # "thpack9_7",
        # "thpack9_8",
        # "thpack9_9",
        # "thpack9_10",
        # "thpack9_11",
        # "thpack9_12",
        # "thpack9_13",
        # "thpack9_14",
        # "thpack9_15",
        # "thpack9_16",
        # "thpack9_17",
        # "thpack9_18",
        # "thpack9_19",
        # "thpack9_20",
        # "thpack9_21",
        # "thpack9_22",
        # "thpack9_23",
        # "thpack9_24",
        # "thpack9_25",
        # "thpack9_26",
        # "thpack9_27",
        # "thpack9_28",
        # "thpack9_29",
        # "thpack9_30",
        # "thpack9_31",
        # "thpack9_32",
        # "thpack9_33",
        # "thpack9_34",
        # "thpack9_35",
        "thpack9_36",
        "thpack9_37",
        "thpack9_38",
        "thpack9_39",
        "thpack9_40",
        "thpack9_41",
        "thpack9_42",
        "thpack9_43",
        "thpack9_44",
        "thpack9_45",
        "thpack9_46",
        "thpack9_47",
    ]

    solver = masterAco()
    for d, dataset in enumerate(ivancic_datasets):
        df_items = pd.read_csv(
            os.path.join(".", "data", dataset, "items.csv"),
        )
        df_vehicles = pd.read_csv(
            os.path.join(".", "data", dataset, "vehicles.csv"),
        )
        width = df_vehicles["width"].iloc[0]
        length = df_vehicles["length"].iloc[0]
        height = df_vehicles["height"].iloc[0]
        sol_file_name = f"{solver.name}_{dataset}_sol.csv"
        # measure execution time

        print(f"{dataset}\n============================================\n")
        if not ONLY_STATS:
            for i in range(N_ITER):
                if not IVANCIC:
                    print(f"++++++++++++++++++ Iteration {i} ++++++++++++++++++")
                    solver = masterAco()

                    t = solver.solve(
                        df_items, df_vehicles, sol_file_name, time_limit=300
                    )
                    # read dataframe solution
                    df_sol = pd.read_csv(
                        os.path.join("results", sol_file_name),
                    )
                    df_sol.to_csv(
                        f"./results/{dataset}/{sol_file_name}_{random.randint(0,100)}"
                    )
                    # Check if solution is correct
                    try:
                        of = sol_check(df_sol, df_vehicles, df_items)
                    except Exception as e:
                        of = f"{e}"
                        print(of)
                        continue
                    # Evaluate the total cost
                    cost = eval_cost(df_sol, df_vehicles)
                    print(f"\nIteration {i}: cost={cost}, time={t}\n")

                    # save checkpoint
                    f_checkp = open(
                        CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv", "a"
                    )
                    if (
                        os.stat(
                            CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv"
                        ).st_size
                        == 0
                    ):
                        f_checkp.write(f"cost,time\n")
                    f_checkp.write(f"{cost},{t}\n")

                    f_checkp.close()
                else:
                    for j in range(6):
                        if j == 1:
                            df_vehicles["width"].iloc[0] = height
                            df_vehicles["height"].iloc[0] = width
                        elif j == 2:
                            df_vehicles["length"].iloc[0] = width
                            df_vehicles["width"].iloc[0] = length
                            df_vehicles["height"].iloc[0] = height
                        elif j == 3:
                            df_vehicles["length"].iloc[0] = width
                            df_vehicles["width"].iloc[0] = height
                            df_vehicles["height"].iloc[0] = length
                        elif j == 4:
                            df_vehicles["length"].iloc[0] = height
                            df_vehicles["width"].iloc[0] = length
                            df_vehicles["height"].iloc[0] = width
                        elif j == 5:
                            df_vehicles["length"].iloc[0] = height
                            df_vehicles["width"].iloc[0] = width
                            df_vehicles["height"].iloc[0] = length
                        print(f"++++++++++++++++++ Iteration {i} ++++++++++++++++++")
                        solver = masterAco()

                        try:
                            t = solver.solve(
                                df_items, df_vehicles, sol_file_name, time_limit=300
                            )
                            # read dataframe solution
                            df_sol = pd.read_csv(
                                os.path.join("results", sol_file_name),
                            )
                            if not os.path.exists(f"./results/{dataset}"):
                                os.mkdir(f"./results/{dataset}")
                            df_sol.to_csv(
                                f"./results/{dataset}/{sol_file_name}_{random.randint(0,100)}"
                            )
                            # Check if solution is correct
                            try:
                                of = sol_check(df_sol, df_vehicles, df_items)
                            except Exception as e:
                                of = f"{e}"
                                print(of)
                                continue
                            # Evaluate the total cost
                            cost = eval_cost(df_sol, df_vehicles)
                            print(f"\nIteration {i}: cost={cost}, time={t}\n")

                            # save checkpoint
                            f_checkp = open(
                                CHECKPOINT_PATH
                                + f"{solver.name}_{dataset}_checkpoint.csv",
                                "a",
                            )
                            if (
                                os.stat(
                                    CHECKPOINT_PATH
                                    + f"{solver.name}_{dataset}_checkpoint.csv"
                                ).st_size
                                == 0
                            ):
                                f_checkp.write(f"cost,time,length,width,height\n")
                            f_checkp.write(
                                f"{cost},{t},{df_vehicles['length'].iloc[0]},{df_vehicles['width'].iloc[0]},{df_vehicles['height'].iloc[0]}\n"
                            )

                            f_checkp.close()
                        except Our_exception:
                            print(
                                f"\nTruck structure number {j+1} of {dataset} can not be created.\n"
                            )

        # stats_properties(
        #     CHECKPOINT_PATH + f"{solver.name}_{dataset}_checkpoint.csv",
        #     SUMMARY_PATH + f"{solver.name}_summary.csv",
        #     dataset,
        # )
