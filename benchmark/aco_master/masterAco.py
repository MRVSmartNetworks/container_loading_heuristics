#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Optional

import pandas as pd
from benchmark.aco.solver_ACO import SolverACO
from benchmark.aco.sub.utilities import buildSingleStack, stackInfo_creation_weight

from sol_representation import *

from .masterProblem import MasterProblem
from .sub.config import ONE_LAYER

N_PAT_SKIP = 0


class masterAco:
    """
    acoMaster
    ---------
    Implementation of the column gen master problem using as subproblem the ACO
    to solve the 2d bin packing problem. The stacks are created using
    a greedy approach and are the starting point for the ACO meta-heuristic.

    #### INPUT PARAMETERS
    - df_vehicles: pandas dataframe containing different types of trucks
    - df_items: pandas dataframe containing different items

    #### METHODS
    - solve
    - genereateSolutions
    """

    def __init__(self):
        self.name = "masterAco"
        self.df_vehicles = None
        self.df_items = None

        self.sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }
        self.id_vehicle = 0
        self.id_stack = 1

    def generateSolution(
        self, nTruck, index, df_items, stackInfo, patterInfo, pattern_list
    ):
        global N_PAT_SKIP
        pattern = [row for row in patterInfo if row["pattern"] == index]
        n_stack = len(pattern)
        vehicle = self.df_vehicles[
            self.df_vehicles["id_truck"] == pattern_list[index]["vehicle"]
        ].iloc[0]
        for i in range(nTruck):
            tot_weight = 0
            n_it = 0
            true_n_it = 0
            items_lost = False
            jump_pattern = False

            for stack_code, n_it in enumerate(pattern_list[index]["pattern"]):
                n_it = int(n_it)
                if len(df_items[df_items["stackability_code"] == stack_code]) < n_it:
                    jump_pattern = True
                    break
            if not jump_pattern:
                for j in range(n_stack):
                    avg_stack_W = (vehicle.max_weight / n_stack) * j
                    stack, df_items, tot_weight, items_lost = buildSingleStack(
                        df_items,
                        stackInfo,
                        vehicle,
                        pattern[j]["stack_Nitems"],
                        pattern[j]["stack_code"],
                        pattern[j]["orient"],
                        avg_stack_W,
                        tot_weight,
                    )

                    z_origin = 0
                    # Saving all the item with their information in the dictionary solution
                    for y in range(stack.n_items):
                        self.sol["type_vehicle"].append(vehicle["id_truck"])
                        self.sol["idx_vehicle"].append(self.id_vehicle)
                        self.sol["id_stack"].append(f"S{self.id_stack}")
                        self.sol["id_item"].append(stack.items[y])
                        self.sol["x_origin"].append(pattern[j]["x_origin"])
                        self.sol["y_origin"].append(pattern[j]["y_origin"])
                        self.sol["z_origin"].append(z_origin)
                        self.sol["orient"].append(pattern[j]["orient"])
                        z_origin += stack.h_items[y]
                    self.id_stack += 1
                    n_it += stack.n_items
                    true_n_it += pattern[j]["stack_Nitems"]
                    # if items_lost:
                    #     print("Items lost")
                    # Update of the vehicle id
                self.id_vehicle += 1
                # print(f"\nTot_weight = {round(tot_weight,1)} N_it = {n_it}\nVel_weight = {vehicle.max_weight} T_Nit = {true_n_it} StackCode items are finished? -> {items_lost}\n\n")
            else:
                N_PAT_SKIP += 1
        return df_items

    def solve(self, df_items, df_vehicles, sol_file_name, *, time_limit=300, pass_t_aco: Dict[str, float] = {}):
        t1 = time.time()
        self.df_items = df_items
        self.df_vehicles = df_vehicles

        if ONE_LAYER:
            self.df_vehicles["height"] = 1000
            # df_vehicles["max_weight"] = 100000
            self.df_vehicles["max_density"] = 100000
            # df_vehicles["max_weight_stack"] = 100000
            self.df_items["height"] = 1000

        t2 = time.time()
        sol_aco = SolverACO()
        (
            df_sol,
            totCost,
            pattern_list,
            patterInfo,
            time_spent,
            self.df_items,
            stackInfo,
        ) = sol_aco.solver(self.df_items, self.df_vehicles, 60000)
        tACO = round(time.time() - t2, 2)

        pass_t_aco["tACO"] = tACO

        of = sol_check(df_sol, self.df_vehicles, self.df_items)
        # print(of)

        master = MasterProblem(pattern_list, self.df_vehicles, self.df_items)
        master.solveModel(file_name=f"benchmark/aco_master/results/{sol_file_name}.lp")

        df_items_copy = self.df_items.copy()
        vars = master.getVars()
        for i, v in enumerate(vars):
            if v.X > 0:
                print(
                    f"{int(v.X)} vehicles {pattern_list[i]['vehicle']}"
                    f" with area {round(pattern_list[i]['area'], 3)}"
                    f" Pattern: {pattern_list[i]['pattern']}"
                )
                df_items_copy = self.generateSolution(
                    int(v.X),
                    i,
                    df_items_copy,
                    stackInfo,
                    patterInfo,
                    pattern_list,
                )

        if len(df_items_copy) != 0:
            print(f"\nItems not inserted: {len(df_items_copy)}")
            df_sol = sol_aco.solver_end(df_items_copy, df_vehicles, self.sol)
        df_sol.to_csv(os.path.join("results", sol_file_name), index=False)

        print(f"Number of pattern not inserted:", N_PAT_SKIP)
        print("Optimal value of solverACO:", totCost)
        t = round(time.time() - t1, 2)
        return t, totCost
