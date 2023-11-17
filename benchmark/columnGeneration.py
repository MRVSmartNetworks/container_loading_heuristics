#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import pandas as pd
import os

try:
    from .masterProblem import MasterProblem
    from .sub.utilities import stackInfo_creation_weight, buildStacks, buildSingleStack
    from .sub.aco_bin_packing import ACO
    from .sub.config import ALPHA, BETA, N_ANTS, N_ITER
    from .sub.configCG import N_INIT_COLS, N_COLS, TIME_LIMIT
except ImportError:
    from masterProblem import MasterProblem
    from sub.utilities import stackInfo_creation_weight, buildStacks, buildSingleStack
    from sub.aco_bin_packing import ACO
    from sub.config import ALPHA, BETA, N_ANTS, N_ITER
    from sub.configCG import N_INIT_COLS, N_COLS, TIME_LIMIT

# from sub.aco_bin_packing import ACO


class columnGeneration:
    """
    columnGeneration
    -----------------------------------------
    Implementation of column generation using as subproblem the ACO
    to solve the 2d bin packing problem. The stacks are created using
    a greedy approach and are the starting point for the ACO meta-heuristic.

    #### INPUT PARAMETERS
    - df_vehicles: pandas dataframe containing different types of trucks
    - df_items: pandas dataframe containing different items

    #### METHODS
    - solve
    - generateColumns
    - genereateInitialColumns
    """

    def __init__(
        self,
    ):
        self.name = "columnGeneration"
        self.df_vehicles = None
        self.df_items = None
        self.stackInfo = None
        self.items_type = 0
        self.n_vehicles = 0

        # Creation of the stacks for each vehicle
        self.stack_lst = {}
        self.stack_quantity = {}

        # List containing dictionaries of patterns, vehicle, area
        self.pattern_list = []

        # List containing dictionaries of patte
        self.pattern_info = []
        self.n_pattern = 0

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

    def solve(self, df_items, df_vehicles, sol_file_name=None):
        t_start = time.time()

        self.df_vehicles = df_vehicles
        self.df_items, self.stackInfo = stackInfo_creation_weight(df_items)
        self.items_type = len(self.stackInfo)
        self.n_vehicles = len(self.df_vehicles)

        # Evaluate the number of items per type in df_items
        self.n_items_type = np.zeros(self.items_type)
        for i in range(self.items_type):
            self.n_items_type[i] = len(
                self.df_items.loc[self.df_items["stackability_code"] == i]
            )

        # Ant Colony Optimizazion initialization
        self.aco = ACO(
            self.stackInfo, alpha=ALPHA, beta=BETA, n_ants=N_ANTS, n_iter=N_ITER
        )

        # Creation of stacks for each vehicle
        self.stacksBuilding()

        # Generate initial columns for each vehicle
        initColumns = self.generateInitialColumns(n_cols=N_INIT_COLS)

        # Definition of the master problem and build of the model
        master = MasterProblem(initColumns, self.df_vehicles, self.df_items)

        _iter = 0
        duals = 0
        ind_vehicle = 0
        t_start = time.time()
        while (time.time() - t_start) <= TIME_LIMIT:
            try:
                # Solve relaxed master
                objVal = master.solveRelaxedModel(_iter)
                if objVal:
                    duals = master.getDuals()
                # FIXME: Dummy solution
                if _iter != 0:
                    ind_vehicle += 1
                    if ind_vehicle >= self.n_vehicles:
                        ind_vehicle = 0

                vehicle = self.df_vehicles.iloc[ind_vehicle].to_dict()

                # Solve the Ant Colony optimization
                newColumns = self.generateColumns(
                    n_cols=N_COLS, duals=duals, vehicle=vehicle
                )

                # Add new columns to master
                master.addColumns(vehicle["cost"], newColumns)
                _iter += 1
            except KeyboardInterrupt:
                break

        master.solveModel(file_name="benchmark/results/model.lp")
        print(f"\nTotal elapsed time: {round(time.time() - t_start, 2)} s")

        # Print patterns used
        print("\nPATTERN USED\n")
        vars = master.getVars()

        df_items_copy = self.df_items.copy()
        for i, v in enumerate(vars):
            if v.X > 0:
                print(
                    f"{int(v.X)} vehicles {self.pattern_list[i]['vehicle']}"
                    f" with area {round(self.pattern_list[i]['area'], 3)}"
                    f" Pattern: {self.pattern_list[i]['pattern']}"
                )
                df_items_copy = self.generateSolution(int(v.X), i, df_items_copy)

        print(f"\nItems not inserted: {len(df_items_copy)}")
        df_sol = pd.DataFrame.from_dict(self.sol)
        df_sol.to_csv(os.path.join("results", sol_file_name), index=False)

    def generateColumns(self, n_cols, duals, vehicle):
        """
        generateColumns
        ---------------------------------------------
        Method to generate a given number of columns for a given
        vehicle using the ACO and the dual variables coming from
        the master problem.
        """
        columns = np.zeros((self.items_type, n_cols))
        # Give a vehicle to ACO and the specific stack list
        self.aco.getVehicle(vehicle)
        self.aco.getStacks(
            self.stack_lst[vehicle["id_truck"]],
            self.stack_quantity[vehicle["id_truck"]],
        )

        # Solve the 2D bin packing problem
        bestAnts, bestAreas = self.aco.aco_2D_bin(n_bestAnts=n_cols, dualVars=duals)
        for j, ant in enumerate(bestAnts):
            for stack in ant:
                # TODO: use code to state
                columns[stack.stack_code][j] += len(stack.items)
                self.pattern_info.append(
                    {
                        "pattern": self.n_pattern,
                        "vehicle": vehicle["id_truck"],
                        "stack_code": stack.stack_code,
                        "stack_Nitems": stack.n_items,
                        "x_origin": stack.vertexes[0][0],
                        "y_origin": stack.vertexes[0][1],
                        "orient": stack.orient,
                    }
                )

            self.n_pattern += 1

            self.pattern_list.append(
                {
                    "pattern": columns[:, j],
                    "vehicle": vehicle["id_truck"],
                    "area": bestAreas[j],
                }
            )

        return columns

    def generateInitialColumns(self, n_cols):
        """
        generateInitialColumns
        ---------------------------------------------
        Method to generate the initial set of columns for each type
        of vehicle in df_vehicles. The columns are generated using
        the ACO.
        """
        print("\nGENERATING THE INITIAL SET OF COLUMNS FOR EACH VEHICLE...\n")
        t_start = time.time()
        columns = np.zeros((self.items_type, n_cols * self.n_vehicles))
        _iter = 0
        for i in range(self.n_vehicles):
            vehicle = self.df_vehicles.iloc[i].to_dict()
            columns_vehicle = self.generateColumns(
                n_cols=N_INIT_COLS, duals=0, vehicle=vehicle
            )
            print(f"Solution found for vehicle {vehicle['id_truck']}")

            columns[:, _iter : _iter + N_INIT_COLS] = columns_vehicle

            _iter += N_INIT_COLS

        print(
            f"\nElapsed time to create initial columns: "
            f"{round(time.time() - t_start, 2)} s"
        )
        return columns

    def stacksBuilding(self):
        # Creation of the stacks for each vehicle
        t_start = time.time()
        print("BUILDING STACKS FOR EACH VEHICLE\n")
        for i in range(self.n_vehicles):
            id_truck = self.df_vehicles.iloc[i]["id_truck"]
            self.stack_lst[id_truck], self.stack_quantity[id_truck] = buildStacks(
                self.df_vehicles.iloc[i], self.df_items, self.stackInfo
            )
            print(f"Stack for vehicle {id_truck} created")
        print(f"\nElapsed time to create stacks: {round(time.time() - t_start, 2)} s")

    def generateSolution(self, nTruck, index, df_items):
        pattern = [row for row in self.pattern_info if row["pattern"] == index]
        vehicle = self.df_vehicles[
            self.df_vehicles["id_truck"] == self.pattern_list[index]["vehicle"]
        ].iloc[0]
        for i in range(nTruck):
            tot_weight = 0
            for j in range(len(pattern)):
                stack, df_items, tot_weight = buildSingleStack(
                    df_items,
                    self.stackInfo,
                    vehicle,
                    pattern[j]["stack_Nitems"],
                    pattern[j]["stack_code"],
                    pattern[j]["orient"],
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
                # Update of the vehicle id
            self.id_vehicle += 1

        return df_items
