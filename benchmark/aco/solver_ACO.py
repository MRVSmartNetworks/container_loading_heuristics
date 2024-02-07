#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
import statistics as st

from benchmark.aco.sub.utilities import *
from benchmark.aco.sub.stack import Stack
from benchmark.aco.sub.config import *

if SLICES:
    from benchmark.aco.sub.aco_bin_packing_slices import ACO
else:
    from benchmark.aco.sub.aco_bin_packing import ACO
from sol_representation import *


class SolverACO:
    def __init__(self):
        """
        SolverACO
        ----------------------------------------------------------------
        ANT COLONY OPTIMIZATION
        -----------------------------------------------------------------
        From solver/group23/
        """
        self.name = "solverACO"

        # Id of the vehicle used for solution format
        self.id_vehicle = 0

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

    def solve(self, df_items, df_vehicles, sol_file_name, time_limit=30 * 60):
        """
        solve
        -----
        #### INPUT PARAMETERS:
        - df_items: dataframe containing all the items
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        - sol_file_name: solution file name obtained from the specific database
        - time_limit: time limit imposed externally to conclude the iteration after a certain amount of time
        """
        t1 = time.time()
        # Max value that will be decreased with the best solution value
        # and other parameters for loop test
        best_cost = float("inf")
        best_N_Truck = 0
        best_alpha = 1
        best_beta = 1
        time_spent = 0
        iter_time = []
        test_ind = 1

        # If the parameters must be tested alpha and beta have to be resetted
        # and only in this case the file must be opened
        if TEST_PAR == True:
            globals()["ALPHA"] = 0
            globals()["BETA"] = 0
            f = open(PARAM_FILE, "w")

        # Create N_SOL_ITER solution and take only the one with the lowest cost
        for iter in range(N_SOL_ITER):
            print(f"\nSolution iteration number: {iter+1}\n")

            # When the code is runned normally (one or multiple solution)
            # only this part is executed
            if TEST_PAR == False:
                (
                    df_sol,
                    tot_cost,
                    pattern_list,
                    pattern_info,
                    time_spent,
                    out_df_items,
                    stackInfo,
                ) = self.solver(df_items, df_vehicles, time_limit)
                iter_time.append(time_spent)

                # Adjust the value when a better solution is found
                if tot_cost < best_cost:
                    best_sol = df_sol
                    best_cost = tot_cost
                    best_N_Truck = df_sol["idx_vehicle"].nunique()

            # This part of code is runned only when the parameters alpha and beta must be tested
            while TEST_PAR:
                if ALPHA < 1:
                    globals()["ALPHA"] += 0.25
                else:
                    globals()["ALPHA"] = 0.25

                if BETA < 1 and ALPHA == 0.25:
                    globals()["BETA"] += 0.25

                # Multiple iteratio every alpha and beta to obtain average and std of the total cost
                list_cost = []
                for _ in range(PARAM_ITER):
                    print(f"Iteration number : {test_ind}")
                    test_ind += 1
                    (
                        df_sol,
                        tot_cost,
                        pattern_list,
                        pattern_info,
                        time_spent,
                        out_df_items,
                        stackInfo,
                    ) = self.solver(df_items, df_vehicles, time_limit)
                    list_cost.append(tot_cost)
                    iter_time.append(time_spent)

                # Average cost with this specific alpha and beta and saving it on the file
                tot_cost = sum(list_cost) / len(list_cost)
                f.write(
                    f"ALPHA = {ALPHA}   BETA = {BETA}   Avg_cost = {tot_cost}   std = {st.stdev(list_cost)}   best_value = {min(list_cost)}\n"
                )

                # Adjust the value when a better solution is found
                if tot_cost < best_cost:
                    best_sol = df_sol
                    best_cost = tot_cost
                    best_N_Truck = df_sol["idx_vehicle"].nunique()
                    best_alpha = ALPHA
                    best_beta = BETA

                # Here the program will enter only when the test of the parameters is completed
                # so the best value are saved into the file, then it is closed
                if BETA == 1 and ALPHA == 1:
                    globals()["TEST_PAR"] = False
                    print(f"Best ALPHA: {best_alpha} \nBest BETA: {best_beta}")
                    f.write(f"\n\nBest ALPHA: {best_alpha} \nBest BETA: {best_beta}\n")
                    f.close()

        # Final information of the best solution
        # Only when multiple iteration are done the average and std of the time are evaluated
        if len(iter_time) > 1:
            print(
                f"Average time spent by every iteration: {sum(iter_time)/len(iter_time)} s"
            )
            print(f"With standard deviation: {st.stdev(iter_time)} s")
        print(f"\nThe best solution objective value is: {best_cost}")
        print(f"\nN truck: {best_N_Truck}")

        # Saving the best dataframe solution in the file
        best_sol.to_csv(os.path.join("results", sol_file_name), index=False)

        t = round(time.time() - t1, 2)
        return t

    def solver(self, df_items, df_vehicles, time_limit):
        """
        solver
        ------

        #### INPUT PARAMETERS:
        - df_items: dataframe containing all the items
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        - time_limit: time limit imposed externally to conclude the iteration after a certain amount of time
        """
        st_time = time.time()

        # Id of the vehicle used for solution format
        self.id_vehicle = 0
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
        # Used for randomizing the items dataframe
        # df_items = df_items.sample(frac=1).reset_index()

        if SINGLE_PLOT:
            df_items_or = df_items
            df_vehicles_or = df_vehicles

        # Improvement od the vehicle dataframe with useful data
        # as area, volume and efficiency
        df_vehicles = df_vehicles_improv(df_vehicles)

        # Creation of a dataframe containing information related
        # to the items and useful for the creation of stacks
        df_items, stackInfo = stackInfo_creation_weight(df_items)
        out_df_items = df_items.copy()

        # Initialization of the ACO object
        aco = ACO(stackInfo, alpha=ALPHA, beta=BETA, n_ants=N_ANTS, n_iter=N_ITER)
        more_items = True
        tot_cost = 0

        # Loop parameters
        # Area and weight are set in this way to choose the efficient vehicle at the first cicle
        prev_area_ratio = 1
        prev_weight_ratio = 0
        i = 0
        id_prev_truck = None
        pattern_list = []
        pattern_info = []
        Npattern = 0
        # Loop until there are no more items left
        while more_items:
            # Decision for the most convenient vehicle
            vehicle, last_iter = self.vehicle_decision(
                df_vehicles, df_items, prev_area_ratio, prev_weight_ratio
            )
            aco.getVehicle(vehicle)

            if id_prev_truck is not None:
                n_stacks_before = len(aco.stack_lst)

            n_items_before = df_items.shape[0]

            # Create the stacks given a vehicle and give it to ACO
            if vehicle.id_truck != id_prev_truck:
                stack_list, stack_quantity = buildStacks(vehicle, df_items, stackInfo)
                aco.getStacks(stack_list, stack_quantity)
            else:
                update_stack_lst(bestAnt, aco.stack_lst, aco.stack_quantity)

            n_stacks_after = len(aco.stack_lst)
            if DEBUG_LOCAL and id_prev_truck is not None:
                if vehicle.id_truck == id_prev_truck:
                    assert n_stacks_before - n_stacks_after == len(bestAnt)
                    print("Stacks add up")

            id_prev_truck = vehicle.id_truck

            # Check if there are stacks left
            if VERB_LOCAL:
                print("Truck: ", self.id_vehicle)

            # Method to solve the 2D bin packing problem
            bestAnts, bestAreas, bestWeights = aco.aco_2D_bin(last_iter=last_iter)
            # Pick only the first one among the best
            bestAnt = bestAnts[0]
            prev_weight_ratio = bestWeights[0]
            prev_area_ratio = bestAreas[0]

            column = np.zeros(len(stackInfo))

            for stack in bestAnt:
                # TODO: use code to state
                column[int(stack.stack_code)] += len(stack.items)

                pattern_info.append(
                    {
                        "pattern": Npattern,
                        "vehicle": vehicle["id_truck"],
                        "stack_code": int(stack.stack_code),
                        "stack_Nitems": stack.n_items,
                        "stack_weight": stack.weight,
                        "x_origin": stack.vertexes[0][0],
                        "y_origin": stack.vertexes[0][1],
                        "orient": stack.orient,
                    }
                )

            Npattern += 1

            pattern_list.append(
                {
                    "pattern": column,
                    "vehicle": vehicle["id_truck"],
                    "area": bestAreas[0],
                }
            )

            self.solUpdate(bestAnt, vehicle)

            # Remove the items already added to the solution
            df_items = df_items[df_items.id_item.isin(self.sol["id_item"]) == False]
            tot_cost += vehicle["cost"]

            n_items_after = df_items.shape[0]

            if DEBUG_LOCAL:
                # Get items placed in each stack of the best ant
                n_items_placed = sum([s.n_items for s in bestAnt])
                assert (
                    n_items_before - n_items_after == n_items_placed
                ), f"Items placed = {n_items_placed}, but {n_items_before - n_items_after} have been removed"

            # Boolean to print every 2D vehicle plot
            if SINGLE_PLOT:
                orthogonal_plane(
                    df_items_or,
                    df_vehicles_or,
                    pd.DataFrame.from_dict(self.sol),
                    idx_vehicle=i,
                )
                i += 1

            if df_items.empty or (time.time() - st_time) >= time_limit:
                if time_limit != 0 or df_items.empty:
                    more_items = False
                    if (time.time() - st_time) >= time_limit:
                        print("[!] - Time limit exceeded!")

        # Printing step of the solution
        df_sol = pd.DataFrame.from_dict(self.sol)
        print("\nN trucks = ", df_sol["idx_vehicle"].nunique())
        print("Tot cost: ", tot_cost)
        print("Tot items: ", len(self.sol["id_item"]))

        # Evaluating the time taken by every iteration
        time_spent = time.time() - st_time
        print(f"\nTime: {time_spent} s")

        # Return the dataframe solution and its cost to check the best solution among all the iteration
        return (
            df_sol,
            tot_cost,
            pattern_list,
            pattern_info,
            time_spent,
            out_df_items,
            stackInfo,
        )

    def solver_end(self, df_items, df_vehicles, sol):
        """
        solver
        ------

        #### INPUT PARAMETERS:
        - df_items: dataframe containing all the items
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        - time_limit: time limit imposed externally to conclude the iteration after a certain amount of time
        """
        st_time = time.time()

        # Id of the vehicle used for solution format
        self.id_vehicle = (sol["idx_vehicle"][-1]) + 1
        self.sol = sol

        # Improvement od the vehicle dataframe with useful data
        # as area, volume and efficiency
        df_vehicles = df_vehicles_improv(df_vehicles)

        # Creation of a dataframe containing information related
        # to the items and useful for the creation of stacks
        df_items, stackInfo = stackInfo_creation_weight(df_items)

        # Initialization of the ACO object
        aco = ACO(stackInfo, alpha=ALPHA, beta=BETA, n_ants=N_ANTS, n_iter=N_ITER)
        more_items = True
        tot_cost = 0

        # Loop parameters
        # Area and weight are set in this way to choose the efficient vehicle at the first cicle
        area_ratio = 1
        weightRatio = 0
        i = 0
        id_prev_truck = None
        # Loop until there are no more items left
        while more_items:
            # Decision for the most convenient vehicle
            vehicle, last_iter = self.vehicle_decision(
                df_vehicles, df_items, area_ratio, weightRatio
            )
            aco.getVehicle(vehicle)

            if id_prev_truck is not None:
                n_stacks_before = len(aco.stack_lst)

            n_items_before = df_items.shape[0]

            # Create the stacks given a vehicle and give it to ACO
            if vehicle.id_truck != id_prev_truck:
                stack_list, stack_quantity = buildStacks(vehicle, df_items, stackInfo)
                aco.getStacks(stack_list, stack_quantity)
            else:
                update_stack_lst(bestAnt, aco.stack_lst, aco.stack_quantity)

            n_stacks_after = len(aco.stack_lst)
            if DEBUG_LOCAL and id_prev_truck is not None:
                if vehicle.id_truck == id_prev_truck:
                    assert n_stacks_before - n_stacks_after == len(bestAnt)
                    print("Stacks add up")

            id_prev_truck = vehicle.id_truck

            # Check if there are stacks left
            if VERB_LOCAL:
                print(f"Truck: {self.id_vehicle}")

            # Method to solve the 2D bin packing problem
            bestAnts, bestAreas, bestWeights = aco.aco_2D_bin(last_iter=last_iter)
            bestAnt = bestAnts[0]

            self.solUpdate(bestAnt, vehicle)

            # Remove the items already added to the solution
            df_items = df_items[df_items.id_item.isin(self.sol["id_item"]) == False]

            if DEBUG_LOCAL:
                n_items_after = df_items.shape[0]
                # Get items placed in each stack of the best ant
                n_items_placed = sum([s.n_items for s in bestAnt])
                assert (
                    n_items_before - n_items_after == n_items_placed
                ), f"Items placed = {n_items_placed}, but {n_items_before - n_items_after} have been removed"

            if df_items.empty:
                more_items = False

        # Printing step of the solution
        df_sol = pd.DataFrame.from_dict(self.sol)
        print("\nN trucks = ", df_sol["idx_vehicle"].nunique())
        print("Tot items: ", len(self.sol["id_item"]))

        # Evaluating the time taken by every iteration
        time_spent = time.time() - st_time
        print(f"\nTime: {time_spent} s")

        # Return the dataframe solution and its cost to check the best solution among all the iteration
        return df_sol

    def solUpdate(self, bestAnt, vehicle):
        """
        solUpdate
        -----
        ### Input parameters:
            - bestAnt: the best ant output of the ACO solution of the 2D bin packing problem
            - vehicle: the vehicle of the solution
        """

        lenSol = len(self.sol["id_stack"])
        for i, stack in enumerate(bestAnt):
            z_origin = 0

            # Saving all the item with their information in the dictionary solution
            for j in range(len(stack.items)):
                self.sol["type_vehicle"].append(vehicle["id_truck"])
                self.sol["idx_vehicle"].append(self.id_vehicle)
                self.sol["id_stack"].append(f"S{i+lenSol}")
                self.sol["id_item"].append(stack.items[j])
                self.sol["x_origin"].append(stack.vertexes[0][0])
                self.sol["y_origin"].append(stack.vertexes[0][1])
                self.sol["z_origin"].append(z_origin)
                self.sol["orient"].append(stack.orient)
                z_origin += stack.h_items[j]

        # Update of the vehicle id
        self.id_vehicle += 1

    def vehicle_decision(self, df_vehicles, items_left, area_ratio, weightRatio):
        """
        vehicle_decision
        ----------------
        Choose the best truck to be filled in the next iteration of ACO bin packing.

        ### Input parameters:
            - df_vehicles: dataframe containing all the different types of trucks that can be choose
            - items_left: dataframe containing all of the remaining items
            - area_ratio: area ratio of the previous truck
            - weightRatio: weight ratio of the previous truck
        ### Output parametes:
            - the choice of the vehicle to use based on the type of vehicles and items left
            - last_iter : boolean that is returned to true if this truck can be the last one \n
                            based on volume and weight of the remaining items

        """

        last_iter = False
        # evaluation of important parameters for the truck choice
        tot_weight = sum(items_left["weight"])
        tot_vol = sum(items_left["length"] * items_left["width"] * items_left["height"])

        # Select the max weight, height and desity of the left items.
        # Particularly useful for the choice of the last trucks.
        maxIt_W = max(items_left["weight"])
        maxIt_H = max(items_left["height"])
        maxIt_D = max(
            (items_left["weight"] / items_left["width"] * items_left["height"])
        )

        # List of the trucks that can respect all the constrain of the remained items.
        Usable_vehicles = df_vehicles[df_vehicles.max_weight_stack >= maxIt_W]
        Usable_vehicles = Usable_vehicles[Usable_vehicles.height >= maxIt_H]
        Usable_vehicles = Usable_vehicles[Usable_vehicles.max_density >= maxIt_D]

        # This exception happen when exist one or more items that singularly cannot be put in any truck
        if Usable_vehicles.empty:
            raise ValueError(
                "No vehicles can carry certain items due to constrain not respected"
            )

        # Possible last vehicles
        Last_vehicles = Usable_vehicles[
            Usable_vehicles.max_weight >= (tot_weight * 1.1)
        ]
        Last_vehicles = Last_vehicles[
            Last_vehicles.volume >= (tot_vol * 1.25)
        ]  # *1.25 is to have some margin

        # Best efficiency vehicles
        BestEff_vehicles = Usable_vehicles.sort_values(
            by=["tot_eff_ratio"], ascending=False
        ).reset_index()
        BestWeight_vehicles = Usable_vehicles.sort_values(
            by=["weight_eff"], ascending=False
        ).reset_index()

        # Case in which some vehicles can carry all the items remained
        if Last_vehicles.empty == False:
            Last_vehicles = Last_vehicles.sort_values(
                by=["cost"]
            )  # Little cost choosen
            last_iter = True
            return Last_vehicles.iloc[0], last_iter
        else:
            # If not take the vehicles with the best efficiency or weight ratio
            # Best weight ratio is taken only when the last vehicle have low
            # area covered and a weightRatio satured
            if weightRatio > 0.95 and area_ratio < 0.95:
                if (
                    BestEff_vehicles.at[0, "area"] - BestWeight_vehicles.at[0, "area"]
                ) < (1 - area_ratio) * BestEff_vehicles.at[0, "area"]:
                    return BestWeight_vehicles.iloc[0], last_iter
                else:
                    return BestEff_vehicles.iloc[0], last_iter
            else:
                # In any other cases take the best efficient vehicle in general
                return BestEff_vehicles.iloc[0], last_iter
