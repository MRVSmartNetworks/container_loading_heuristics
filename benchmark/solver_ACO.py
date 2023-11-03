#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
import statistics as st

from benchmark.sub.utilities import *
from benchmark.sub.stack import Stack

# from solver.benchmark.aco.aco_bin_packing import ACO
from benchmark.sub.aco_bin_packing_slices import ACO
from sol_representation import *
from benchmark.sub.config import *


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
                df_sol, tot_cost, time_spent = self.solver(
                    df_items, df_vehicles, time_limit
                )
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
                    df_sol, tot_cost, pattern_list, time_spent = self.solver(
                        df_items, df_vehicles, time_limit
                    )
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
        df_items, stackInfo = stackInfo_creation(df_items)

        # Initialization of the ACO object
        aco = ACO(stackInfo, alpha=ALPHA, beta=BETA, n_ants=N_ANTS, n_iter=N_ITER)
        more_items = True
        totCost = 0

        # Loop parameters
        # Area and weight are set in this way to choose the efficient vehicle at the first cicle
        area_ratio = 1
        weightRatio = 0
        i = 0
        id_prev_truck = None
        pattern_list = []
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
                stack_list, stack_quantity = self.buildStacks(
                    vehicle, df_items, stackInfo
                )
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
            bestAnts, bestAreas = aco.aco_2D_bin(last_iter=last_iter)
            bestAnt = bestAnts[0]

            column = np.zeros(len(stackInfo))
            for stack in bestAnt:
                # TODO: use code to state
                column[stack.stack_code] += len(stack.items)

            pattern_list.append(
                {
                    "pattern": column,
                    "vehicle": vehicle.id_truck,
                    "area": bestAreas[0],
                }
            )

            self.solUpdate(bestAnt, vehicle)

            # Remove the items already added to the solution
            df_items = df_items[df_items.id_item.isin(self.sol["id_item"]) == False]
            totCost += vehicle["cost"]

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
        print("Tot cost: ", totCost)
        print("Tot items: ", len(self.sol["id_item"]))

        # Evaluating the time taken by every iteration
        time_spent = time.time() - st_time
        print(f"\nTime: {time_spent} s")

        # Return the dataframe solution and its cost to check the best solution among all the iteration
        return df_sol, totCost, pattern_list, time_spent

    def buildStacks(self, vehicle, df_items, stackInfo):
        """
        buildStacks
        -----------
        Function used to create the stack with all the specificity of the selected truck.

        #### INPUT PARAMETERS:
            - vehicle: vehicle type, needed to check the height, weight, max density and max\n
                            stack weight for creating the stacks for this specific truck
            - df_items: dataframe containing all the items that will be put into the trucks
            - stackInfo: dataframe containing all the main information fro each stackability code
        #### OUTPUT PARAMETERS:
            - stack_lst: list of all the stack created
            - stack_quantity: dictionary containing all the numbers of the items divided in stackability codes
        """

        removed = []
        stack_lst = []
        stack_quantity = {code: 0 for code in stackInfo.stackability_code}

        # Vehicle constrain in a dictionary, ready to be passed to the addItem function
        constraints = {
            "max_height": vehicle["height"],
            "max_weight_stack": vehicle["max_weight_stack"],
            "max_density": vehicle["max_density"],
        }

        # Loop over all the stackability code
        for code in stackInfo.stackability_code:
            stack_quantity[code] = 0
            stack_feat = (stackInfo[stackInfo.stackability_code == code].values)[0]
            stack = Stack(code, stack_feat[0], stack_feat[1], stack_feat[3])

            # Taking only the first n items of the specific stack code to speed up the computation
            items_code = df_items[df_items.stackability_code == code]  # .head(1000)

            # Obtaining all the unique value of height and weight for a better stack creation
            unique_height = np.sort(items_code.height.unique())[::-1]
            unique_weight = np.sort(items_code.weight.unique())[::-1]

            # Loop over the items for the stack creation
            for i, row in items_code.iterrows():
                new_stack_needed = False

                # Check that the item is not already been used
                if not row.id_item in removed:
                    stack_added = stack.addItem(row, constraints)

                    # Returned code 0 means that the max stackability code is reached
                    if stack_added == 0:
                        new_stack_needed = True

                    # Returned code -1 means max_height reached
                    if stack_added == -1:
                        new_stack_needed = True

                        # If other item with different height exist then another iteam is searched to be fitted in this stack
                        if len(unique_height) > 1:
                            fit = False
                            h = 0
                            while h < len(unique_height) and not fit:
                                # If an item respect the height constrain is found, all the other constrain are also checked
                                if (
                                    unique_height[h] + stack.height
                                    <= constraints["max_height"]
                                ):
                                    valid_items = items_code[
                                        items_code.height == unique_height[h]
                                    ]
                                    i = 0
                                    while i < len(valid_items) and not fit:
                                        # If all the constrain are respected the item is added to the stack
                                        if (
                                            stack.addItem(
                                                valid_items.iloc[i], constraints
                                            )
                                            == 1
                                        ):
                                            fit = True
                                            id_item = valid_items.iloc[i].id_item
                                            items_code = items_code[
                                                items_code.id_item
                                                != valid_items.iloc[i].id_item
                                            ]
                                            removed.append(id_item)
                                        i += 1
                                h += 1

                    # Returned code -2 means max_weight reached(the following procedure is the same as the height)
                    if stack_added == -2:
                        new_stack_needed = True
                        if len(unique_weight) > 1:
                            fit = False
                            w = 0
                            while w < len(unique_weight) and not fit:
                                if (
                                    unique_weight[w] + stack.weight
                                    <= constraints["max_weight_stack"]
                                ):
                                    valid_items = items_code[
                                        items_code.weight == unique_weight[w]
                                    ]
                                    i = 0
                                    while i < len(valid_items) and not fit:
                                        if (
                                            stack.addItem(
                                                valid_items.iloc[i], constraints
                                            )
                                            == 1
                                        ):
                                            fit = True
                                            id_item = valid_items.iloc[i].id_item
                                            items_code = items_code[
                                                items_code.id_item
                                                != valid_items.iloc[i].id_item
                                            ]
                                            removed.append(id_item)
                                        i += 1
                                w += 1

                    # Returned code -3 means max_density reached(another item is searched)
                    if stack_added == -3:
                        new_stack_needed = True
                        if len(unique_weight) > 1:
                            fit = False
                            w = 0
                            while w < len(unique_weight) and not fit:
                                density = (unique_weight[w] + stack.weight) / stack.area
                                if density <= constraints["max_density"]:
                                    valid_items = items_code[
                                        items_code.weight == unique_weight[w]
                                    ]
                                    i = 0
                                    while i < len(valid_items) and not fit:
                                        if (
                                            stack.addItem(
                                                valid_items.iloc[i], constraints
                                            )
                                            == 1
                                        ):
                                            fit = True
                                            id_item = valid_items.iloc[i].id_item
                                            items_code = items_code[
                                                items_code.id_item
                                                != valid_items.iloc[i].id_item
                                            ]
                                            removed.append(id_item)
                                        i += 1
                                w += 1

                    # Returned code 1 means item correctly added to the stack
                    if stack_added == 1:
                        items_code = items_code[items_code.id_item != row.id_item]

                    # When the stack il ready must be added to the stackList
                    if new_stack_needed:
                        stack.updateHeight()
                        stack_lst.append(stack)
                        stack_quantity[code] += 1

                        stack = Stack(code, stack_feat[0], stack_feat[1], stack_feat[3])
                        stack.addItem(row, constraints)
                        items_code = items_code[items_code.id_item != row.id_item]

            # After the loop if the last stack created have some items must be added
            if stack.n_items > 0:
                stack.updateHeight()
                stack_lst.append(stack)
                stack_quantity[code] += 1

        return stack_lst, stack_quantity

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
