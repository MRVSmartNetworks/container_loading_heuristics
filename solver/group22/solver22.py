# -*- coding: utf-8 -*-
import itertools
import os
import random
import time
import numpy as np
import pandas as pd

from solver.group22.solution_representation import myStack3D
from solver.group22.stack import Stack
from solver.group22.stack_creation_heur import create_stack_heur, refill_stacks
from solver.group22.stack_creation_gurobi_new import create_stack_gurobi

# VERB and MORE_VERB are used to print more information during runs
VERB = False
MORE_VERB = False
# STATS prints scores of 2D solutions
STATS = True

# GUROBI is used to choose whether to use Gurobi to create stacks or not
GUROBI = True

# Fraction (percentage) of deleted trucks when improving solution
FRAC_DELETED_TRUCKS = 35  # %

# N_DEBUG is used to decide whether to execute assertions (much faster solution without)
N_DEBUG = False

MAX_ITER = 10000
MAX_TRIES = 3


class Solver22:
    def __init__(self, df_items=None, df_vehicles=None):
        """
        Solver22
        ---------------------------------------------------------------
        "Decision rule"

        Ordering the items using a given criterion and then trying to
        allocate the items considering one bin at a time. When no more
        items can be allocated in the current bin, we close such a bin
        and open a new one. The process stops when all items have been
        allocated.
        ---------------------------------------------------------------
        """
        self.name = "solver22"

        self.df_items = None
        self.df_vehicles = None

        # Current solution
        self.curr_sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }
        # Current value of the objective function
        self.curr_obj_value = 0

        # Current best solution (so far)
        self.curr_best_sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }
        # Objective value of the best solution - to be updated
        # Initialized to -1 - check it to detect 1st iteration
        self.best_obj_value = -1

        # Store the 2D solutions obtained and associated scores
        self.list_sol_2D = []
        self.scores_2D = []

        # Store the scores of the BEST solution only (updated at the end of each iteration)
        self.scores_2D_best = []

        # Store time for single iteration
        self.runtimes = []

        # Id of truck
        self.idx_vehicle = 0

        # The following are variables used during the run to guide some decision
        self.last_truck_was_empty = False
        self.unusable_trucks = []

        self.iter = 0  # Iterator for single solution
        self.tries = 0  # Iterator used for looping between different solution attempts
        self.stack_number = 1  # Used for choosing the stacks ID, needs to be incremented after each stack creation

        self.random_choice_factor = (
            0  # Probability of random behavior will be e^{random_choice_factor * iter}
        )
        self.truck_score_bound = 0.1  # Deviation from optimal truck score in selection

        self.count_broken = 0

        # Add the possibility to recover stacks if truck ID matches the one from before
        self.stacks_list = []
        self.discarded_stacks = []
        self.curr_truck_type = ""
        self.last_truck_type = ""

    ##########################################################################
    ## Solver

    def solve(
        self,
        df_items,
        df_trucks,
        max_tries=MAX_TRIES,
        used_trucks_dict=None,
        recur=True,
    ):
        """
        solve
        ---
        Solution of the problem with the proposed heuristics.

        ### Input parameters:
        - df_items: dataframe containing all items
        - df_trucks: dataframe containing all trucks
        - max_tries (default const. MAX_TRIES): maximum number of solutions to be found
        """
        random.seed(315054)

        self.df_items = df_items
        self.df_vehicles = df_trucks

        for self.tries in range(max_tries):
            print(f"Solution {self.tries + 1}")
            t_0 = time.time()
            sublist_sol_2D = []
            sublist_scores_2D = []

            self.curr_sol = {
                "type_vehicle": [],
                "idx_vehicle": [],
                "id_stack": [],
                "id_item": [],
                "x_origin": [],
                "y_origin": [],
                "z_origin": [],
                "orient": [],
            }
            self.curr_obj_value = 0

            tmp_items = pd.DataFrame.copy(self.df_items)
            tmp_vehicles = pd.DataFrame.copy(self.df_vehicles)

            # TODO: review lower bound evaluation
            # min_cost, min_n_trucks = self.getLowerBound(tmp_items, tmp_vehicles)
            # print(f"The minimum cost possible is {min_cost} and it is achieved with {min_n_trucks}")

            tmp_items["surface"] = tmp_items["width"] * tmp_items["length"]
            tmp_items["volume"] = tmp_items["surface"] * tmp_items["height"]

            # Order according to dimensions * weight / cost ratio
            if (
                "dim_cost_ratio" not in tmp_vehicles.columns
                and "dim_wt_cost_ratio" not in tmp_vehicles.columns
            ):
                tmp_vehicles["volume"] = (
                    tmp_vehicles["width"]
                    * tmp_vehicles["length"]
                    * tmp_vehicles["height"]
                )
                tmp_vehicles["section"] = tmp_vehicles["width"] * tmp_vehicles["height"]
                tmp_vehicles["dim_cost_ratio"] = (
                    tmp_vehicles["volume"] / tmp_vehicles["cost"]
                )
                tmp_vehicles["dim_wt_cost_ratio"] = (
                    tmp_vehicles["dim_cost_ratio"]
                    * tmp_vehicles["max_weight"]
                    * tmp_vehicles["max_weight_stack"]
                )

            # Used to track the different types of used vehicles and assign unique IDs:
            # NOTE: it can be assigned by the
            if used_trucks_dict is None:
                self.idx_vehicle = 0
                n_trucks = {}
                for id in tmp_vehicles.id_truck.unique():
                    n_trucks[id] = 0
            else:
                n_trucks = used_trucks_dict.copy()
                self.idx_vehicle = sum(used_trucks_dict.values())

            all_trucks_id = []

            self.iter = 0
            print_trucks_ID = []

            self.curr_truck_type = ""
            self.last_truck_type = ""
            while len(tmp_items.index) > 0 and self.iter < MAX_ITER:
                print(f"\nIter {self.iter}")
                if VERB:
                    print(f"> Items left: {len(tmp_items.index)}")

                if self.last_truck_was_empty:
                    self.unusable_trucks.append(str(curr_truck.id_truck[:2]))

                self.last_truck_was_empty = False

                # Strategy for selecting the trucks
                curr_truck = self.selectNextTruck(
                    tmp_vehicles, tmp_items, self.unusable_trucks
                )

                # Update the attributes used for recycling stacks
                self.last_truck_type = self.curr_truck_type
                self.curr_truck_type = curr_truck.id_truck

                n_trucks[curr_truck.id_truck] += 1
                # The value is updated with sol
                curr_truck["idx_vehicle"] = self.idx_vehicle

                if self.iter % 50 == 0:
                    print_trucks_ID.append(curr_truck.idx_vehicle)

                all_trucks_id.append(curr_truck.id_truck)

                if True:
                    print(
                        f"> Truck type: {curr_truck.id_truck}\n> Truck ID: {curr_truck.idx_vehicle}"
                    )

                ##################################
                # Improved implementation
                #
                #
                if self.curr_truck_type != self.last_truck_type:
                    # Build stacks with the copied list of items 'tmp_items'

                    if not GUROBI:
                        self.stacks_list, self.stack_number = create_stack_heur(
                            tmp_items, curr_truck, self.stack_number
                        )
                    else:
                        self.stacks_list, self.stack_number = create_stack_gurobi(
                            tmp_items, curr_truck, self.stack_number
                        )

                    tot_it_in_stacks = sum([len(st.items) for st in self.stacks_list])
                    if N_DEBUG:
                        assert tot_it_in_stacks == len(
                            tmp_items.index
                        ), f"Items in stack = {tot_it_in_stacks}, total items = {len(tmp_items.index)}"
                else:
                    # Destroy the worse stacks that are present in the list
                    # (More than 30% height margin and no max stack)

                    del_index = np.zeros(len(self.stacks_list))
                    t_height = curr_truck["height"]
                    thresh = 0.3 * t_height
                    for k in range(len(self.stacks_list)):
                        if (
                            t_height - self.stacks_list[k].tot_height
                        ) > thresh and not self.stacks_list[k].isMaxStack():
                            del_index[k] = 1

                    for k in list(range(len(del_index)))[::-1]:
                        if del_index[k]:
                            # Extract items from the stacks
                            self.discarded_stacks.append(self.stacks_list[k])
                            del self.stacks_list[k]

                    # The stacks present in the current list are still valid!
                    # Can create new stacks with the items in the discarded ones
                    if len(self.discarded_stacks) > 0:
                        # Extract items from stacks
                        it_recycle = pd.DataFrame(columns=tmp_items.columns)
                        for st in self.discarded_stacks:
                            for it in st.items:
                                it_recycle.loc[len(it_recycle)] = it

                        # Check that there are no discarded elements among the ones in the solution
                        for ind, row in it_recycle.iterrows():
                            if N_DEBUG:
                                for st in self.stacks_list:
                                    assert row["id_item"] not in [
                                        it["id_item"] for it in st.items
                                    ], f"Item {row['id_item']} was discarded, but appears in valid stack {st.id}!"

                        tot_it_not_in_sol = sum(
                            [len(st.items) for st in self.stacks_list]
                        ) + len(it_recycle.index)

                        if N_DEBUG:
                            assert tot_it_not_in_sol == len(
                                tmp_items.index
                            ), "The items not in the solution do not match"

                        self.stacks_list, self.stack_number = refill_stacks(
                            self.stacks_list,
                            it_recycle,
                            curr_truck,
                            self.stack_number,
                            create_stack_heur,
                        )

                # Make sure all items left appear in the stacks
                # (Code borrowed from stack_creation_heur.py)

                # Isolate the single
                n_unique_ids = len(tmp_items.index)

                stack_items_ids = []
                for st in self.stacks_list:
                    stack_items_ids += [it.id_item for it in st.items]

                stack_items_ids = np.array(stack_items_ids)
                used_unique_ids, used_counts = np.unique(
                    stack_items_ids, return_counts=True
                )

                tot_items_in_stacks = sum([len(st.items) for st in self.stacks_list])

                if N_DEBUG:
                    assert tot_items_in_stacks == len(
                        used_unique_ids
                    ), "The items in the stacks are not matching!"

                    assert (
                        len(used_unique_ids) == n_unique_ids
                    ), f"The stacks contain {len(used_unique_ids)} elements, while the remaining items are {n_unique_ids}"
                #
                #
                ##################################

                ##################################
                # Original implementation:
                # NOTE: with this approach, it is not possible to create
                # stacks using gurobi due to time constraints, as it
                # recreates all the stacks from scratch at each iteration

                # Build stacks with the copied list of items 'tmp_items'
                # self.stacks_list, self.stack_number = create_stack_cs(
                #     tmp_items, curr_truck, self.stack_number
                # )

                ##################################

                # Cleanup stacks list
                # self.stacks_list = self.cleanStackList(self.stacks_list)

                if VERB:
                    print(f"Total number of generated stacks: {len(self.stacks_list)}")

                ####### 2D PROBLEM SOLUTION (2D bin packing)
                sol_2D, self.stacks_list, curr_score = self.solve2D(
                    self.stacks_list,
                    curr_truck,
                    scores=True,
                )

                sublist_sol_2D.append(sol_2D)
                sublist_scores_2D.append(curr_score)

                # Use the 2D solution to update the overall solution
                tmp_items = self.updateCurrSol(sol_2D, curr_truck, tmp_items)

                self.iter += 1

            self.curr_obj_value = self.evalObj(df_trucks)

            self.list_sol_2D.append(sublist_sol_2D)
            self.scores_2D.append(sublist_scores_2D)

            used_trucks = 0
            for t in n_trucks.keys():
                used_trucks += n_trucks[t]

            if VERB:
                print(f"Number of trucks analyzed: {used_trucks}")
                print(f"Actual number of used trucks: {self.idx_vehicle}")

            self.checkValidSolution(self.curr_sol, self.df_items, self.df_vehicles)
            print(f"Current objective value: {self.curr_obj_value}")
            print(f"Broken Stacks: {self.count_broken}")

            if N_DEBUG:
                assert (
                    len(tmp_items.index) == 0
                ), "Not all items have been used in current solution!"

            if self.updateBestSol() != 0:
                print("Solution was updated!")
                self.scores_2D_best = sublist_scores_2D

            self.runtimes.append(time.time() - t_0)
            print("Time for solution: ", self.runtimes[self.tries], "\n")

            self.unusable_trucks = []
            self.last_truck_was_empty = False

        print(f"Optimal initial value: {self.best_obj_value}")

        ##### SOLUTION IMPROVEMENT
        # Approach: start from the last truck which was filled, try to extract items
        # and place them in other trucks
        if recur:
            was_improv = self.improveSolution(df_items, df_trucks, n_trucks)

            if was_improv == 1:
                print(f"Improved objective value: {self.best_obj_value}")
            else:
                print(f"Optimal initial value: {self.best_obj_value}")

        # Append best solution for current truck
        # Need to make sure the items left have been updated
        df_sol = pd.DataFrame.from_dict(self.curr_best_sol)
        df_sol.to_csv(os.path.join("results", f"{self.name}_sol.csv"), index=False)

        ### PLOTS:
        if used_trucks_dict is None:
            # Only plot if the trucks dictionary has been created from scratch
            for t_id in print_trucks_ID:
                myStack3D(self.df_items, self.df_vehicles, df_sol, t_id)

            # Get last used truck
            last_truck_id = df_sol.idx_vehicle.iloc[-1]
            myStack3D(self.df_items, self.df_vehicles, df_sol, last_truck_id)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def selectNextTruck(self, trucks_df, remaining_items, forbidden_trucks=[]):
        """
        selectNextTruck
        ---
        Choose the best truck to be selected next, depending on the amount of items left.

        It is possible to specify a list of truck IDs which have to be avoided, in case
        an attempt was already done with these trucks and no item could have been placed.

        ### Input parameters
        - trucks_df: dataframe containing all available trucks - it should contain the
        column "volume" and "dim_cost_ratio".
        - remaining_items: dataframe containing the remaining items; it should contain
        the column "volume"
        - forbidden_trucks: list of IDs of trucks which have found to be not viable to store items
        """

        if "volume" not in remaining_items.columns:
            remaining_items["volume"] = (
                remaining_items["length"]
                * remaining_items["width"]
                * remaining_items["height"]
            )

        tot_item_vol = sum(remaining_items["volume"])
        tot_item_wt = sum(remaining_items["weight"])

        if "volume" not in trucks_df.columns:
            trucks_df["volume"] = (
                trucks_df["length"] * trucks_df["width"] * trucks_df["height"]
            )

        if "dim_cost_ratio" not in trucks_df.columns:
            trucks_df["dim_cost_ratio"] = trucks_df["volume"] / trucks_df["cost"]

        if "dim_wt_cost_ratio" not in trucks_df.columns:
            trucks_df["dim_wt_cost_ratio"] = (
                trucks_df["dim_cost_ratio"]
                * trucks_df["max_weight"]
                * trucks_df["max_weigth_stack"]
            )

        if all(trucks_df.volume < tot_item_vol) and all(
            trucks_df.max_weight < tot_item_wt
        ):
            # Introduce possibility to choose truck randomly which increases with iteration number
            # If the volume of all trucks is lower than the overall volume:
            # return truck with highest dim/cost ratio (first which is not in the string of forbidden trucks)
            ord_vehicles = trucks_df.sort_values(
                by=["dim_wt_cost_ratio"], ascending=False
            )

            best_val = float(ord_vehicles.dim_wt_cost_ratio.iloc[0])
            bound_from_best = best_val - self.truck_score_bound * best_val
            similar_trucks = ord_vehicles[
                ord_vehicles.dim_wt_cost_ratio >= bound_from_best
            ]

            choice = random.random()

            if len(forbidden_trucks) > 0:
                for i, row in ord_vehicles.iterrows():
                    if str(row.id_truck) not in forbidden_trucks:
                        return ord_vehicles.iloc[i]

            elif choice < np.exp(self.random_choice_factor * self.iter):
                return ord_vehicles.iloc[0]

            else:
                n_trucks = len(similar_trucks.index)
                return similar_trucks.iloc[random.randint(0, n_trucks - 1)]

        else:
            # Else: return the truck with the lowest cost among the ones which are bigger than
            # the whole volume
            # Get trucks having volume > 110% total remaining volume (empirical choice)
            valid_trucks = trucks_df[trucks_df.volume >= tot_item_vol * 1.1]
            valid_trucks = valid_trucks[valid_trucks.max_weight >= tot_item_wt]
            sort_vehicles = valid_trucks.sort_values(by=["cost"], ascending=True)

            if len(sort_vehicles.index) > 0:
                for i, row in sort_vehicles.iterrows():
                    if str(row.id_truck) not in forbidden_trucks:
                        return row
            else:
                ord_vehicles = trucks_df.sort_values(
                    by=["dim_wt_cost_ratio"], ascending=False
                )

                return ord_vehicles.iloc[0]

    def create_stack(self, df_items, truck):  # NOT USED!
        """
        create_stack
        ---
        Given an object dataframe and a truck, create stacks which
        can be placed into the truck.

        ### Input parameters:
        - df_items: pandas Dataframe of usable items.
        - truck: pandas Series object containing the truck information.

        NOTE: this method is deprecated as it is far from ideal!

        Having isolated all stackability codes, iterate on all items for each code value.
        Place each item in a stack, until an item does not pass the checks for being added.
        Once this happens, close current stack and add it to the list of stacks, then
        start a new stack by placing the current item in a new one.

        This method makes use of the Stack class and its method(s) 'add_item' (and
        'add_item_override').

        Stacks can be created only for items with the same stackability code.
        """
        stack_codes = df_items.stackability_code.unique()
        stacks_list = []  # Outcome of this function

        for code in stack_codes:
            new_stack_needed = False
            other_constraints = {  # Enforce constraints on the
                "max_height": truck["height"],
                "max_weight": truck["max_weight_stack"],
                "max_dens": truck["max_density"],
            }

            new_stack = Stack()
            for i, row in df_items[df_items.stackability_code == code].iterrows():
                # FIXME: the procedure is not ideal! If an item is not added because too heavy it does
                # not mean that we need to start a new stack...

                was_added = new_stack.add_item_override(row, other_constraints)
                # The value of 'new_stack_needed' can be:
                # 0: cannot add item as it won't satisfy constraint (weight, height, density, stackability)
                # 1: success
                # NOT HERE - {-1: cannot add item since it would lock the orientation property}
                if was_added != 1:
                    new_stack_needed = True
                    # In all other cases can still try to add elements to the stack
                    # FIXME: it may happen that one element cannot be added because too tall/heavy
                    # need to allow for a search for compatible items
                    # IDEA: only stop if max_stackability was violated, else act as in the

                # if a new stack is needed (current element was not added):
                if new_stack_needed:
                    stacks_list.append(new_stack)
                    # Open new stack (with current element as first)
                    new_stack = Stack(row, other_constraints)
                    new_stack_needed = False

                    # NOTE: this approach also works when we end the loop for the current
                    # stackability code value, as next item will not be added (won't pass
                    # checks in add_item[_override])...
                else:
                    # The item was already added
                    pass

            # Last stack is probably not appended to the list, since it does
            # not trigger new_stack_needed
            stacks_list.append(new_stack)

        for i in range(len(stacks_list)):
            stacks_list[i].assignID(i)

        return stacks_list

    def solve2D(self, stacks, truck, scores=False):
        """
        solve2D
        ---
        Solution of the 2D problem (which is performed once the stacks are provided).

        The heuristics is based on the methods "priceStack", "buildSlice" and "pushSlice" and
        follows an approach based on the 'Peak Filling Slices Push'.

        ### Input parameters
        - stacks: list of (remaining) stacks, i.e., the one available to fill the current truck
        - truck: pandas Series object containing the information of the truck currently being filled

        ### Output parameters
        - sol_2D: python dict containing the 2D solution obtained; it contains:
            - x_sol: x coordinates of the origin of each stack
            - y_sol: y coordinates of the origin of each stack
            - stack: Stack object used in the solution
            - orient: flag which is 1 if the stack was rotated by 90 degrees
        """
        # Create a copy, which will be modified (elem. removed)
        up_stacks = stacks.copy()
        tot_passed_items = sum([len(st.items) for st in stacks])

        # Solution based on the bound given by the stacks which were already placed
        # Extract truck info
        x_truck = truck["length"]
        y_truck = truck["width"]

        truck_area = x_truck * y_truck
        truck_volume = truck_area * truck["height"]
        # No need for height (we are solving 2D currently)
        max_weight = truck["max_weight"]

        # This solution simply consists of a 2D bin packing with no constraint but the
        # dimensions of the bin and max_weight: it is assumed the stacks have been built
        # satisfying the other constraint (height, density, stack weight)

        # Initialize solution
        # Keep in mind: this is just the 2D solution, the task of the main solver is
        # that of "translating" this solution into the overall one
        sol_2D = {"x_sol": [], "y_sol": [], "stack": [], "orient": []}

        # Initialize bound
        bound = [[0, 0], [0, y_truck]]

        space_left = True
        weight_left = max_weight

        count = 0
        self.discarded_stacks = []
        while space_left and weight_left > 0:
            # 1. Assign prices to each stack:
            self.priceStack(up_stacks, override=[0, 1, 2, 3])

            curr_stacks_n = len(up_stacks)

            # 2. Build slices - choose stacks according to highest price
            # Brute force
            rightmost = max([p[0] for p in bound])
            x_dim = x_truck - rightmost

            # BUILD A NEW SLICE
            new_slice, up_stacks, disc_stacks = self.buildSlice(
                up_stacks, x_dim, y_truck, weight_left
            )

            self.discarded_stacks += disc_stacks

            if len(new_slice) > 0:  # Slice is not empty
                # Having built the slice:
                # Update weight_left (NOTE: check is done at slice creation):
                weight_added = 0
                for el in new_slice:
                    weight_added += el[0].tot_weight
                weight_left = weight_left - weight_added

                # 'Push' stack towards bottom
                sol_2D, bound = self.pushSlice(bound, new_slice, sol_2D)

                if N_DEBUG:
                    assert all(
                        len(st.items) > 0 for st in sol_2D["stack"]
                    ), "(2) Some stacks of the solution are empty!"

                    if bound[-1][1] != truck["width"]:
                        print("Error!")

                    assert (
                        bound[-1][1] == truck["width"]
                    ), f"Bound was built wrong! last: {bound[-1][1]}, width: {truck['width']}"

            else:
                # If the new slice is empty, close the bin
                if len(sol_2D["x_sol"]) == 0:
                    print("Cannot fit any item in this truck!")
                    self.last_truck_was_empty = True

                # TODO: check if the bound has some space left
                space_left = False

            count += 1

        if scores:
            # Evaluate area, volume and weight utilization
            items_area = 0
            items_volume = 0
            items_weight = 0
            for st in sol_2D["stack"]:
                if N_DEBUG:
                    assert len(st.items) > 0, "The current stack is empty!!!"
                items_area += st.area
                items_volume += st.area * st.tot_height
                items_weight += st.tot_weight

            percent_area = items_area / truck_area
            # percent_vol = items_volume / truck_volume
            percent_weight = items_weight / truck.max_weight

            # Assigning scores (2 possible approaches)
            # score = [percent_area, percent_weight, percent_area * percent_weight]
            score = [percent_area, percent_weight, percent_area + percent_weight]

        if STATS:
            print(f"> Truck area utilization: {percent_area}")
            # print(f"> Truck volume utilization: {percent_vol}")
            print(f"> Truck weight utilization: {percent_weight}")

        # Check that all the items are preserved
        tot_sol_items = sum([len(st.items) for st in sol_2D["stack"]])
        tot_discarded_items = sum([len(st.items) for st in self.discarded_stacks])
        tot_remaining_items = sum([len(st.items) for st in up_stacks])

        tot_items_after = tot_sol_items + tot_discarded_items + tot_remaining_items

        if N_DEBUG:
            assert (
                tot_passed_items == tot_items_after
            ), f"Passed items: {tot_passed_items}\nFinal count: {tot_items_after}"

        # Something else?
        if scores:
            return sol_2D, up_stacks, score

    def priceStack(self, stacks, override=None):
        """
        pricesStack
        ---
        Assign to each stack a price, which will be used to choose which
        one to place first when solving the 2D bin packing problem.

        There are 4 different rules to assign the price, chosen randomly:
        - Price = area
        - Price = length
        - Price = width
        - Price = perimeter
        - Price = stack height ---- Not so good
        - Price = total volume
        - Price = 1 / density - as defined by the specs (weight / area)
        - Price = height / weight

        The input variable 'stacks' is a list of Stack objects.
        This method updates the 'price' attribute inside each Stack object.

        Via the parameter 'override', it is possible to force the choice
        on one or more cost types.
        This parameter can be either a list of ints or a single int value,
        specifying the price strategy.
        """
        # Select which pricing type
        if override is None:
            val = random.randint(0, 7)
        elif isinstance(override, int):
            val = override
        elif isinstance(override, list):
            val = override[random.randint(0, len(override) - 1)]
        else:
            raise ValueError("Parameter 'override' should be of type int (or None)!")

        if val == 0:
            for i in range(len(stacks)):
                # Area
                stacks[i].assignPrice(stacks[i].area)
        elif val == 1:
            for i in range(len(stacks)):
                # Length
                stacks[i].assignPrice(stacks[i].length)
        elif val == 2:
            for i in range(len(stacks)):
                # Width
                stacks[i].assignPrice(stacks[i].width)
        elif val == 3:
            for i in range(len(stacks)):
                # Perimeter
                stacks[i].assignPrice(stacks[i].perimeter)
        elif val == 4:
            for i in range(len(stacks)):
                # Stack height
                stacks[i].assignPrice(stacks[i].tot_height)
        elif val == 5:
            for i in range(len(stacks)):
                # Volume
                stacks[i].assignPrice(stacks[i].tot_height * stacks[i].area)
        elif val == 6:
            for i in range(len(stacks)):
                # Area / weight
                stacks[i].assignPrice(stacks[i].area / stacks[i].tot_weight)
        elif val == 7:
            for i in range(len(stacks)):
                # Height / weight
                stacks[i].assignPrice(stacks[i].tot_height / stacks[i].tot_weight)

    def cleanStackList(self, stacks):
        """
        cleanStackList
        ---
        Remove from the stacks list the ones which are incorrect (empty)
        to prevent errors in other functions.

        ### Input parameters
        - stacks: list of Stack objects

        ### Output variables
        - stacks: updated stacks list
        """
        count_rem = 0
        for i in range(len(stacks)):
            if len(stacks[i].items) < 1:
                del stacks[i]
                count_rem += 1
            else:
                i += 1

        if VERB:
            print(f"{count_rem} stacks were removed because empty!")

        return stacks

    def buildSlice(self, stacks, x_dim, y_dim, max_weight):
        """
        buildSlice
        ---
        This method is used to populate slices of the trailer to be filled.
        This is done by choosing first slices with higher 'price'.

        The ordered list is then read sequentially and stacks which fit (fulfilling
        constraints) are placed.

        NOTE: Used stacks are removed from the list.

        Once all stacks have been placed, if there is still weight and space left,
        the remaining stacks are analyzed and it is tried to add them by 'breaking'
        them, i.e., by removing the heaviest elements in the stacks to try and fit
        them in the new slice.

        [O(n^2)]

        ### Input parameters:
        - stacks: list of Stack object, need price to be initialized already.
        - x_dim: available space in the x direction (length)
        - y_dim: available space in the y direction (width) - slice is built
        along this direction

        ### Output variables:
        - new_slice: list of sublists; each sublist contains:
          - Stack object
          - Index of the stack in the initial list
          - Rotation - 0 if not rotated, 1 if rotated
          - y coordinate of the origin

        Note that this method can be used to build slices of arbitrary 2D
        dimensions, so it may also be used to fill spaces with smaller/fewer boxes...

        This method contains the main procedure used to fill the truck.
        To change strategy, just change this function.
        """
        # Mechanism for avoiding stack losses
        n_stacks_init = len(stacks)
        n_stacks_placed = 0

        weight_left = max_weight
        new_slice = []

        if MORE_VERB:
            print("Starting to build slice:")
            print("  Number of available stacks: ", len(stacks))

        if N_DEBUG:
            assert all(
                len(st.items) > 0 for st in stacks
            ), "buildSlice was provided with empty stacks"

        # Sort the stacks according to decreasing price
        stacks.sort(key=lambda x: x.price, reverse=True)
        # stack_added_flags = np.zeros((len(stacks)))

        i = 0  # i tracks the index of the stack list
        j = 0  # j tracks the number of stacks in the current slice
        delta_y = y_dim
        # Until all possible stacks have been visited, try to add new one to fill slice
        # NOTE: also the check of the weight is done here!
        while i < len(stacks):
            stack_added = False
            # It may happen that some stacks are created empty
            if len(stacks[i].items) > 0 and weight_left >= stacks[i].tot_weight:
                if delta_y >= stacks[i].width and x_dim >= stacks[i].length:
                    # Stack is good as is - insert it
                    new_slice.append([stacks[i], i, 0])

                    if MORE_VERB:
                        print("Stack added to slice!")

                    delta_y -= stacks[i].width
                    stack_added = True
                    n_stacks_placed += 1
                    del stacks[i]
                    i -= 1

                elif (
                    stacks[i].forced_orientation == "n"
                    and delta_y >= stacks[i].length
                    and x_dim >= stacks[i].width
                ):
                    # If the stack cannot be placed, try rotating it by 90 degrees, if allowed
                    new_slice.append([stacks[i], i, 1])

                    if MORE_VERB:
                        print("Stack added to slice! (2)")
                    # Rotated stack - can place it width-wise
                    delta_y -= stacks[i].length
                    stack_added = True
                    n_stacks_placed += 1
                    del stacks[i]
                    i -= 1

            if stack_added:
                if N_DEBUG:
                    assert (
                        j == len(new_slice) - 1
                    ), f"Wrong j = {j}, but slice contains {len(new_slice)} items"

                # Update weight_left - remove the weight of the last added stack
                weight_left -= new_slice[-1][0].tot_weight

                # Update origin y coordinate
                if j != 0:
                    # Get width (length if rotated) of 2nd to last element
                    if new_slice[-2][2] == 0:
                        if MORE_VERB:
                            print("Success here")
                        w_min2 = new_slice[j - 1][0].width
                    else:
                        if MORE_VERB:
                            print("Success here (2)")
                        w_min2 = new_slice[j - 1][0].length
                    # Add the width to the origin of the stack to get new origin
                    # This ensures no space is left
                    new_slice[j].append(new_slice[-2][-1] + w_min2)
                else:
                    # Stack is placed at y = 0
                    new_slice[j].append(0)
                j += 1

            i += 1
        # When out of the loop, the slice has been built - but it may be sub-optimal
        # E.g., there is still some space/weight left, but full stacks are too
        # heavy --> It may be possible to remove some itmes from stacks to try and
        # place them.

        if N_DEBUG:
            assert (
                len(stacks) == n_stacks_init - n_stacks_placed
            ), "The stacks do not add up... (1)"

        number_old_stacks = i

        if MORE_VERB:
            print("  Number of full stacks used: ", n_stacks_placed)
            print("  Number of stacks left: ", len(stacks))

        ########## BREAKING STACKS ##########
        # Keep track of the discarded items (placed in new stacks):
        discarded_stacks = []
        n_placed_broken = 0

        # If there are still stacks left, iterate
        if weight_left > 0 and len(stacks) > 0:
            i = 0
            # Review the possibility to add *pieces* of remaining stacks
            while i < len(stacks):
                stack_added = False
                # Iterate over the remaining stacks
                # If the stack would be too heavy:
                if len(stacks[i].items) > 0 and weight_left < stacks[i].tot_weight:
                    # If there is still space in the current slice:
                    if delta_y >= stacks[i].width and x_dim >= stacks[i].length:
                        # Initialize extra stack for DISCARDED ITEMS
                        new_stack = Stack()

                        # Try removing elements from the stack (heaviest first) until it
                        # is (possibly) light enough to be placed OR there are no more items in the stack
                        while (
                            weight_left < stacks[i].tot_weight
                            and len(stacks[i].items) > 0
                        ):
                            # Keep the removed item and add it to the extra stack
                            rem_item = stacks[i].removeHeaviestItem()

                            # Create new stack with removed item
                            new_stack.add_item_override(rem_item)

                        # If the current (broken) stack still contains elements, it means that it can
                        # be added (loop was broken because stack weight became < available weight)
                        if len(stacks[i].items) > 0:
                            # The broken stack can be added to the solution
                            new_slice.append([stacks[i], i, 0])
                            n_placed_broken += 1
                            n_stacks_placed += 1

                            # Update y dimension left
                            delta_y -= stacks[i].width
                            stack_added = True
                            # Remove stack from original list
                            del stacks[i]
                            self.count_broken += 1

                            # Add the new stack with discarded elements to the 'stacks' list
                            new_stack.assignID(number_old_stacks)
                            number_old_stacks += 1
                            if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
                                # NOTE: adding the half stack back into the list of stacks causes 'inefficient'
                                # stacks to be kept -> instead, add them to the discarded ones
                                discarded_stacks.append(new_stack)
                        else:
                            # The stack was emptied completely
                            if N_DEBUG:
                                assert (
                                    len(stacks[i].items) == 0
                                ), "The stack is not actually empty..."

                            # Remove the empty element from the stack list
                            del stacks[i]
                            # Also, store among the discarded stacks the other one. (Will be recycled)
                            discarded_stacks.append(new_stack)
                            i -= 1  # Needed to prevent skipping an element

                    # Try changing orientation
                    # (Repeat same procedure as before)
                    elif (
                        stacks[i].forced_orientation == "n"
                        and delta_y >= stacks[i].length
                        and x_dim >= stacks[i].width
                    ):
                        new_stack = Stack()
                        while (
                            weight_left < stacks[i].tot_weight
                            and len(stacks[i].items) > 0
                        ):
                            rem_item = stacks[i].removeHeaviestItem()
                            n_placed_broken += 1
                            n_stacks_placed += 1

                            # Create new stack with removed item
                            new_stack.add_item_override(rem_item)

                        if len(stacks[i].items) > 0:
                            new_slice.append([stacks[i], i, 1])
                            delta_y -= stacks[i].width
                            stack_added = True
                            del stacks[i]
                            self.count_broken += 1

                            new_stack.assignID(number_old_stacks)
                            number_old_stacks += 1
                            if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
                                discarded_stacks.append(new_stack)

                        else:
                            del stacks[i]
                            discarded_stacks.append(new_stack)
                            i -= 1  # Needed to prevent skipping an element

                    # else:
                    #     # if stack cannot be placed in the current slice, delete it
                    #     # from the list (is it???)
                    #     # -> wrong, maybe it can be broken and placed in another slice!
                    #     stack_added = False
                    #     del stacks[i]
                    #     i -= 1

                    #########

                    if stack_added:
                        if N_DEBUG:
                            assert (
                                j == len(new_slice) - 1
                            ), f"Wrong j = {j}, but slice contains {len(new_slice)} items"
                        # Update weight_left - remove the weight of the current stack
                        weight_left -= new_slice[-1][0].tot_weight

                        # Update origin y coordinate
                        if j > 0:
                            # Get width (length if rotated) of 2nd to last element
                            if new_slice[j - 1][2] == 0:
                                if MORE_VERB:
                                    print("Success here")
                                w_min2 = new_slice[j - 1][0].width
                            else:
                                if MORE_VERB:
                                    print("Success here (2)")
                                w_min2 = new_slice[j - 1][0].length
                            # Add the width to the origin of the stack to get new origin
                            # This ensures no space is left
                            new_slice[j].append(new_slice[j - 1][-1] + w_min2)
                        else:
                            # Stack is placed at y = 0
                            new_slice[j].append(0)
                        j += 1

                elif len(stacks[i].items) == 0:
                    # Clean stacks - remove possibly empty stacks
                    del stacks[i]

                else:
                    if len(stacks[i].items) > 0 and (
                        (delta_y >= stacks[i].width and x_dim >= stacks[i].length)
                        or (
                            stacks[i].forced_orientation == "n"
                            and delta_y >= stacks[i].length
                            and x_dim >= stacks[i].width
                        )
                    ):
                        # If the stack would fit in terms of width, length and weight
                        if stacks[i].weight < weight_left:
                            raise ValueError(
                                "BuildSlice did not use all possible slices!"
                            )
                            # This means that the current stack contains elements and it
                            # could have been inserted as-is in the slice.

                i += 1

        if MORE_VERB:
            print(f"N. stacks in new slice: {len(new_slice)}")

        if VERB:
            print("Number of discarded stacks: ", len(discarded_stacks))

        if N_DEBUG:
            assert (
                len(stacks)
                == n_stacks_init
                - n_stacks_placed
                - len(discarded_stacks)
                + n_placed_broken
            ), "The stacks do not add up... (2)"

        for st in discarded_stacks:
            st.assignID(number_old_stacks)
            number_old_stacks += 1

        self.stack_number = number_old_stacks

        return new_slice, stacks, discarded_stacks

    def pushSlice(self, bound, new_slice, curr_sol_2D):
        """
        pushSlice
        ---
        Perform the 'push' operation on the new slice.

        ### Input parameters
        - bound: current bound - will be updated by the function (NOTE: property of Python
        language - modifying strings in a method also modifies them outside, for how they
        are referenced)
        - new_slice: slice to be pushed; the format is the same as the output of 'buildSlice'

        ### Return values
        - curr_sol_2D: the updated 2D solution (dict)

        ### Push operation
        - For each new stack 'i':
          - Isolate the points in the current bound which have y coordinates in the range
          [y_origin[i], y_origin[i] + y_dim[i]], being y_origin the y coordinate of the origin
          of the stack (fixed at slice creation) and y_dim the dimension of the stack along
          the y direction (it is the width if not rotated, the length if rotated)
          - The x coordinate of the origin in the stack will be the max value of x for the
          isolated points

        ### Updating the bound
        The new boundary is obtained by determining the vertices of all elements which have been
        placed in last slice.
        Since by definition the boundary has to have as last item a point having as y coordinate
        the truck width, to prevent missing points, a check is performed to possibly add points
        to the new bound to fill the gap.
        """
        new_bound = []

        # Store the index of the first element in the bound which is valid
        for new_stack in new_slice:
            y_i = new_stack[3]
            if new_stack[2] == 0:
                w_i = new_stack[0].width
            else:
                w_i = new_stack[0].length

            # Find lower bound starting from 0
            ind_bound = 0
            while ind_bound < len(bound) and bound[ind_bound][1] <= y_i:
                ind_bound += 1

            if ind_bound < len(bound):
                assert bound[ind_bound][1] > y_i
                ind_bound -= 1
                # This point has the same x coordinate as the one at which
                # the loop was broken and it is for sure not 'above' the
                # current stack
            else:
                raise ValueError("Out of boundary array bounds!")

            # Search for valid points
            ind_top = ind_bound + 0
            # (Needed to prevent to just copy the reference and update both indices...)
            while ind_top < len(bound) and bound[ind_top][1] <= y_i + w_i:
                ind_top += 1
            # When the loop finishes, the element bound[ind_top] contains the upper end

            if ind_top >= len(bound):
                assert bound[ind_top - 1][1] == y_i + w_i
                ind_top -= 1

            if N_DEBUG:
                assert (
                    len(bound[ind_bound : ind_top + 1]) > 1
                ), "The considered elements of the bound are less than 2! Something went wrong"

            # The x coordinate is the max between the x coord of the elements of
            # index between ind_bound and ind_top
            x_i = max([p[0] for p in bound[ind_bound : ind_top + 1]])

            # Build new (current) solution
            """
            sol_2D = {
                "x_sol":[],
                "y_sol":[],
                "stack":[],
                "orient":[]
            }
            """
            curr_sol_2D["x_sol"].append(x_i)
            curr_sol_2D["y_sol"].append(y_i)
            curr_sol_2D["stack"].append(new_stack[0])
            curr_sol_2D["orient"].append(new_stack[2])

            # Update the bound
            # Simply add the points of the 'rightmost' points of the current stack
            if new_stack[2] == 0:
                l_i = new_stack[0].length
            else:
                l_i = new_stack[0].width
            x_br = x_i + l_i
            y_br = y_i

            x_tr = x_i + l_i
            y_tr = y_i + w_i
            new_bound.append([x_br, y_br])
            new_bound.append([x_tr, y_tr])

        # Fill the bound if the current slice does not reach the full width
        if new_bound[-1][1] < bound[-1][1]:
            # Increase the index from 0 until the element of the old bound is bigger
            ind_extra = 0

            while bound[ind_extra][1] < new_bound[-1][1] and ind_extra < len(bound):
                ind_extra += 1

            # ind_extra locates the 1st corner in the old bound which has y bigger
            # than the current last element in the new bound

            # Add adjustment point:
            # x is the one of the old bound
            # y is the same as the last element in the current bound
            if ind_extra < len(bound):
                new_bound.append([bound[ind_extra][0], new_bound[-1][1]])

                for p in bound[ind_extra:]:
                    new_bound.append(p)

                if N_DEBUG:
                    assert (
                        bound[-1][1] == new_bound[-1][1]
                    ), f"The last y of the bound does not match - {bound[ind_extra - 1][1]} (old) vs. {new_bound[-1][1]}"

            elif bound[ind_extra - 1][1] < new_bound[-1][1]:
                raise ValueError("The last point of the bound was lost!")

            else:
                # Ind_extra == len(bound)
                if N_DEBUG:
                    assert (
                        bound[ind_extra - 1][1] == new_bound[-1][1]
                    ), f"The last y of the bound should have been {bound[ind_extra - 1][1]}, it is instead {new_bound[-1][1]}"

        return curr_sol_2D, new_bound

    def evalObj(self, df_trucks, sol=None):
        """
        evalObj
        ---
        Evaluate the objective value of the given solution.

        ### Input parameters
        - sol (default None): solution, Dictionary; the format is the one specified in the
        attributes of Solver22; if None, the current solution is used
        (self.curr_sol)
        """
        if sol is None:
            s = self.curr_sol
        else:
            s = sol

        o_val = 0
        vehicle_list_type = np.array(s["type_vehicle"])
        vehicle_list_id = np.array(s["idx_vehicle"])

        unique_ids, idx = np.unique(vehicle_list_id, return_index=True)
        all_vehicle_types = vehicle_list_type[idx]

        for t_id in all_vehicle_types:
            o_val += float(df_trucks.loc[df_trucks.id_truck == t_id, "cost"].values)

        return o_val

    def improveSolution(self, df_items, df_trucks, used_trucks_dict):
        """
        improveSolution
        ---
        Idea: evaluate score of filled trucks (% weight * % surface) and try destroying the
        trucks with lowest values and rebuilding them from scratch.
        The process is successful whenever the final cost of the trucks is lower than before.

        The

        ### Input parameters
        - df_items: initial dataframe containing all items
        - df_trucks: dataframe containing all truck types

        ### Output values
        - 1: solution was updated
        - 0: solution was not updated
        """
        # Work on self.current_best_sol and self.scores_2D_best

        # Get the individual trucks
        if N_DEBUG:
            assert len(self.curr_best_sol["idx_vehicle"]) == len(
                self.scores_2D_best
            ), "The number unique truck IDs does not correspond to the number of scores..."

        # Extract scores
        trucks_scores = [e[2] for e in self.scores_2D_best]

        # Get indices of sorted scores (INCREASING)
        ind_sorted_scores = np.argsort(trucks_scores)

        # Sort trucks according to the increasing score
        sorted_trucks = np.unique(np.arange(self.idx_vehicle))[ind_sorted_scores]

        # The number of selected trucks is given by the percentage
        n_del_trucks = round((FRAC_DELETED_TRUCKS * len(sorted_trucks)) / 100)
        worst_trucks = sorted_trucks[:n_del_trucks]
        # Get their type
        worst_trucks_type = []
        for i in range(len(self.curr_best_sol["type_vehicle"])):
            if (
                self.curr_best_sol["idx_vehicle"][i] in worst_trucks
                and self.curr_best_sol["type_vehicle"][i] not in worst_trucks_type
            ):
                worst_trucks_type.append(self.curr_best_sol["type_vehicle"][i])

        print(worst_trucks)

        # Extract items in the current solution which are placed in these specific trucks
        item_cols = df_items.columns
        collected_items = pd.DataFrame(columns=item_cols)

        # Isolate the list of items IDs
        best_sol_items = np.array(self.curr_best_sol["id_item"])
        # Isolate the list of truck IDs
        best_sol_trucks = np.array(self.curr_best_sol["idx_vehicle"])
        for tr_id in worst_trucks:  # tr_id is an ID (int)
            items_in_curr_truck = best_sol_items[
                best_sol_trucks == tr_id
            ]  # Isolate items in current truck
            for it_id in items_in_curr_truck:
                collected_items.loc[len(collected_items)] = df_items.loc[
                    df_items.id_item == it_id
                ].iloc[0]

        # Shuffle the items dataframe:
        collected_items = collected_items.sample(frac=1)

        # Now it is possible to evaluate the price associated with these trucks
        # and then we can proceed with a new solution involving the specified
        # objects only

        old_cost = sum(
            float(df_trucks.loc[df_trucks.id_truck == t_id, "cost"].values)
            for t_id in worst_trucks_type
        )

        print("#########################")
        print("#  Improvement Attempt  #")
        print("#########################")

        sub_solver = Solver22()
        sub_solver.solve(
            collected_items,
            df_trucks,
            max_tries=5,
            used_trucks_dict=used_trucks_dict,
            recur=False,
        )

        print("######################")

        out = 0

        if sub_solver.best_obj_value < old_cost:
            # If the obj value is improved, update the solution!
            print("Solution was improved!")
            out = 1
            # First, gotta remove the old worse trucks from the current solution
            i = 0
            while i < len(self.curr_best_sol["idx_vehicle"]):
                if self.curr_best_sol["idx_vehicle"][i] in worst_trucks:
                    del self.curr_best_sol["type_vehicle"][i]
                    del self.curr_best_sol["idx_vehicle"][i]
                    del self.curr_best_sol["id_stack"][i]
                    del self.curr_best_sol["id_item"][i]
                    del self.curr_best_sol["x_origin"][i]
                    del self.curr_best_sol["y_origin"][i]
                    del self.curr_best_sol["z_origin"][i]
                    del self.curr_best_sol["orient"][i]

                else:
                    i += 1

            # TODO: maybe check for correct length of the solution lists

            # Then, extract the (best) solution obtained from rebuilding the bad trucks
            best_rebuilt_sol = sub_solver.curr_best_sol

            self.curr_best_sol["type_vehicle"] += best_rebuilt_sol["type_vehicle"]
            self.curr_best_sol["idx_vehicle"] += best_rebuilt_sol["idx_vehicle"]
            self.curr_best_sol["id_stack"] += best_rebuilt_sol["id_stack"]
            self.curr_best_sol["id_item"] += best_rebuilt_sol["id_item"]
            self.curr_best_sol["x_origin"] += best_rebuilt_sol["x_origin"]
            self.curr_best_sol["y_origin"] += best_rebuilt_sol["y_origin"]
            self.curr_best_sol["z_origin"] += best_rebuilt_sol["z_origin"]
            self.curr_best_sol["orient"] += best_rebuilt_sol["orient"]

            # Update the solution value
            self.best_obj_value = self.evalObj(df_trucks, sol=self.curr_best_sol)

        return out

    def updateCurrSol(self, sol_2D, truck, items):
        """
        updateCurrSol
        ---
        Update the current solution for this iteration, contained in class
        attribute 'self.curr_sol'.

        ### Input parameters
        - sol_2D: 2D solution - dict containing info on stack placement
        - truck: pandas Series object containing the current truck info
        - items: DataFrame containing the list of items to be updated

        ### Output parameters
        - upd_items: updated list of items (the ones used in the solution
        have been removed)
        """
        upd_items = items.copy()

        for i in range(len(sol_2D["stack"])):
            z_lst = sol_2D["stack"][i].getZList()
            j = 0
            for it in sol_2D["stack"][i].items:
                # if it.id_item == "I0283":
                #     print("Incriminated element!")
                self.curr_sol["type_vehicle"].append(truck["id_truck"][:2])
                self.curr_sol["idx_vehicle"].append(self.idx_vehicle)
                self.curr_sol["id_stack"].append(sol_2D["stack"][i].id)
                self.curr_sol["id_item"].append(it["id_item"])
                self.curr_sol["x_origin"].append(sol_2D["x_sol"][i])
                self.curr_sol["y_origin"].append(sol_2D["y_sol"][i])
                self.curr_sol["z_origin"].append(z_lst[j])
                if sol_2D["orient"][i] == 1:
                    self.curr_sol["orient"].append("w")
                else:
                    self.curr_sol["orient"].append("l")
                j += 1

                # Remove used items from the items DF
                upd_items = upd_items[upd_items.id_item != it["id_item"]]

        if i > 0:
            # Only update if the truck was used
            self.idx_vehicle += 1
        return upd_items

    def updateBestSol(self):
        """
        updateBestSol
        ---
        Update the best solution by comparing the current result with the
        best one so far.

        ### Return values
        - 0: best sol was not updated
        - 1: solution was updated
        - -1: sol was updated (1st iteration)
        """
        if self.tries == 0:
            self.curr_best_sol = self.curr_sol.copy()
            self.best_obj_value = self.curr_obj_value
            return -1
        elif self.curr_obj_value < self.best_obj_value:
            self.curr_best_sol = self.curr_sol.copy()
            self.best_obj_value = self.curr_obj_value
            return 1
        else:
            return 0

    ##########################################################################
    # Utilities

    def getLowerBound(self, df_items, df_trucks):
        """
        getLowerBound
        ---
        Obtain the lower bound on the number of trucks & objective function cost
        for the solution of the problem

        FIXME: fix behavior - any type of truck can be selected as many times as possible
        """
        # Get overall volume of the items
        df_items["volume"] = df_items["length"] * df_items["width"] * df_items["height"]
        tot_item_vol = sum(df_items["volume"])

        print(f"Total items volume: {tot_item_vol}")

        # Get volume of trucks
        df_trucks["volume"] = (
            df_trucks["height"] * df_trucks["width"] * df_trucks["height"]
        )
        print(df_trucks["volume"])
        # Get dim/cost ratio
        df_trucks["dim_cost_ratio"] = (
            df_trucks["width"] * df_trucks["length"] * df_trucks["height"]
        ) / df_trucks["cost"]

        # Get all possible combinations of elements from 0 to len(df_trucks.index)-1
        possib = list(itertools.permutations(list(df_trucks.index)))

        n_trucks_min = len(df_trucks.index)
        best_cost = sum(df_trucks["cost"])

        for i in range(len(possib)):
            vol_tot = 0
            cost_tot = 0
            j = 0
            while vol_tot < tot_item_vol and j < len(possib[i]):
                vol_tot += df_trucks.iloc[possib[i][j]]["volume"]
                cost_tot += df_trucks.iloc[possib[i][j]]["cost"]
                j += 1

            if cost_tot < best_cost:
                best_cost = cost_tot
                n_trucks_min = j

        return best_cost, n_trucks_min

    def checkValidSolution(self, sol, items_df, trucks_df, verb=False):
        """
        checkValidSolution
        ---
        Check the validity of a solution.

        The syntax of the solution must be the same as the one of 'curr_sol'
        attribute of Solver22 class.

        ### Input parameters
        - sol: solution to be tested (dict)
        - items_df: initial dataframe of items
        - trucks_df: initial dataframe of trucks
        """
        valid = True

        # 1. Check all items have been used exactly once:
        item_ids = items_df.id_item.tolist()
        item_counted = np.zeros((len(item_ids),))

        n_items_in_sol = len(np.unique(sol["id_item"]))

        if n_items_in_sol != len(items_df.index):
            valid = False
            print(
                f"The number of items in the solution is {n_items_in_sol}, but the total items are: {len(items_df.index)}"
            )
            # Find the duplicate items
            used_ids, counts = np.unique(sol["id_item"], return_counts=True)

            for j in range(len(used_ids)):
                if counts[j] > 1:
                    print(f"Item {used_ids[j]} was used {counts[j]} times!")

            return valid

        for i in range(n_items_in_sol):
            if sol["id_item"][i] in item_ids:
                item_counted[i] += 1

        if not all(item_counted == 1):
            valid = False
            print(f"Not all items have been used!")

            return valid

        # 2. Check non overlapping
        used_trucks_idx = np.unique(sol["idx_vehicle"])
        used_trucks_type = np.unique(sol["type_vehicle"])

        # for i in range(len(used_trucks_id)):
        #     truck_type = str(used_trucks_id[i][:2])
        #     curr_truck = trucks_df.loc[trucks_df.id_truck == truck_type]

        #     curr_items = np.array(sol["id_item"])[
        #         sol["idx_vehicle"] == used_trucks_id[i]
        #     ]
        #     curr_x = np.array(sol["x_origin"])[sol["idx_vehicle"] == used_trucks_id[i]]
        #     curr_y = np.array(sol["y_origin"])[sol["idx_vehicle"] == used_trucks_id[i]]
        #     curr_z = np.array(sol["z_origin"])[sol["idx_vehicle"] == used_trucks_id[i]]
        #     curr_orient = np.array(sol["orient"])[
        #         sol["idx_vehicle"] == used_trucks_id[i]
        #     ]

        #     # Build the 2d surface

        # 3. Check constraints

        for i in range(len(used_trucks_idx)):
            truck_type = sol["type_vehicle"][i]
            curr_truck = trucks_df.loc[trucks_df.id_truck == truck_type]

            # 3.1 - Max area
            # Assuming the items are not overlapped:
            curr_items_ground = []
            for j in range(len(sol["idx_vehicle"])):
                if (
                    sol["idx_vehicle"][j] == used_trucks_idx[i]
                    and sol["z_origin"][i] == 0
                ):
                    curr_items_ground.append(sol["id_item"][i])

            curr_items_ground = np.array(curr_items_ground)

            bool_items_ind = np.zeros((len(items_df.index),))
            items_id_list = items_df.id_item.tolist()
            for j in range(len(items_id_list)):
                if items_id_list[j] in curr_items_ground:
                    bool_items_ind[j] = 1

            items_gnd = items_df.loc[bool_items_ind == 1]
            items_gnd_area = pd.DataFrame()

            items_gnd_area["area"] = items_gnd["length"] * items_gnd["width"]

            tot_used_area = sum(items_gnd_area.area)

            if tot_used_area > int(curr_truck.length.iloc[0]) * int(
                curr_truck.width.iloc[0]
            ):
                valid = False
                print(f"Truck {used_trucks_idx[i]} includes too many elements!")
                return valid

            # 3.2 - Max weight
            curr_tot_wt = curr_truck["max_weight"].values

            items_in_truck_id = np.array(sol["id_item"])[
                sol["idx_vehicle"] == used_trucks_idx[i]
            ]

            bool_items_ind = np.zeros((len(items_df.index),))
            items_id_list = items_df.id_item.tolist()
            for j in range(len(items_id_list)):
                if items_id_list[j] in items_in_truck_id:
                    bool_items_ind[j] = 1

            items_in_truck = items_df.loc[bool_items_ind == 1]

            tot_wt_items = sum(items_in_truck.weight)
            if tot_wt_items > curr_tot_wt:
                valid = False
                print(
                    f"Truck {used_trucks_idx[i]} has its max_weight constraint violated"
                )
                return valid

        if valid:
            print("###############\nValid solution!\n###############")

        return valid
