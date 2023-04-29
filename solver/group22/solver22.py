# -*- coding: utf-8 -*-
import itertools
import math
import os
import random

import numpy as np
import pandas as pd

from solver.group22.stack import Stack

# TODO: remove full list of items from stack attributes, only store the item ID, which can be used
# in the dataframe to locate the correct row (supposing NO DUPLICATES).

DEBUG = False
MAX_ITER = 10000

class Solver22:
    def __init__(self):
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

        # Current solution
        self.curr_sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
        }
        # Current value of the objective function
        # TODO: function to evaluate the objective value of the current solution...
        self.curr_obj_value = 0

        # Current best solution (so far)
        # TODO: method to update it - at each complete iteration
        self.curr_best_sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
        }
        # Objective value of the best solution - to be updated
        # Initialized to -1 - check it to detect 1st iteration
        self.best_obj_value = -1

        ###### The following are variables used during the run to guide some decision
        self.last_truck_was_empty = False
        self.last_truck = []

    ##########################################################################
    ## Solvers

    def solve(self, df_items, df_vehicles):
        tmp_items = pd.DataFrame.copy(df_items)
        tmp_vehicles = pd.DataFrame.copy(df_vehicles)

        # TODO: review lower bound evaluation
        # min_cost, min_n_trucks = self.getLowerBound(tmp_items, tmp_vehicles)
        # print(f"The minimum cost possible is {min_cost} and it is achieved with {min_n_trucks}")

        tmp_items['surface'] = tmp_items['width']*tmp_items['length']
        tmp_items['volume'] = tmp_items['surface']*tmp_items['height']

        # Order according to dimensions/cost ratio
        if "dim_cost_ratio" not in tmp_vehicles.columns:
            tmp_vehicles["volume"] = tmp_vehicles['width']*tmp_vehicles['length']*tmp_vehicles['height']
            tmp_vehicles["dim_cost_ratio"] = tmp_vehicles['volume']/tmp_vehicles['cost']

        ord_vehicles = tmp_vehicles.sort_values(by=['dim_cost_ratio'], ascending=False)
        
        # Used to track the different types of used vehicles and assign unique IDs:
        n_trucks = {}
        for id in ord_vehicles.id_truck.unique():
            n_trucks[id] = 0

        i = 0
        while len(tmp_items.index) > 0 and i < MAX_ITER:
            print(f"Iter {i}")
            if DEBUG:
                print(f"> Items left: {len(tmp_items.index)}")

            # Strategy for selecting the trucks
            curr_truck = self.selectNextTruck(ord_vehicles, tmp_items, self.last_truck)    
            self.last_truck.append(str(curr_truck.id_truck))

            # Having selected the truck type, update its ID by appending the counter found in n_trucks
            # NOTE: the padding done in this case allows for at most 999 trucks of the same type...
            n_trucks[curr_truck.id_truck] += 1
            curr_truck.id_truck = f"{curr_truck.id_truck}_{str(n_trucks[curr_truck.id_truck]).zfill(3)}"

            if DEBUG:
                print(f"Truck ID: {curr_truck.id_truck}")

            # Iterate over the vehicles to pack first 'better' trucks
            # TODO: find more efficient solution for reading all rows one at a time (if possible)
            # curr_truck = ord_vehicles.iloc[i]
            # print(curr_truck)

            # Build stacks with the copied list of items 'tmp_items'
            valid_stacks_list = self.create_stack(tmp_items, curr_truck)

            if DEBUG:
                print(f"Total number of generated stacks: {len(valid_stacks_list)}")

            # Solve 2D problems to place the stacks
            sol_2D = self.solve2D(valid_stacks_list, curr_truck, df_items)

            # Use the 2D solution to update the overall solution
            tmp_items = self.updateCurrSol(sol_2D, curr_truck, tmp_items)

            i += 1

        self.curr_obj_value = self.evalObj()

        print(f"Current optimal value: {self.curr_obj_value}")

        used_trucks = 0
        for t in n_trucks.keys():
            used_trucks += n_trucks[t]

        print(f"Number of trucks analyzed: {used_trucks}")
        print(f"Actual number of used trucks: {len(list(set(self.curr_sol['idx_vehicle'])))}")

        # Append best solution for current truck
        # Need to make sure the items left have been updated
        df_sol = pd.DataFrame.from_dict(self.curr_sol)
        df_sol.to_csv(
            os.path.join('results', f'{self.name}_sol.csv'),
            index=False
        )
    
    def selectNextTruck(self, trucks_df, remaining_items, forbidden_trucks=[]):
        """ 
        selectNextTruck
        ---
        Choose the best truck to be selected next, depending on the amount of items left.

        Introduce random behavior: TODO (maybe)

        ### Input parameters
        - trucks_df: dataframe containing all available trucks - it should contain the 
        column "volume" and "dim_cost_ratio".
        - remaining_items: dataframe containing the remaining items; it should contain 
        the column "volume"
        """
        # TODO: introduce some random behavior, as the current procedure gets stuck...

        if "volume" not in remaining_items.columns:
            remaining_items["volume"] = remaining_items["length"]*remaining_items["width"]*remaining_items["height"]
        
        tot_item_vol = sum(remaining_items["volume"])

        if "volume" not in trucks_df.columns:
            trucks_df["volume"] = trucks_df["length"]*trucks_df["width"]*trucks_df["height"]

        if "dim_cost_ratio" not in trucks_df.columns:
            trucks_df["dim_cost_ratio"] = trucks_df["volume"]/trucks_df["cost"]

        if all(trucks_df.volume < tot_item_vol):
            # If the volume of all trucks is lower than the overall volume: 
            # return truck with highest dim/cost ratio
            ord_vehicles = trucks_df.sort_values(by=['dim_cost_ratio'], ascending=False)
            
            return ord_vehicles.iloc[0]
        else:
            # Else: return the truck with the lowest cost among the ones which are bigger than 
            # the whole volume
            valid_trucks = trucks_df[trucks_df.volume >= tot_item_vol]
            ord_vehicles = valid_trucks.sort_values(by=['cost'], ascending=True)

            return ord_vehicles.iloc[0]

    def create_stack(self, df_items, truck):
        """
        create_stack
        ---
        Given an object dataframe and a truck, create stacks which 
        can be placed into the truck.

        ### Input parameters:
        - df_items: pandas Dataframe of usable items.
        - truck: pandas Series object containing the truck information.

        FIXME: *Approach*

        Having isolated all stackability codes, iterate on all items for each code value.
        Place each item in a stack, until an item does not pass the checks for being added.
        Once this happens, close current stack and add it to the list of stacks, then 
        start a new stack by placing the current item in a new one.

        This method makes use of the Stack class and its method(s) 'add_item' (and 
        'add_item_override').

        Stacks can be created only for items with the same stackability code.
        """
        stack_codes = df_items.stackability_code.unique()
        stacks_list = []        # Outcome of this function
        
        for code in stack_codes:
            tot_high = 0
            tot_weight = 0
            new_stack_needed = False
            other_constraints = {           # Enforce constraints on the 
                "max_height": truck["height"],
                "max_weight": truck["max_weight_stack"],
                "max_dens": truck["max_density"]
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
                if was_added == 0:
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

    def solve2D(self, stacks, truck, items_df):
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
        if DEBUG:
            print("Started solution in 2D")
       
        # Create a copy, which will be modified (elem. removed)
        up_stacks = stacks.copy()
        # Solution based on the bound given by the stacks which were already placed
        # Extract truck info
        x_truck = truck["length"]
        y_truck = truck["width"]
        # No need for height (we are solving 2D currently)
        max_weight = truck["max_weight"]

        # This solution simply consists of a 2D bin packing with no constraint but the 
        # dimensions of the bin and max_weight: it is assumed the stacks have been built 
        # satisfying the other constraint (height, density, stack weight)

        # Initialize solution
        # Keep in mind: this is just the 2D solution, the task of the main solver is 
        # that of "translating" this solution into the overall one
        # TODO: check memory efficency...
        sol_2D = {
            "x_sol":[],
            "y_sol":[],
            "stack":[],
            "orient": []
        }

        ########### Initialize bound
        bound = [[0,0],[0,y_truck]]

        # TODO: add iteration

        space_left = True

        while space_left:
            # 1. Assign prices to each stack:
            self.priceStack(up_stacks)

            curr_stacks_n = len(up_stacks)

            # 2. Build slices - choose stacks according to highest price
            # Brute force
            # TODO: understand what is the best value of x_dim to be passed to the method below
            rightmost = max([p[0] for p in bound])
            x_dim = x_truck - rightmost
            new_slice = self.buildSlice(up_stacks, x_dim, y_truck)

            assert (len(up_stacks) + len(new_slice) == curr_stacks_n), "Something went wrong! The stacks don't add up"

            if len(new_slice) > 0:
                # Having built the slice:
                # 'Push' stack towards bottom

                # TODO: create 'pushSlice' method
                sol_2D, bound = self.pushSlice(bound, new_slice, sol_2D)

                assert (bound[-1][1] == truck["width"]), "Bound was built wrong!"

            else:
                # If the new slice is empty, close the bin
                # Maybe can also check for big spaces to fill with arbitrary slices
                # but tricky (buildSlice can be used for arbitrary dimensions)
                if len(sol_2D['x_sol']) == 0:
                    print("Cannot fit any item in this truck!")
                    self.last_truck_was_empty = True

                ## Translate solution into 3D one
                # First, build lists of the same size, then assign them
                space_left = False
                self.updateCurrSol(sol_2D, truck, items_df)

        # Something else?
        
        return sol_2D

    def priceStack(self, stacks):
        """
        pricesStack
        ---
        Assign to each stack a price, which will be used to choose which
        one to place first when solving the 2D bin packing problem.

        There are 4 different rules to assign the price, chosen randomly:
        - Price = Area
        - Price = length
        - Price = width
        - Price = perimeter

        TODO: think of new proces to assign
        - total volume
        - delta height wrt truck
        - number of items

        The input variable 'stacks' is a list of Stack objects.
        This method updates the 'price' attribute inside each Stack object.
        """
        # Select which pricing type
        val = random.randint(0,3)

        if val == 0:
            for i in range(len(stacks)):
                stacks[i].assignPrice(stacks[i].area)
        elif val == 1:
            for i in range(len(stacks)):
                stacks[i].assignPrice(stacks[i].length)
        elif val == 2:
            for i in range(len(stacks)):
                stacks[i].assignPrice(stacks[i].width)
        elif val == 3:
            for i in range(len(stacks)):
                stacks[i].assignPrice(stacks[i].perimeter)

    def buildSlice(self, stacks, x_dim, y_dim):
        """
        buildSlice
        ---
        This method is used to populate slices of the trailer to be filled.
        This is done by choosing first slices with higher 'price'.

        ### Input parameters:
        - stacks: list of Stack object, need price to be initialized already.
        - x_dim: available space in the x direction (length)
        - y_dim: available space in the y direction (width) - slice is built 
        along this direction

        ### Output variables:
        - new_slice: list of sublists; each sublist contains:
          - Stack object
          - Index of the stack in the initial list (TODO: check if needed)
          - Rotation - 0 if not rotated, 1 if rotated
          - y coordinate of the origin

        Note that this method can be used to build slices of arbitrary 2D 
        dimensions, so it may also be used to fill spaces with smaller/fewer boxes...

        This method contains the main procedure used to fill the truck. 
        To change strategy, just change this function.
        """
        new_slice = []

        # Sort the stacks according to price
        stacks.sort(key=lambda x: x.price, reverse=True)
        
        i = 0       # i tracks the index of the stack list
        j = 0       # j tracks the number of stacks in the current slice
        delta_y = y_dim
        # Until all possible stacks have been visited, try to add new one to fill slice
        # NOTE: this can be improved in the future, e.g., by finding optimal slice at each
        # iteration, in terms of minimum delta_y left
        # For now, I will keep this approach as it follows what explained in the paper...
        while i < len(stacks):
            # if DEBUG:
            #     print(f"Analyzing stack {i}")
            stack_added = False
            if len(stacks[i].items) > 0:        # (Don't know why) but sometimes stacks are created empty...
                if delta_y >= stacks[i].width and x_dim >= stacks[i].length:
                    # Stack is good as-is - insert it
                    new_slice.append([stacks[i], i, 0])

                    if DEBUG:
                        print("Stack added to slice!")

                    delta_y -= stacks[i].width
                    stack_added = True
                elif stacks[i].forced_orientation == "n" and delta_y >= stacks[i].length and x_dim >= stacks[i].width:
                    # If the stack cannot be placed, try rotating it by 90 degrees, if allowed
                    new_slice.append([stacks[i], i, 1])

                    if DEBUG:
                        print("Stack added to slice! (2)")
                    # Rotated stack - can place it width-wise
                    delta_y -= stacks[i].length
                    stack_added = True

            if stack_added:
                # Update origin y coordinate
                if j != 0:
                    # Get width (length if rotated) of 2nd to last element
                    if new_slice[-2][2] == 0:
                        if DEBUG:
                            print("Success here")
                        w_min2 = new_slice[-2][0].width
                    else:
                        if DEBUG:
                            print("Success here (2)")
                        w_min2 = new_slice[-2][0].length
                    # Add the width to the origin of the stack to get new origin
                    # This ensures no space is left
                    new_slice[-1].append(new_slice[-2][-1] + w_min2)
                else:
                    # Stack is placed at y = 0
                    new_slice[-1].append(0)
                
                j += 1

            i += 1
        # When out of the loop, the slice has been built 
        # NOTE: this is not the optimal slice in terms of delta_y left! 
        # This is the best stack in terms of maximum price, but we are sure 
        # that in the delta_y left no other item can be placed!

        # TODO: its possible to think of a way to solve the slice filling sub-problem with
        #  other approaches (e.g., Gurobi/OR tools)

        # Remove used stacks from the initial list
        # This modifies the 'stacks' list which is passed to the function
        for i in [x[1] for x in new_slice[::-1]]:
            del stacks[i]

        # Technically the indices of the stacks are not used anymore (and cannot be used...)
        
        if DEBUG:
            print(f"N. stacks in new slice: {len(new_slice)}")
        
        return new_slice

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
        TODO
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
            while bound[ind_bound][1] < y_i:
                ind_bound += 1

            if bound[ind_bound][1] == y_i:
                pass
            else:
                ind_bound -= 1
            
            # Search for valid points
            ind_top = ind_bound + 0             # Needed to prevent to just copy the reference and update both indices...
            while bound[ind_top][1] < y_i+w_i:
                ind_top += 1
            # When the loop finishes, the element bound[ind_top] contains the upper end 
            
            assert (len(bound[ind_bound:ind_top+1]) > 1), "The considered elements of the bound are less than 2! Something went wrong"

            # The x coordinate is the max between the x coord of the elements of 
            # index between ind_bound and ind_top
            x_i = max([p[0] for p in bound[ind_bound:ind_top+1]])
            
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

            while bound[ind_extra][1] < new_bound[-1][1]:
                ind_extra += 1

            # ind_extra locates the 1st corner in the old bound which has y bigger 
            # than the current last element in the new bound

            # Add adjustment point:
            # x is the one of the old bound
            # y is the same as the last element in the current bound
            new_bound.append([bound[ind_extra][0], new_bound[-1][1]])

            for p in bound[ind_extra:]:
                new_bound.append(p)

        return curr_sol_2D, new_bound
            
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
                self.curr_sol["type_vehicle"].append(truck["cost"])
                self.curr_sol["idx_vehicle"].append(truck["id_truck"])
                self.curr_sol["id_stack"].append(sol_2D["stack"][i].id)
                self.curr_sol["id_item"].append(it["id_item"])
                self.curr_sol["x_origin"].append(sol_2D["x_sol"][i])
                self.curr_sol["y_origin"].append(sol_2D["y_sol"][i])
                self.curr_sol["z_origin"].append(z_lst[j])
                self.curr_sol["orient"].append(sol_2D["orient"][i])
                j += 1

                # Remove used items from the items DF
                upd_items = upd_items[upd_items.id_item != it["id_item"]]

        return upd_items

    def evalObj(self, sol=None):
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
        # FIXME: don't know why but it works - seems a bit weak...
        for t_id in list(set(s["idx_vehicle"])):
            o_val += float(s["type_vehicle"][s["idx_vehicle"] == t_id])

    ##########################################################################
    ## Utilities

    def getLowerBound(self, df_items, df_trucks):
        """
        getLowerBound
        ---
        Obtain the lower bound on the number of trucks & objective function cost
        for the solution of the problem
        
        FIXME: fix behavior - any type of truck can be selected as many times as possible
        """
        # Get overall volume of the items
        df_items["volume"] = df_items['length']*df_items["width"]*df_items['height']
        tot_item_vol = sum(df_items["volume"])

        print(f"Total items volume: {tot_item_vol}")

        # Get volume of trucks
        df_trucks["volume"] = df_trucks["height"]*df_trucks["width"]*df_trucks["height"]
        print(df_trucks["volume"])
        # Get dim/cost ratio
        df_trucks["dim_cost_ratio"] = (df_trucks['width']*df_trucks['length']*df_trucks['height'])/df_trucks['cost']

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

    ##########################################################################
    ## Displaying the solution

    def myStack3D(df_items, df_vehicles, df_sol, idx_vehicle=0):
        """
        myStack3D
        ---

        """
        pass
