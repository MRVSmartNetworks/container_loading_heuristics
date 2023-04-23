# -*- coding: utf-8 -*-
import itertools
import math
import os
import random

import numpy as np
import pandas as pd

from solver.group22.stack import Stack


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

        # self.current_stacks

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

    def solve(self, df_items, df_vehicles):
        tmp_items = pd.DataFrame.copy(df_items)
        tmp_vehicles = pd.DataFrame.copy(df_vehicles)

        min_cost, min_n_trucks = self.getLowerBound(tmp_items, tmp_vehicles)
        
        print(f"The minimum cost possible is {min_cost} and it is achieved with {min_n_trucks}")

        tmp_items['surface'] = tmp_items['width']*tmp_items['length']
        # Order items according to the surface
        ord_items = tmp_items.sort_values(by=['surface'], ascending=False)

        # Iterate over trucks (order?)
        # Order according to dimensions/cost ratio
        if "dim_cost_ratio" not in tmp_vehicles.columns:
            tmp_vehicles["dim_cost_ratio"] = (tmp_vehicles['width']*tmp_vehicles['length']*tmp_vehicles['height'])/tmp_vehicles['cost']
        # print(tmp_vehicles)

        ord_vehicles = tmp_vehicles.sort_values(by=['dim_cost_ratio'], ascending=False)

        # Initialize solution - list of lists (one for each truck, in order)
        final_sol = []

        for i in range(len(ord_vehicles.index)):
            # Iterate over the vehicles to pack first 'better' trucks
            # TODO: find more efficient solution for reading all rows one at a time (if possible)
            curr_truck = ord_vehicles.iloc[i]
            # print(curr_truck)

            curr_sol = {
                "type_vehicle": [],
                "idx_vehicle": [],
                "id_stack": [],
                "id_item": [],
                "x_origin": [],
                "y_origin": [],
                "z_origin": [],
                "orient": []
            }

            # Start filling from bottom - find best combination 
            # of widths that fills the full width of the truck
            # OR
            # Fill with equal-dimension items, and build stacks

        # Proceed with possible solutions 

        # Append best solution for current truck
        # Need to make sure the items left have been updated
        final_sol.append(curr_sol)

        with open(os.path.join('results', f'{self.name}_sol.csv'), "w") as f:
            f.close()

        return
    
    def create_stack(self, df_items, truck):
        """
        create_stack
        ---
        Given an object dataframe and a truck, create stacks which 
        can be placed into the truck.

        ### Input parameters:
        - df_items: pandas Dataframe of usable items.
        - truck: pandas Series object containing the truck information.

        *Approach*

        Stacks can be created only for items with the same stackability code.
        """
        ###################### TODO: Next up
        
        pass
    
    def fill_width(self, df_items, truck):
        pass

    def solve2D(self, stacks, truck):
        """
        solve2D
        ---
        Solution of the 2D problem (which is performed once the stacks are provided).
        """
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
            # Having built the slice (and removed stacks from the ):
            # 'Push' stack towards bottom

            # TODO: create 'pushSlice' method
            sol_2D = self.pushSlice(bound)
        else:
            # If the new slice is empty, close the bin
            # Maybe can also check for big spaces to fill with arbitrary slices
            # but tricky (buildSlice can be used for arbitrary dimensions)
            
            ## Translate solution into 3D one
            # First, build lists of the same size, then assign them
            
            self.updateCurrSol(sol_2D, truck)

        # Something else?

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
        
        i = 0
        delta_y = y_dim
        # Until all possible stacks have been visited, try to add new one to fill slice
        # NOTE: this can be improved in the future, e.g., by finding optimal slice at each
        # iteration, in terms of minimum delta_y left
        # For now, I will keep this approach as it follows what explained in the paper...
        while i < len(stacks):
            if delta_y >= stacks[i].width and x_dim >= stacks[i].length:
                # Stack is good as-is - insert it
                new_slice.append([stacks[i], i, 0])
                delta_y -= stacks[i].width
            elif stacks[i].forced_orientation == "n" and delta_y >= stacks[i].length and x_dim >= stacks[i].width:
                # If the stack cannot be placed, try rotating it by 90 degrees, if allowed
                new_slice.append([stacks[i], i, 1])
                delta_y -= stacks[i].length

            # Update origin y coordinate
            if i != 0:
                # Get width (length if rotated) of 2nd to last element
                if new_slice[-2][2] == 0:
                    w_min2 = new_slice[-2][0].width
                else:
                    w_min2 = new_slice[-2][0].length
                # Add the width to the origin of the stack to get new origin
                # THis ensures no space is left
                new_slice[i].append(new_slice[-2][-1] + w_min2)
            else:
                new_slice[i].append(0)
            i += 1
        # When out of the loop, the slice has been built 
        # NOTE: this is not the optimal slice in terms of delta_y left! 
        # This is the best stack in terms of maximum price, but we are sure 
        # that in the delta_y left no other item can be placed!

        # Remove used stacks from the initial list
        for i in [x[1] for x in new_slice[::-1]]:
            del stacks[i]

        # Technically the indices of the stacks are not used anymore (and cannot be used...)

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

        """
        # Store the index of the first element in the bound which is valid
        ind_bound = 0
        for new_stack in new_slice:
            y_i = new_stack[3]
            if new_stack[2] == 0:
                w_i = new_stack[0].length
            else:
                w_i = new_stack[0].width
            
            # search for valid points
            ind_top = ind_bound
            while bound[ind_top][1] < y_i+w_i:
                ind_top += 1
            # When the loop finishes, the element bound[ind_top] contains the upper end 

            ########## HERE
            # The lower end considered should either be:
            # - The previous upper end if the point coincides with y_0
            # - The one before (in the bound list) else
            if bound[ind_bound][1] == y_i:
                pass
            else:
                ind_bound -= 1
            
            assert (len(bound[ind_bound:ind_top]) > 1), "The considered elements of the bound are less than 2! Something went wrong"

            # The x coordinate is the max between the x coord of the elements of 
            # index between ind_bound and ind_top
            x_i = max([p[0] for p in bound[ind_bound:ind_top]])
            
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
                        
            # Update the index of the low bound with the current top
            # Needed in order to do the update as before
            ind_bound = ind_top
        
        return curr_sol_2D
            
    def updateCurrSol(self, sol_2D, truck):
        """
        updateCurrSol
        ---
        Update the current solution for this iteration, contained in class
        attribute 'self.curr_sol'.

        ### Input parameters
        - sol_2D: 2D solution - dict containing info on stack placement
        """

        for i in range(len(sol_2D["stack"])):
            z_lst = sol_2D["stack"][i].getZList()
            j = 0
            for it in sol_2D["stack"][i].items:
                self.curr_sol["type_vehicle"].append(truck["cost"])
                self.curr_sol["idx_vehicle"].append(truck["idtruck"])
                self.curr_sol["id_stack"].append(sol_2D["stack"][i].id)
                self.curr_sol["id_item"].append(it["id_item"])
                self.curr_sol["x_origin"].append(sol_2D["x_sol"][i])
                self.curr_sol["y_origin"].append(sol_2D["y_sol"][i])
                self.curr_sol["z_origin"].append(z_lst[j])
                self.curr_sol["orient"].append(sol_2D["orient"][i])

                j += 1

    def getLowerBound(self, df_items, df_trucks):
        """
        getLowerBound
        ---
        Obtain the lower bound on the number of trucks & objective function cost
        for the solution of the problem
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
