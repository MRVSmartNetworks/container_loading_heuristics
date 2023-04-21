import os
import math
import random
import pandas as pd
import numpy as np
import itertools

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

        Input parameters:
        - df_items: pandas Dataframe of usable items.
        - truck: pandas Series object containing the truck information.

        *Approach*

        Stacks can be created only for items with the same stackability code.
        """
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

        # Initialize bound
        bound = [[0,0],[0,y_truck]]

        # 1. Assign prices to each stack:
        self.priceStack(up_stacks)

        # 2. Build slices - choose stacks according to highest price
        # Brute force
        # TODO: understand what is the best value of x_dim to be passed to the method below
        rightmost = max([p[0] for p in bound])
        x_dim = x_truck - rightmost
        new_slice = self.buildSlice(up_stacks, x_dim, y_truck)

        

        pass
    
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

        Note that this method can be used to build slices of arbitrary 2D 
        dimensions, so it may also be used to fill spaces with smaller/fewer boxes...
        """
        new_slice = []
        stacks_ids = []

        # Sort the stacks according to price
        stacks.sort(key=lambda x: x.price, reverse=True)
        
        i = 0
        delta_y = y_dim
        # Until all possible stacks have been visited, try to add new one to fill slice
        # NOTE: this can be improved in the future, e.g., by finding optimal slice at each
        # iteration, in terms of minimum delta_y left
        # For now, I will keep this approach as it follows what explained in the paper...
        while i < len(stacks):
            if delta_y > stacks[i].width:
                new_slice.append(stacks[i])
                stacks_ids.append(i)
                delta_y -= stacks[i].width
            elif stacks[i].forced_orientation == "n" and delta_y > stacks[i].length:
                new_slice.append(stacks[i])
                stacks_ids.append(i)
                delta_y -= stacks[i].length

            ######################## TODO: keep track of the 'x' dimension

            i += 1



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

        


        
