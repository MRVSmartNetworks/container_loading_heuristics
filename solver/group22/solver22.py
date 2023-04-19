import os
import math
import pandas as pd
import numpy as np

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

    def solve(self, df_items, df_vehicles):
        tmp_items = pd.DataFrame.copy(df_items)
        tmp_vehicles = pd.DataFrame.copy(df_vehicles)

        tmp_items['surface'] = tmp_items['width']*tmp_items['length']
        # Order items according to the surface
        ord_items = tmp_items.sort_values(by=['surface'], ascending=False)

        # Iterate over trucks (order?)
        # Order according to dimensions/cost ratio
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

        return
    
    def create_stack(self, df_items, truck):
        pass
    
    def fill_width(self, df_items, truck):
        pass

        
