import os
import math
import pandas as pd

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

        # Iterate over trucks (order?)
        # Order according to dimensions/cost ratio
        tmp_vehicles["dim_cost_ratio"] = (tmp_vehicles['width']*tmp_vehicles['length']*tmp_vehicles['height'])/tmp_vehicles['cost']
        # print(tmp_vehicles)

        ord_vehicles = tmp_vehicles.sort_values(by=['dim_cost_ratio'], ascending=False)

        for i in range(len(ord_vehicles.index)):
            # Iterate over the vehicles to pack first 'better' trucks

            curr_truck = ord_vehicles.iloc[i]

            print(curr_truck)

        # Start filling from bottom - find best combination 
        # of widths that fills the full width of the truck
        # OR
        # Fill with equal-dimension items, and build stacks

        # Proceed with possible solutions 
        return

        
