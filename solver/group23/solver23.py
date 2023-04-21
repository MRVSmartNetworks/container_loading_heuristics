#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd

class Solver23():
    def __init__(self):
        """
        Solver: 
        """
        self.name = "solver23"

    def solve_single_vehicle(self, df_items, df_vehicles):
        """ 
        solve_single_vehicle
        -----
        test function to work with ACO on the 2D bin packing of
        a single vehicle
        """
        self.df_items = df_items
        self.df_vehicles = df_vehicles
        
        # for on truck's type
        # work on single truck
        vehicle = self.df_vehicles.iloc[0] # vehicle type V0

        stack_lst = self.buildStacks(self, vehicle)


    def buildStacks(self, vehicle):
        """"
        buildStacks
        -----------
        vehicle: vehicle type, needed to check the height for
                 creating the stacks for this specific truck
        """
        
        # stack creation:
        stackability_codes = self.df_items.stackability_code.unique()
        stack_lst = []
        for code in stackability_codes:
            stack = []
            tot_high = 0
            tot_weight = 0
            new_stack_needed = False
            for i, row in self.df_items[self.df_items.stackability_code == code].iterrows():
                tot_high += row.height - row.nesting_height
                tot_weight += row.weight
                if tot_high > vehicle['height']:
                    new_stack_needed = True
                if tot_weight > vehicle['max_weight_stack']:
                    new_stack_needed = True
                if len(stack) == row.max_stackability:
                    new_stack_needed = True
                # if a new stack is needed:
                if new_stack_needed:
                    stack_lst.append(stack)
                    stack = [row.id_item]
                    tot_high = row.height
                    tot_weight = row.weight
                    new_stack_needed = False
                else:
                    # else add the item
                    stack.append(row.id_item)
        
        return stack_lst
    


    def solve():
        """ 
        solve
        -----
        df_items: dataframe containing all the items 
                  that are to be put into the trucks
        df_vehicles: dataframe containing all the different
                     types of trucks that can be choose
        """
        pass


if __name__ == "__main__":
    
    sol = Solver23()
    sol.single_truck()
