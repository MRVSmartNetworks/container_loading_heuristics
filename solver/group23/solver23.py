#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from sol_representation import *

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
        
        # stack creation: adesso è fatta in modo molto stupido ma dato che item con lo stesso
        # stackability code possono avere diverse altezze probabilmente si può ottimizzare molto 
        # date le diverse altezze dei trucks
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
    


    def solve(self, df_items, df_vehicles):
        """ 
        solve
        -----
        df_items: dataframe containing all the items 
                  that are to be put into the trucks
        df_vehicles: dataframe containing all the different
                     types of trucks that can be choose
        """
        self.solve_single_vehicle(df_items, df_vehicles)
        


if __name__ == "__main__":
    
    os.path.join('.', 'results', "sol1.csv")
    df_items = pd.read_csv(
        os.path.join(
            '.', 'data',
            'items.csv'
        ),
    )
    df_vehicles = pd.read_csv(
        os.path.join(
            '.', 'data',
            'vehicles.csv'
        ),
    )
    tmp = Solver23()
    tmp.solve(df_items, df_vehicles)
    # read dataframe solution
    df_sol = pd.read_csv(
        os.path.join('results', f'{tmp.name}_sol.csv'),
    )

    orthogonal_plane(df_items, df_vehicles, df_sol)

    stack_3D(df_items, df_vehicles, df_sol)

    