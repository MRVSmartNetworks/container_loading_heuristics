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
        
        stack_lst = self.buildStacks(vehicle)
        pass


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
    
    def ACO_2D_bin(self, stack_lst, alpha = 1, beta = 1, n_ants = 10):
        """ 
        Ant Colony Optimization for 2D bin packing
        ---
        #### INPUT PARAMETERS:
            - stack_lst: list of stacks (format: [[I001, I002, NaN], [I003, I005, NaN]])
            - alpha: realative trail importance (default 1)
            - beta: relative attractivness importance (default 1)
            - n_ants: number of ants
        #### ACO PARAMETERS:
            - attr: matrix of attractiveness from state i to j
            - trail: matrix of trails from state i to j
            - pr_move: matrix of probabilities of moves from i to j 
        """
        #TODO: outer loop contaning a termination condition (no. of iterations, solution's goodness???)
        ants = []
        for ant_k in range(n_ants):
            free_space = True   
            while(free_space):  # loop until free space available in vehicle
                #stack_toAdd = #TODO:function to retrieve random from pr_move 
                pass



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


    