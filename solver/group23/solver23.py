#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from numpy.random import choice
from sub.stack import Stack
from sub.utilities import *

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

        self.ACO_2D_bin(stack_lst, alpha = 1, beta = 1, n_ants = 10)



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
            stack_feat = getStackFeatures(self.df_items, code)
            
            stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
            tot_high = 0
            tot_weight = 0
            new_stack_needed = False
            for i, row in self.df_items[self.df_items.stackability_code == code].iterrows():
                stack.updateHeight(row.height - row.nesting_height)
                stack.updateWeight(row.weight)
                if stack.height > vehicle['height']:
                    new_stack_needed = True
                if stack.weight > vehicle['max_weight_stack']:
                    new_stack_needed = True
                if stack.n_items == row.max_stackability:
                    new_stack_needed = True
                # if a new stack is needed:
                if new_stack_needed:
                    stack_lst.append(stack)
                    stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
                    stack.addItem(row.id_item)
                    tot_high = row.height
                    tot_weight = row.weight
                    new_stack_needed = False
                else:
                    # else add the item
                    stack.addItem(row.id_item)
                    #TODO: control max_weight of vehicle
                    
        
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
            - attr(η): matrix of attractiveness from state i to j
            - trail(τ): matrix of trails from state i to j #TODO:6x6 matrix considering stackability codes
            - pr_move: matrix of probabilities of moves from i to j 
        """
        attr = np.random.rand(6, 6) #NOTE: take random value for attractivenss matrix just to test
        pr_move = np.full((6,6), 1./12) # initialize pr_move with same prob for each movement
        #TODO: outer loop contaning a termination condition (no. of iterations, solution's goodness???)
        ants = []
        for ant_k in range(n_ants):
            free_space = True 
            empty_vehicle = True
            
            while(free_space):  # loop until free space available in vehicle
                code_toAdd = self.choose_move(pr_move, Stack.stack_code) #BUG: consider the initial case with empty veichle 
                # ants[ant_k].append()
                new_stack, stack_lst = popStack(stack_lst, code_toAdd)
                pass


    def choose_move(self, pr_move, stack_code):
        """ 
        choose_move
        -----
        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - stack_code: stackability code of the last stack placed into the truck
        #### OUTPUT PARAMETERS:
            - stack_code: stackability code of the next stack to be placed into the truck
        """
        row_to_choose = pr_move[stack_code][:] # select the row from the stack the ant is moving
        stack_code = choice(range(len(row_to_choose)), p=row_to_choose) + 1 # indexes start from 0
        return stack_code 
    

    def solve(self, df_items, df_vehicles):
        """ 
        solve
        -----
        df_items: dataframe containing all the items 
                  that are to be put into the trucks
        df_vehicles: dataframe containing all the different
                     types of trucks that can be choose
        """
        self.solve_single_vehicle(df_items[:200][:], df_vehicles)
        


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


    