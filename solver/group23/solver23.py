#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from numpy.random import choice
from sub.stack import Stack
from sub.utilities import *
#TODO:
# - costruire dimensione della matrice considerando orientation (aggiungere stato per orientation)
# - iniziare a mettere gli stack nel camion
# - 
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

        self.pr_move = statesCreation(df_items[["stackability_code","forced_orientation"]].drop_duplicates())
        
        # for on truck's type
        # work on single truck
        vehicle = self.df_vehicles.iloc[0] # vehicle type V0
        
        stack_lst = self.buildStacks(vehicle)

        self.ACO_2D_bin(stack_lst, vehicle,alpha = 1, beta = 1, n_ants = 10)

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
        #TODO: controllo max density
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
                    
        return stack_lst
        
    def ACO_2D_bin(self, stack_lst, vehicle, alpha = 1, beta = 1, n_ants = 10):
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
            - trail(τ): matrix of trails from state i to j
            - pr_move: 8x8 matrix of probabilities of moves from i to j (state 7 is related to empty vehicle)
        """
        attr = np.random.rand(6, 6) #NOTE: take random value for attractivenss matrix just to test
        # initialize pr_move with same prob for each movement but set to zero prob to move to no stack
        pr_move = np.full((8,8), 1./7) * np.array([1, 1, 1, 1, 1, 1, 1, 0]) #TODO: create automatically the matrix
        #TODO: outer loop contaning a termination condition (no. of iterations, solution's goodness???)
        ants = []
        for ant_k in range(n_ants):
            free_space = True 
            prev_s_code = 6 # empty vehicle state
            # initialize solution of ant k
            sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
            }
            x_pos = 0
            y_pos = 0
            while(free_space):  # loop until free space available in vehicle
                next_s_code = self.choose_move(pr_move, prev_s_code) 
                # ants[ant_k].append()
                new_stack, stack_lst = popStack(stack_lst, next_s_code) #TODO: what if no more stacks with this stack code??
                #TODO: se widthwise mettere come length la width
                #TODO: posizionare new_stack

                prev_s_code = next_s_code
                #TODO: controllo se free space
                
    def choose_move(self, pr_move, prev_s_code):
        """ 
        choose_move
        -----
        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - prev_s_code: stackability code of the last stack placed into the truck
        #### OUTPUT PARAMETERS:
            - next_s_code: stackability code of the next stack to be placed into the truck
        """
        row_to_choose = pr_move[prev_s_code][:] # select the row from the stack the ant is moving
        next_s_code = int(choice(range(len(row_to_choose)), p=row_to_choose))
        
        return next_s_code 
    
    def addStack(self, toAddStack, x_pos, vehicle):
        """  
        
        """
        if x_pos + toAddStack.length < vehicle['length']:
            pass
        else:
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
        #TODO: take random 200 items for less computation (random for getting different stack code)
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