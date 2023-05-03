#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sub.stack import Stack
from sub.utilities import *
from sub.ACO import ACO
#TODO:
# - in popStack pensare a cosa fare se non ci sono più stack con quel
#   stack_code
# - aggiornamento pr_move (controllare che somma delle probabilite su
#   sia uguale ad 1)
# - creare matrice attractiveness
# - pensare a come inizializzare pr_move in modo che piaccia a fox
# - creare gli output in modo da avere in stampa le immagini del camion
# - evaluate  ant.weigthSolution: trovare un  modo per valutare la bontà della soluzione

#TODO: less important
# - certi stack code potrebbero avere la stessa dimensione
class Solver23():
    def __init__(self):
        """
        Solver: 
        """
        self.name = "solver23"

    def solve_single_vehicle(self, df_items, df_vehicles):
        """ 
        solve_single_vehicle
        --------------------

        test function to work with ACO on the 2D bin packing of a single vehicle
        """
        self.df_items = df_items
        self.df_vehicles = df_vehicles
        
        # for on truck's type
        # work on single truck
        vehicle = self.df_vehicles.iloc[0] # vehicle type V0
        
        stack_lst = self.buildStacks(vehicle)
        aco = ACO(stack_lst, vehicle)
        aco.pr_move = statesCreation(df_items[["stackability_code",
                                               "forced_orientation"]].drop_duplicates())
        aco.aco_2D_bin()
        

    def buildStacks(self, vehicle):
        """"
        buildStacks
        -----------

        - vehicle: vehicle type, needed to check the height for
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
        
                
    def solve(self, df_items, df_vehicles):
        """ 
        solve
        -----
        - df_items: dataframe containing all the items 
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
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