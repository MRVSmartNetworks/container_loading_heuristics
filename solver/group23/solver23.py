#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sub.stack import Stack
from sub.utilities import *
from sub.ACO import ACO
from sub.aco_bin_packing import aco_bin_packing
from sub.aco_vehicle import aco_vehicle
from sub.projection import *

#TODO:
# - in popStack pensare a cosa fare se non ci sono più stack con quel
#   stack_code
# - aggiornamento pr_move (controllare che somma delle probabilite su
#   sia uguale ad 1)
# - creare matrice attractiveness


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
        
        stack_lst, stack_quantity = self.buildStacks(vehicle)
        aco = aco_bin_packing(stack_lst=stack_lst, stack_quantity=stack_quantity, vehicle=vehicle)
        aco.statesCreation(df_items[["stackability_code",
                                               "forced_orientation"]].drop_duplicates())
        aco.aco_2D_bin()

        df_sol = pd.DataFrame.from_dict(aco.sol)
        df_sol.to_csv(
            os.path.join('results', f'{self.name}_sol.csv'),
            index=False
        )

        
                
    def solve(self, df_items, df_vehicles):
        """ 
        solve
        -----
        - df_items: dataframe containing all the items 
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        """
        aco_sol = aco_vehicle(df_items, df_vehicles)
        
        aco_sol.aco_vehicle_sol()
        


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
    pass