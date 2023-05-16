#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from sub.utilities import *
from sub.aco_bin_packing import ACO
from sub.projection import *
import time

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
        self.id_vehicle = 0
        self.sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
        }
    def solve(self, df_items, df_vehicles):
        """ 
        solve
        -----
        
        - df_items: dataframe containing all the items 
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        """
        st_time = time.time()
        vehicle = self.vehicle_decision(df_vehicles)
        stackInfo = df_items[["length", "width", "stackability_code", "forced_orientation"]].drop_duplicates()
        aco = ACO(stackInfo)
        more_items = True
        while(more_items):
            aco.buildStacks(vehicle=vehicle, df_items=df_items)
            if sum(aco.stack_quantity.values()) != 0:
                bestAnt = aco.aco_2D_bin()
                self.solUpdate(bestAnt, vehicle)
                df_items = df_items[df_items.id_item.isin(self.sol["id_item"]) == False]
            else:
                more_items = False
        
        df_sol = pd.DataFrame.from_dict(self.sol)
        print("\nN trucks = ", df_sol['idx_vehicle'].nunique())
        print("Tot items: ", len(self.sol["id_item"]))
        print("\nTime:", time.time() - st_time)
    
    def solUpdate(self, bestAnt, vehicle):
        lenSol = len(self.sol['id_stack'])
        for i,stack in enumerate(bestAnt):
            z_origin = 0
            for item in stack.items:
                self.sol['type_vehicle'].append(vehicle['id_truck'])
                self.sol['idx_vehicle'].append(self.id_vehicle)
                self.sol['id_stack'].append(f"S{i+lenSol}")
                self.sol['id_item'].append(item[0])
                self.sol['x_origin'].append(stack.x_origin)
                self.sol['y_origin'].append(stack.y_origin)
                self.sol['z_origin'].append(z_origin)
                self.sol['orient'].append(stack.orient)
                z_origin += item[1]
        self.id_vehicle += 1
    
    def vehicle_decision(self, df_vehicles):
        eff_ratio = []
        for _, vehicle in df_vehicles.iterrows():
            volume = vehicle['length'] * vehicle['width'] * vehicle['height']
            eff_ratio.append((volume * vehicle['max_weight'])/vehicle['cost'])
        max_index = eff_ratio.index(max(eff_ratio))
        return df_vehicles.iloc[max_index]

    def solve_single_vehicle(self, df_items, df_vehicles):
        """ 
        solve_single_vehicle
        --------------------

        test function to work with ACO on the 2D bin packing of a single vehicle
                
        - df_items: dataframe containing all the items 
                    that are to be put into the trucks
        - df_vehicles: dataframe containing all the different
                       types of trucks that can be choose
        """
        self.df_items = df_items
        self.df_vehicles = df_vehicles
        
        # for on truck's type
        # work on single truck
        vehicle = self.df_vehicles.iloc[1] # vehicle type V*
        
        aco = ACO()
        aco.buildStacks(vehicle, df_items)
        aco.statesCreation(df_items[["length", "width", "stackability_code", "forced_orientation"]].drop_duplicates())
        aco.aco_2D_bin()

        df_sol = pd.DataFrame.from_dict(aco.sol)
        df_sol.to_csv(
            os.path.join('results', f'{self.name}_sol.csv'),
            index=False
        )

        
                
            


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