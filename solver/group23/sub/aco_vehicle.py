import numpy as np
import os
import pandas as pd
from sub.ACO import ACO
from sub.aco_bin_packing import aco_bin_packing
from sub.utilities import *
import time
from sub.projection import *

class aco_vehicle(ACO):
    def __init__(self, df_items, df_vehicles, alpha=1, beta=1, 
                 n_ants=1, n_iter=1, evaporationCoeff=0.5):
        super().__init__(alpha, beta, n_ants, n_iter, evaporationCoeff)
        self.df_items = df_items
        print(len(df_items))
        self.df_vehicles = df_vehicles
        self.N = len(df_vehicles) + 1   # no. of rows for pr_move, attract, trailMatrix
        self.pr_move = np.full((self.N, self.N),1/(self.N-1)) * [1,1,1,1,1,1,1,0] # initialize pr_move with equal probabilities
        self.trailMatrix = np.zeros((self.N, self.N))
        

    def aco_vehicle_sol(self):
        st_time = time.time()
        for _ in range(self.n_iter):
            self.ants = []
            antsCost = []
            totCost = 0
            totItems = 0
            for _ in range(self.n_ants):
                more_items = True
                ant_k = []
                prev_vehicle = self.N - 1
                df_items_ant = self.df_items.copy()
                aco = aco_bin_packing()
                i = 0
                while(more_items):
                    i += 1
                    print("Truck number:",i)
                    next_vehicle = self.choose_move(prev_vehicle)
                    aco.buildStacks(vehicle=self.df_vehicles.iloc[1], df_items=df_items_ant)
                    if sum(aco.stack_quantity) != 0:#TODO: change please
                        aco.statesCreation(self.df_items[["length", "width", "stackability_code", "forced_orientation"]].drop_duplicates())
                        # aco.changeVehicle(self.df_vehicles.iloc[next_vehicle])
                        sol_truck = aco.aco_2D_bin()
                        #sol_truck = pd.DataFrame.from_dict(sol_truck)
                        to_remove_items = sol_truck["id_item"]
                        df_items_ant = df_items_ant[df_items_ant.id_item.isin(to_remove_items) == False]
                        totItems += len(to_remove_items)
                        
                        
                        ant_k.append(self.df_vehicles.iloc[next_vehicle]["id_truck"])
                        totCost += self.df_vehicles.iloc[next_vehicle]["cost"]
                        prev_vehicle = next_vehicle
                    else:
                        break
                    """i += 1
                    if i>50:
                        df_sol = pd.DataFrame.from_dict(aco.sol)
                        df_sol.to_csv(os.path.join('results', f'solver23_sol.csv'),
                        index=False)
                        df_sol = pd.read_csv(os.path.join('results', f'solver23_sol.csv'),)
                        orthogonal_plane(self.df_items, self.df_vehicles, df_sol)"""
                print("\nTime:", time.time() - st_time)
                print("\nN trucks = ", len(ant_k))
                print("Tot items: ", totItems)
                break
                self.ants.append(ant_k)
                antsCost.append(totCost)
    
    