import numpy as np
import pandas as pd
from sub.ACO import ACO
from sub.aco_bin_packing import aco_bin_packing
from sub.utilities import *
import time

class aco_vehicle(ACO):
    def __init__(self, df_items, df_vehicles, alpha=1, beta=1, 
                 n_ants=40, n_iter=20, evaporationCoeff=0.5):
        super().__init__(alpha, beta, n_ants, n_iter, evaporationCoeff)
        self.df_items = df_items
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
            for _ in range(self.n_ants):
                more_items = True
                ant_k = []
                prev_vehicle = self.N - 1
                df_items_ant = self.df_items.copy()
                aco = aco_bin_packing()
                while(more_items):
                    next_vehicle = self.choose_move(prev_vehicle)
                    aco.buildStacks(vehicle=self.df_vehicles.iloc[next_vehicle], df_items=df_items_ant)
                    if sum(aco.stack_quantity) == 0:#TODO: change please
                        break
                    aco.statesCreation(self.df_items[["stackability_code","forced_orientation"]].drop_duplicates())
                    aco.changeVehicle(self.df_vehicles.iloc[next_vehicle])
                    sol_truck = aco.aco_2D_bin()
                    sol_truck = pd.DataFrame.from_dict(sol_truck)
                    to_remove_items = list(sol_truck["id_item"])
                    df_items_ant = df_items_ant[df_items_ant.id_item.isin(to_remove_items) == False]
                    if not df_items_ant.empty:
                        ant_k.append(self.df_vehicles.iloc[next_vehicle]["id_truck"])
                        totCost += self.df_vehicles.iloc[next_vehicle]["cost"]
                        prev_vehicle = next_vehicle
                    else:
                        more_items = False
                print("\nTime:", st_time - time.time())
                self.ants.append(ant_k)
                antsCost.append(totCost)
    
    