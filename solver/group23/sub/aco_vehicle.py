import numpy as np
import pandas as pd
from sub.ACO import ACO
from sub.aco_bin_packing import aco_bin_packing

class aco_vehicle(ACO):
    def __init__(self, df_vehicles, stack_lst, alpha=1, beta=1, 
                 n_ants=40, n_iter=20, evaporationCoeff=0.5):
        super().__init__(alpha, beta, n_ants, n_iter, evaporationCoeff)
        self.df_vehicles = df_vehicles
        self.N = len(df_vehicles) + 1   # no. of rows for pr_move, attract, trailMatrix
        self.pr_move = np.full((self.N, self.N),1/self.N) # initialize pr_move with equal probabilities
        self.trailMatrix = np.zeros((self.N, self.N))
        

    def aco_vehicle_sol(self):
        for _ in range(self.n_iter):
            self.ants = []
            for _ in range(self.n_ants):
                more_items = True
                ant_k = []
                prev_vehicle = self.N - 1
                while(more_items):
                    next_vehicle = self.choose_move(prev_vehicle)
                    aco = aco_bin_packing(stack_lst=stack_lst, vehicle=self.df_vehicles.iloc[next_vehicle])
                    stack_lst, stack_quantity = aco.aco_2D_bin()
                    if not stack_lst:
                        pass

        def updateRule(self, ant, antsArea, i):
            """  
            updateRule
            ----------

            Update rule used to create the delta trail matrix

            Parameters
            - ant: solution of the current ant
            - antsArea: total cost of the single solution
            - i: index of the ant in consideration
            """
            x = 0  #NOTE: qual'è il primo valore di x? il primo camion testato da ogni formica?
            # vehicleArea = self.vehicle['length'] * self.vehicle['width'] 
            trailApp = np.zeros([len(self.pr_move), len(self.pr_move)])
            for id_truck in ant:  # x and y are the position in the state matrix
                y = id_truck
                trailApp[x,y] += 1 #NOTE: forse il +1 qua non va bene, da verificare a programma completo
                x = y
                    
            deltaTrail = trailApp / antsArea[i] # less is the cost, more is the quality of the solution 
            #NOTE: facendo in questo modo vengono numeri davvero piccoli...
            return deltaTrail