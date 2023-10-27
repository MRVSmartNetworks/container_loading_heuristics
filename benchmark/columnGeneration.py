#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import random
from masterProblem import MasterProblem
from sub.utilities import stackInfo_creation, buildStacks
from sub.aco_bin_packing import ACO
from sub.config import ALPHA, BETA, N_ANTS, N_ITER
from configCG import N_INIT_COLS, N_COLS, TIME_LIMIT

class columnGeneration:
    """  
    columnGeneration
    -----------------------------------------
    Implementation of column generation using as subproblem the ACO
    to solve the 2d bin packing problem. The stacks are created using
    a greedy approach and are the starting point for the ACO meta-heuristic.

    #### INPUT PARAMETERS
    - df_vehicles: pandas dataframe containing different types of trucks
    - df_items: pandas dataframe containing different items

    #### METHODS
    - solve
    - generateColumns
    - genereateInitialColumns
    """
    def __init__(self, df_vehicles, df_items):
        self.df_vehicles = df_vehicles
        self.df_items, self.stackInfo = stackInfo_creation(df_items)
        self.items_type = len(self.stackInfo)
        self.n_vehicles = len(self.df_vehicles)
        
        # Evaluate the number of items per type in df_items
        self.n_items_type = np.zeros(self.items_type)
        for i in range(self.items_type):
            self.n_items_type[i] =  len(df_items.loc[df_items['stackability_code'] == i])
        
        # Ant Colony Optimizazion initialization
        self.aco = ACO(
            self.stackInfo, alpha=ALPHA, beta=BETA, 
            n_ants=N_ANTS, n_iter=N_ITER
        )

        # Creation of the stacks for each vehicle
        self.stack_lst = {}
        self.stack_quantity = {}

        print("BUILDING STACKS FOR EACH VEHICLE\n")
        for i in range(self.n_vehicles):
            id_truck = self.df_vehicles.iloc[i]["id_truck"]
            self.stack_lst[id_truck], self.stack_quantity[id_truck] = buildStacks(self.df_vehicles.iloc[i], self.df_items, self.stackInfo)
            print(f"Stack for vehicle {id_truck} created")

    def solve(self, sol_file_name = None):
        # Generate initial columns for each vehicle
        initColumns = self.generateInitialColumns(n_cols = N_INIT_COLS)

        # Definition of the master problem and build of the model
        master = MasterProblem(initColumns, self.df_vehicles, self.df_items)
        master.buildModel()

        _iter = 1
        duals = 0
        t_start = time.time()
        while  (time.time() - t_start) <= TIME_LIMIT:
            # Solve relaxed master
            feasibile = master.solveRelaxedModel(_iter)
            if feasibile == 0:
                duals = master.getDuals()

            # FIXME: Dummy solution
            vehicle = self.df_vehicles.iloc[random.randint(0, self.n_vehicles)-1].to_dict()
            
            # Solve the Ant Colony optimization
            newColumns = self.generateColumns(n_cols = N_COLS,
                                              duals = duals,
                                              vehicle = vehicle)
            
            # Add new columns to master
            master.addColumns(vehicle["cost"], newColumns)
            _iter += 1
        #TODO: master.solveModel() -> solve the MIP
            

    def generateColumns(self, n_cols, duals, vehicle):
        """  
        generateColumns
        ---------------------------------------------
        Method to generate a given number of columns for a given
        vehicle using the ACO and the dual variables coming from
        the master problem.
        """
        columns = np.zeros((self.items_type, n_cols))
        # Give a vehicle to ACO and the specific stack list
        self.aco.getVehicle(vehicle)
        self.aco.getStacks(self.stack_lst[vehicle["id_truck"]], self.stack_quantity[vehicle["id_truck"]])

        # Solve the 2D bin packing problem
        bestAnts = self.aco.aco_2D_bin(n_bestAnts = n_cols, dualVars = duals)
        for j, ant in enumerate(bestAnts):
            for stack in ant:
                #TODO: use code to state
                columns[stack.stack_code][j] += len(stack.items)
        
        return columns
    
    def generateInitialColumns(self, n_cols):
        """  
        generateInitialColumns
        ---------------------------------------------
        Method to generate the initial set of columns for each type 
        of vehicle in df_vehicles. The columns are generated using
        the ACO.
        """
        print("\nGENERATING THE INITIAL SET OF COLUMNS FOR EACH VEHICLE...\n")
        columns = np.zeros((self.items_type, n_cols * self.n_vehicles))
        _iter = 0
        for i in range(self.n_vehicles):
            vehicle = self.df_vehicles.iloc[i].to_dict()
            columns_vehicle = self.generateColumns(n_cols = N_INIT_COLS,
                                                   duals = 0,
                                                   vehicle = vehicle)
            print(f"Solution found for vehicle {vehicle['id_truck']}")

            columns[:, _iter:_iter+N_INIT_COLS] = columns_vehicle
            _iter += N_INIT_COLS

        return columns