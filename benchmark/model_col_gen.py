#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from aco.config import *
from aco.aco_bin_packing import ACO
from aco.utilities import buildStacks, stackInfo_creation
import time

class model_col_gen:
    
    def __init__(self, vehicle, df_items):
        """ 
        model_col_gen
        ------------
        Establish the performance of a vehicle to be filled with a certain amount\n
        of items. The output of this function then will be the column of the column 
        generation

        ### Input parameters
        - vehicle: the vehicle used for the solution
        - df_items: set of items to be put in the vehicle

        ### Class variables
        - vehicle: the vehicle used for the solution
        - df_items: set of items to be put in the vehicle
        - stackInfo: dataframe containing useful info about the stacks
        - items_type: how many distinct types of items are in df_items.\ 
        Are defined by length and width.
        """
        self.vehicle = vehicle
        self.df_items, self.stackInfo = stackInfo_creation(df_items)
        self.items_type = len(self.stackInfo)
        
        # Evaluate the number of items per type in df_items
        self.n_items_type = np.zeros(self.items_type)
        for i in range(self.items_type):
            self.n_items_type[i] =  len(df_items.loc[df_items['stackability_code'] == i])
        
        # Ant Colony Optimizazion initialization
        self.aco = ACO(self.stackInfo, alpha=ALPHA, beta=BETA, 
                  n_ants=N_ANTS, n_iter=N_ITER)
        self.aco.getVehicle(self.vehicle)

        # Stacks building
        stack_lst, stack_quantity = buildStacks(self.vehicle, self.df_items, self.stackInfo)
        self.aco.getStacks(stack_lst, stack_quantity)

    def add_columns(self, n_cols, duals):
        """ 
        add_columns
        ------------
        Outputs the n_cols solutions of the best performing ants as vectors\
            containing the number of items per type.
        
        ### Input parameter
        - n_cols: number of columns in output. Are the n_cols best ants of\n
                the ACO. 
        ### Output parameter
        - columns: each column contains the number of items per type.
        """
        #TODO:check that number of columns <= N_ITER
        columns = np.zeros((self.items_type, n_cols))
        bestAnts = self.aco.aco_2D_bin(n_bestAnts = n_cols, dualVars = duals)
        
        for j, ant in enumerate(bestAnts):
            for stack in ant:
                #TODO: use code to state
                columns[stack.stack_code][j] += len(stack.items)
        
        return columns


    def model(self):
        """  
        model
        ------------
        Contains the model of the Column Generation which is \n
        defined using Guroby. The colums are added using the\n
        pattern given by the ACO solution.
        """

        model = gp.Model("ColumnGeneration")       
        
        columns = np.zeros([self.items_type, N_COLS])
        duals = np.zeros([self.items_type, 1])
        t_start = time.time()
        n_iter = 4
        _iter = 0
        while (time.time() - t_start) <= TIME_LIMIT or _iter <= n_iter :
            x = model.addVars(_iter + N_COLS, 1, vtype=GRB.CONTINUOUS, name="x")

            # Create new columns from the ACO solution 
            columns[:,_iter:_iter+N_COLS] = self.add_columns(n_cols = N_COLS, duals = duals)

            # Adding constraint to the model
            for i in range(self.items_type):
                constr_i = 0
                for n in range(columns.shape[1]):
                    constr_i += columns[i, n] * x[n, 0]

                model.addConstr(
                    constr_i >= self.n_items_type[i],
                    f"Constraint {i}"
                )
        
            # Definition of the objective function
            obj = gp.LinExpr()
            for n in range(columns.shape[1]):
                obj += self.vehicle["cost"] * x[n, 0]   #TODO: replace with gp.quicksum()?

            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            columns = np.append(columns, np.zeros([self.items_type, N_COLS]), axis=1)

            _iter += N_COLS
            # Check if the model is infeasible 
            if model.Status != 3:

                cons = model.getConstrs()
                
                for i in range(self.items_type): 
                        duals[i] = cons[i].getAttr('Pi')
                # Print the values of the dual variables
                if PRINT:
                    print("\nDUAL VARIABLES")
                    for i in range(self.items_type): 
                        print(f"The dual of constr {i} is :{cons[i].getAttr('Pi')}")

                    # Output decision variable values of the MIP and Fixed model (only if > 0)
                    for v in model.getVars():
                        if v.X > 0:
                            print('%s %g' % (v.VarName, v.X))



###############################################################################
#                                  main                                       #
###############################################################################


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from model_col_gen import model_col_gen

if __name__ == "__main__":
    dataset_name = "datasetA"
    type_vehicle = 0
    n_cols = 3
    sol_file_name = f"{dataset_name}_"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )
    
    vehicle = df_vehicles.iloc[type_vehicle].to_dict()
    model_cg = model_col_gen(vehicle, df_items)
    bestAnts = model_cg.model()
