#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from aco.config import ALPHA, BETA, N_ANTS, N_ITER, N_COLS, N_VARS
from aco.aco_bin_packing import ACO
from aco.utilities import buildStacks, stackInfo_creation, removeRandStacks

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

    def add_columns(self, n_cols):
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
        bestAnts = self.aco.aco_2D_bin(n_bestAnts = n_cols)
        
        for j, ant in enumerate(bestAnts):
            for stack in ant:
                #TODO: use code to state
                columns[stack.stack_code][j] += len(stack.items)
        
        return columns


    def model(self):
        """  
        model
        ------------
        Contains the restricted master problem of the Column Generation\n
        which is defined using Guroby. The colums are added using the\n
        pattern given by the ACO solution.
        """

        m = gp.Model("ColumnGeneration")

        x = m.addVars(N_VARS, 1, vtype=GRB.INTEGER, name="x")
        
        columns = np.zeros([self.items_type, N_VARS])
        #TODO: loop to add columns at every iteration
        for i in range(0, N_VARS, N_COLS):
            if i != 0:
                removeRandStacks(self.aco.stack_lst, self.aco.stack_quantity, 0.1)
            columns[:, i:i+N_COLS] = self.add_columns(n_cols = N_COLS)

        # Adding constraint to the model
        for i in range(self.items_type):
            constr = 0
            for n in range(columns.shape[1]):
                constr += columns[i, n] * x[n, 0]

            m.addConstr(
                constr >= self.n_items_type[i],
                f"Constraint {i}"
            )
        
        # Definition of the objective function
        obj = gp.LinExpr()
        for n in range(N_VARS):
            obj += self.vehicle["cost"] * x[n, 0]   #TODO: replace with gp.quicksum()?
        
        m.setObjective(obj, GRB.MINIMIZE)
        
        m.optimize()

        # Output decision variable values (only if > 0)
        for v in m.getVars():
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
    dataset_name = "dataset_small"
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
