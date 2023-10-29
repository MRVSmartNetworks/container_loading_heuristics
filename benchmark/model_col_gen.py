#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB
<<<<<<< HEAD
from sub.config import ALPHA, BETA, N_ANTS, N_ITER, N_COLS, N_VARS
from sub.aco_bin_packing import ACO
from sub.utilities import buildStacks, stackInfo_creation, removeRandStacks
=======
from aco.config import *
from aco.aco_bin_packing import ACO
from aco.utilities import buildStacks, stackInfo_creation
import time
>>>>>>> main_CG

np.random.seed(0)

# RUN python benchmark/model_col_gen.py
class model_col_gen:
    
    def __init__(self, df_vehicles, df_items):
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

        # Stacks building for each vehicle
        self.stack_lst = {}
        self.stack_quantity = {}

        print("BUILDING STACKS FOR EACH VEHICLE")
        for i in range(self.n_vehicles):
            id_truck = self.df_vehicles.iloc[i]["id_truck"]
            self.stack_lst[id_truck], self.stack_quantity[id_truck] = buildStacks(self.df_vehicles.iloc[i], self.df_items, self.stackInfo)
            print(f"\n Stack for vehicle {id_truck} created")

            
        

    def add_columns(self, n_cols, duals, vehicle):
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


    def model(self):
        """  
        model
        ------------
        Contains the model of the Column Generation which is \n
        defined using Guroby. The colums are added using the\n
        pattern given by the ACO solution.
        """

        model = gp.Model("MasterProblemCG")       
        
        columns = np.zeros([self.items_type, N_COLS])
        # Dict to map the column index to the vehicle ID
        cols_to_vehicle = {}

        # Initialize duals associated to a specific vehicle        
        duals_dict = {}
        for _ in range(self.n_vehicles):
            id_truck = self.df_vehicles.iloc[_]["id_truck"]
            duals_dict[id_truck] = np.zeros([self.items_type, 1])

        t_start = time.time()
        ind_truck = 0
        n_iter = 100 # TODO: set in config
        _iter = 0
        # initialize obj function
        obj = 0
        # initialize constraints
        constrs = []
        vars = []
        while (time.time() - t_start) <= TIME_LIMIT and _iter <= n_iter * N_COLS:
            x = model.addVars(
                N_COLS,
                vtype=GRB.CONTINUOUS,
                lb = 0,
                name=f"X_{_iter}"
            )
    
            for n in range(N_COLS):
                vars.append(x[n])

            # Create new columns from the ACO solution
            vehicle = self.df_vehicles.iloc[ind_truck].to_dict()
            duals = duals_dict[vehicle["id_truck"]]
            columns[:,_iter:_iter+N_COLS] = self.add_columns(
                n_cols = N_COLS,
                duals = duals,
                vehicle = vehicle
            )
            
            # Map columns index with vehicle ID
            for i in range(_iter, columns.shape[1]):
                cols_to_vehicle[i] = vehicle["id_truck"] 
            if _iter == 0:
                # Adding constraint to the model
                for i in range(self.items_type):
                    constr_i = 0
                    for n in range(columns.shape[1]):
                        constr_i += columns[i, n] * x[n]
                    constrs.append(
                        model.addConstr(
                            constr_i >= self.n_items_type[i],
                            f"quantity_item{i}"
                        )
                    )
            else:
                # update the constraints
                for i in range(self.items_type):
                    constr_i = 0
                    ii = 0
                    for n in range(columns.shape[1]-N_COLS, columns.shape[1]):
                        # constr_i += columns[i, n] * x[ii]
                        model.chgCoeff(constrs[i], x[ii], columns[i, n])
                        ii += 1
                    # get row
                    # tmp = model.getRow(constrs[0])
                
            for n in range(N_COLS):
                obj += vehicle["cost"] * x[n]

            model.setObjective(obj, GRB.MINIMIZE)
            model.write(f"./benchmark/results/model_{_iter}.lp")
            model.update()
            model.optimize()

            columns = np.append(columns, np.zeros([self.items_type, N_COLS]), axis=1)

            _iter += N_COLS
            ind_truck += 1
            if ind_truck >= self.n_vehicles:
                ind_truck = 0

            # Check if the model is infeasible 
            if model.Status != 3:
                print([var.X for var in vars])
                cons = model.getConstrs()
                
                for i in range(self.items_type): 
                    duals_dict[vehicle["id_truck"]][i] = cons[i].getAttr('Pi')

                # Print the values of the dual variables
                if PRINT_CG:
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
    dataset_name = "dataset_small"
    sol_file_name = f"{dataset_name}_"
    df_items = pd.read_csv(
        os.path.join(".", "data", dataset_name, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", dataset_name, "vehicles.csv"),
    )
    
    model_cg = model_col_gen(df_vehicles, df_items)
    bestAnts = model_cg.model()
