#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from sub.utilities import *
from sub.stack import Stack
from sub.aco_bin_packing import ACO
from sub.projection import *
import time
#TODO:
# - certi stack code potrebbero avere la stessa dimensione

class Solver23():
    def __init__(self):
        """
        Solver:
        """
        self.name = "solver23"
        self.id_vehicle = 0 # id of the vehicle used for solution format
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
        # Decision for the most convenient vehicle
        vehicle = self.vehicle_decision(df_vehicles)
        # Retrieve information about the features of the items given
        # their stackability code    
        stackInfo = df_items[["length", "width", "stackability_code", "forced_orientation"]].drop_duplicates()
        # Initialization of the ACO object
        aco = ACO(stackInfo)
        aco.getVehicle(vehicle)
        more_items = True
        # Loop until there are no more items left
        while(more_items):
            # Create the stacks given a vehicle and give it to ACO
            aco.getStacks(self.buildStacks(vehicle, df_items, stackInfo))
            # Check if there are stacks left
            if sum(aco.stack_quantity.values()) != 0:
                # Method to solve the 2D bin packing problem
                bestAnt = aco.aco_2D_bin()
                self.solUpdate(bestAnt, vehicle)
                # Remove the items already added to the solution
                df_items = df_items[df_items.id_item.isin(self.sol["id_item"]) == False]
            else:
                more_items = False
        
        df_sol = pd.DataFrame.from_dict(self.sol)
        print("\nN trucks = ", df_sol['idx_vehicle'].nunique())
        print("Tot items: ", len(self.sol["id_item"]))
        print("\nTime:", time.time() - st_time)
        
        df_sol.to_csv(
            os.path.join('results', f'{self.name}_sol.csv'),
            index=False
        )
    
    def buildStacks(self, vehicle, df_items, stackInfo):
        """"
        buildStacks
        -----------
        Function used to create the stack with all the specificity of the selected truck.

        #### INPUT PARAMETERS:
            - vehicle: vehicle type, needed to check the height, weight, max density and max 
                        stack weight for creating the stacks for this specific truck
            - df_items: dataframe containing all the items that will be put into the trucks
        #### OUTPUT PARAMETERS:
            - self.stackList: list of all the stack created 
        """
        #FIXME: controlli non funzionano per stack singoli

        stackability_codes = df_items.stackability_code.unique()
        stack_lst = []
        stack_quantity = {code:0 for code in stackInfo.stackability_code}
        maxStackDensity = (vehicle["length"] * vehicle["width"]) * vehicle["max_density"] #area stack * vehicle max density = maximum stack weight
        for code in stackability_codes:
            stack_quantity[code] = 0
            stack_feat = getStackFeatures(df_items, code)
            
            stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
            
            new_stack_needed = False
            iter_items = df_items[df_items.stackability_code == code].head(200)
            for i, row in iter_items.iterrows():
                height = stack.height + row.height 
                weight = stack.weight + (row.weight)
                if height > vehicle['height']:
                    new_stack_needed = True
                if weight > vehicle['max_weight_stack'] or stack.weight > maxStackDensity:
                    new_stack_needed = True
                if stack.n_items == row.max_stackability:
                    new_stack_needed = True
                # if a new stack is needed:
                if new_stack_needed:
                    if stack.items != []:
                        stack_lst.append(stack)
                        stack_quantity[code] += 1 # number of the stack with this precise stackability code
                    stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
                    stack.addItem(row.id_item, row.height - row.nesting_height)
                    stack.updateHeight(row.height - row.nesting_height)
                    stack.updateWeight(row.weight)
                    new_stack_needed = False
                else:
                    # else add the item
                    stack.addItem(row.id_item, row.height - row.nesting_height)
                    stack.updateHeight(row.height - row.nesting_height)
                    stack.updateWeight(row.weight)
                    if i == iter_items.tail(1).index:
                        stack_lst.append(stack)
                        stack_quantity[code] += 1
        return stack_lst, stack_quantity
    
    def solUpdate(self, bestAnt, vehicle):
        """  
        solUpdate
        -----
        ### Input parameters:
            - bestAnt: the best ant output of the ACO
            solution of the 2D bin packing problem
            - vehicle: the vehicle of the solution
        """
        lenSol = len(self.sol['id_stack'])  # var to keep id stack unique
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
        self.id_vehicle += 1 # update of the vehicle id
    
    def vehicle_decision(self, df_vehicles):
        """ 
        vehicle_decision
        -------
        ### Input parameters:
            - df_vehicles: dataframe containing all the different 
            types of trucks that can be choose
        ### Output parametes:
            - the row corresponding to the best vehicle in terms of
            volume, weight over cost ratio
        """
        eff_ratio = []
        for _, vehicle in df_vehicles.iterrows():
            volume = vehicle['length'] * vehicle['width'] * vehicle['height']
            eff_ratio.append((volume * vehicle['max_weight'])/vehicle['cost'])
        max_index = eff_ratio.index(max(eff_ratio))
        return df_vehicles.iloc[max_index]

    

#######################################################################
############ main         
if __name__ == "__main__":
    
    os.path.join('.', 'results', "sol23.csv")
    df_items = pd.read_csv(
        os.path.join(
            '.', 'data', 'datasetA',
            'items.csv'
        ),
    )
    df_vehicles = pd.read_csv(
        os.path.join(
            '.', 'data', 'datasetA',
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