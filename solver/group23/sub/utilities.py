#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

def getStackFeatures(df_items, code):
    """
    getStackFeatures
    ----------------
    Get the shared features of the stack given the code

    #### INPUT PARAMETERS:
        - df_items: initial dataframe of all the items
        - code: stackability code of the selected item
    """
    item = df_items[df_items.stackability_code == code].iloc[0]
    code_features = {
        "length": item.length,
        "width": item.width,
        "max_stackability": item.max_stackability,
        "orientation": item.forced_orientation
    }
    return code_features

def update_stack_lst(bestAnt, stack_lst, stack_quantity):
    """ 
    update_stack_lst
    ----------------
    Function used to update the stack list by removing the stacks
    that has been already added to the previous vehicle

    #### INPUT PARAMETERS:
        - bestAnt: contains the stacks used for the solution
                   of the previous truck
        - stack_lst: the stack list to be updated
        - stack_quantity: update the stack quantity dictionary
                          together with the stack list
    """
    id_item_lst = []
    i = 0
    for stack_ant in bestAnt:
        for i, stack in enumerate(stack_lst):
            if stack.items == stack_ant.items:
                stack_lst.pop(i)
                stack_quantity[stack_ant.stack_code] -= 1

def df_vehicles_improv(df_vehicles):
    """ 
    df_vehicles_improv
    ----------------
    Function to add new parameters for each type of vehicle, such as volume,
    area, efficiency ratio and weight efficiency, especially used in the 
    vehicle decision algorithm

    #### INPUT PARAMETERS:
        - df_vehicles: dataframe with all type of vehicles
    
    #### OUTPUT PARAMETERS
        - df_vehicles
    """
    # Evaluation of important parameters and added to the vehicles dataframe
    df_vehicles["volume"] = (df_vehicles["length"] * df_vehicles["width"] * df_vehicles["height"])
    df_vehicles["area"] = (df_vehicles['length'] * df_vehicles['width'])
    df_vehicles["eff_ratio"] = (df_vehicles['volume'] * df_vehicles['max_weight'] / df_vehicles['cost'])
    df_vehicles["tot_eff_ratio"] = (df_vehicles['eff_ratio'] * df_vehicles['max_weight_stack'])
    df_vehicles["weight_eff"] = (df_vehicles['max_weight'] * df_vehicles['max_weight_stack'] / df_vehicles['cost'])

    return df_vehicles

def stackInfo_creation(df_items):
    """  
    stackInfo_creation
    ------------------
    Creation of a dataframe containing the carachteristics of items per
    stackability code
    """
    # Retrieve information about the features of the items given their stackability code,
    # then a sorting is needed for the correct performance of states creation
    stackInfo_App = df_items[["length", "width", "stackability_code", "forced_orientation"]].drop_duplicates()
    stackInfo_App = stackInfo_App.sort_values(by=['stackability_code']).reset_index(drop=True) 
    stackInfo = stackInfo_App[["length", "width", "stackability_code"]].drop_duplicates().reset_index(drop=True) 
    stackInfo["forced_orientation"] = ""
    # In case of stackability codes with different orientation, they are divided into
    # different stackability codes for mantaining the standard ACO solution
    for i in range(len(stackInfo_App)):
        stackInfo.at[i,"forced_orientation"] = stackInfo_App.at[i,"forced_orientation"]
        stackInfo.at[i,"length"] = stackInfo_App.at[i,"length"]
        stackInfo.at[i,"width"] = stackInfo_App.at[i,"width"]
        stackInfo.at[i,"stackability_code"] = i
    # If items with same stack code have different characteristics raise an exception
    if(len(stackInfo) != len(stackInfo[["stackability_code", "forced_orientation"]].drop_duplicates())):
        raise ValueError("Items with the same stackability code have different lenght or width!")
    
    # Only when some modification happen to the stackability code the mapping must be done
    if not stackInfo.equals(stackInfo_App):
        df_items = map_items(df_items, stackInfo_App)
    
    return df_items, stackInfo

def map_items(df_items, stackInfo_App):
    """ 
    map_items
    ---------
    Mapping function used to correct the stackability code 
    
    #### INPUT PARAMETERS:
    - df_items: dataframe containing all the items 
                that are to be put into the trucks
    - stackInfo_App: dataframe containing the original stackability codes that need to be mapped to their index also in the item dataframe

    #### OUTPUT PARAMETERS:
    - df_items: the new items dataframe with all the stackability codes corrected
    """

    # For every item the code must be corrected
    for i,code in enumerate(df_items.stackability_code):
        orientation = df_items.at[i,"forced_orientation"]

        # The new code is the corresponding changed from stackInfo_App to stackInfo
        new_code = np.where((stackInfo_App.stackability_code == code) & (stackInfo_App.forced_orientation == orientation))[0]
        df_items.at[i,"stackability_code"] = new_code[0]

    return df_items


