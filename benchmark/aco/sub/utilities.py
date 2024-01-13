#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd

try:
    from .config import N_WEIGHT_CLUSTERS, ONE_LAYER
    from .stack import Stack
except ImportError:
    from sub.config import N_WEIGHT_CLUSTERS, ONE_LAYER
    from sub.stack import Stack


def buildSingleStack(
    df_items,
    stackInfo,
    vehicle,
    n_items,
    stack_code,
    orient,
    avg_stack_W,
    tot_weight,
):
    item_lost = False

    stack_feat = (stackInfo[stackInfo.stackability_code == stack_code].values)[
        0
    ]
    stack = Stack(int(stack_code), stack_feat[0], stack_feat[1], stack_feat[3])

    # Vehicle constrain in a dictionary, ready to be passed to the addItem function
    constraints = {
        "max_height": vehicle["height"],
        "max_weight_stack": vehicle["max_weight_stack"],
        "max_density": vehicle["max_density"],
    }
    items_code = df_items[
        (df_items.stackability_code == stack_code)
        & (
            (df_items.forced_orientation == "n")
            | (df_items.forced_orientation == orient)
        )
    ]
    # Obtaining all the unique value of height and weight for a better stack creation
    unique_height = np.sort(items_code.height.unique())[::-1]
    unique_weight = np.sort(items_code.weight.unique())[::-1]

    new_stack_needed = False
    minFlag = False
    k = 0

    avg_height = vehicle.height / n_items
    if len(unique_height) > 0:
        if (
            min(unique_height) * (n_items - 1) + max(unique_height)
            > vehicle.height
        ):
            minFlag = True

    while (
        stack.n_items < n_items
        and new_stack_needed == False
        and len(items_code) != 0
    ):
        # Optimization on weight
        if tot_weight <= avg_stack_W:
            items_code = items_code.sort_values(by="weight", ascending=False)
        else:
            items_code = items_code.sort_values(by="weight", ascending=True)

        if not ONE_LAYER:
            # Optimization on height
            if stack.height <= avg_height * k and not minFlag:
                items_code = items_code.sort_values(
                    by=["height"], ascending=False
                )
            else:
                items_code = items_code.sort_values(
                    by=["height"], ascending=True
                )
                if stack.n_items + 2 == n_items:
                    minFlag = False

        item = items_code.iloc[0]
        if tot_weight + stack.weight + item.weight <= vehicle.max_weight:
            stack_added = stack.addItem(item, constraints)

            # Returned code 0 means that the max stackability code is reached
            if stack_added == 0:
                new_stack_needed = True

            # Returned code -1 means max_height reached
            if stack_added == -1:
                new_stack_needed = True

                # If other item with different height exist then another iteam is searched to be fitted in this stack
                if len(unique_height) > 1:
                    fit = False
                    h = 0
                    while h < len(unique_height) and not fit:
                        # If an item respect the height constrain is found, all the other constrain are also checked
                        if (
                            unique_height[h] + stack.height
                            <= constraints["max_height"]
                        ):
                            valid_items = items_code[
                                items_code.height == unique_height[h]
                            ]
                            i = 0
                            while i < len(valid_items) and not fit:
                                # If all the constrain are respected the item is added to the stack
                                if (
                                    stack.addItem(
                                        valid_items.iloc[i], constraints
                                    )
                                    == 1
                                ):
                                    fit = True
                                    df_items = df_items[
                                        df_items["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    items_code = items_code[
                                        items_code["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    tot_weight += valid_items.iloc[i].weight
                                i += 1
                        h += 1

            # Returned code -2 means max_weight reached(the following procedure is the same as the height)
            if stack_added == -2:
                new_stack_needed = True
                if len(unique_weight) > 1:
                    fit = False
                    w = 0
                    while w < len(unique_weight) and not fit:
                        if (
                            unique_weight[w] + stack.weight
                            <= constraints["max_weight_stack"]
                        ):
                            valid_items = items_code[
                                items_code.weight == unique_weight[w]
                            ]
                            i = 0
                            while i < len(valid_items) and not fit:
                                if (
                                    stack.addItem(
                                        valid_items.iloc[i], constraints
                                    )
                                    == 1
                                ):
                                    fit = True
                                    df_items = df_items[
                                        df_items["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    items_code = items_code[
                                        items_code["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    tot_weight += valid_items.iloc[i].weight
                                i += 1
                        w += 1

            # Returned code -3 means max_density reached(another item is searched)
            if stack_added == -3:
                new_stack_needed = True
                if len(unique_weight) > 1:
                    fit = False
                    w = 0
                    while w < len(unique_weight) and not fit:
                        density = (unique_weight[w] + stack.weight) / stack.area
                        if density <= constraints["max_density"]:
                            valid_items = items_code[
                                items_code.weight == unique_weight[w]
                            ]
                            i = 0
                            while i < len(valid_items) and not fit:
                                if (
                                    stack.addItem(
                                        valid_items.iloc[i], constraints
                                    )
                                    == 1
                                ):
                                    fit = True
                                    df_items = df_items[
                                        df_items["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    items_code = items_code[
                                        items_code["id_item"]
                                        != valid_items.iloc[i].id_item
                                    ]
                                    tot_weight += valid_items.iloc[i].weight
                                i += 1
                        w += 1

            # Returned code 1 means item correctly added to the stack
            if stack_added == 1:
                df_items = df_items[df_items["id_item"] != item.id_item]
                items_code = items_code[items_code["id_item"] != item.id_item]
                tot_weight += item.weight
        else:
            new_stack_needed = True
            # items_code = items_code[items_code["id_item"] != item.id_item]

        k += 1
    stack.updateHeight()
    if stack.n_items != n_items and not items_code.empty:
        item_lost = True
        pass
    return stack, df_items, tot_weight, item_lost


def buildStacks(vehicle, df_items, stackInfo):
    """ "
    buildStacks
    -----------
    Function used to create the stack with all the specificity of the selected truck.

    #### INPUT PARAMETERS:
        - vehicle: vehicle type, needed to check the height, weight, max density and max\n
                        stack weight for creating the stacks for this specific truck
        - df_items: dataframe containing all the items that will be put into the trucks
        - stackInfo: dataframe containing all the main information fro each stackability code
    #### OUTPUT PARAMETERS:
        - stack_lst: list of all the stack created
        - stack_quantity: dictionary containing all the numbers of the items divided in stackability codes
    """

    removed = []
    stack_lst = []
    stack_quantity = {code: 0 for code in stackInfo.stackability_code}

    # Vehicle constrain in a dictionary, ready to be passed to the addItem function
    constraints = {
        "max_height": vehicle["height"],
        "max_weight_stack": vehicle["max_weight_stack"],
        "max_density": vehicle["max_density"],
    }
    n_items_insert = 0
    # Loop over all the stackability code
    for code in stackInfo.stackability_code:
        stack_quantity[code] = 0
        stack_feat = (stackInfo[stackInfo.stackability_code == code].values)[0]
        stack = Stack(int(code), stack_feat[0], stack_feat[1], stack_feat[3])

        # Taking only the first n items of the specific stack code to speed up the computation
        items_code = df_items[df_items.stackability_code == code]  # .head(1000)

        # Obtaining all the unique value of height and weight for a better stack creation
        unique_height = np.sort(items_code.height.unique())[::-1]
        unique_weight = np.sort(items_code.weight.unique())[::-1]

        # Loop over the items for the stack creation
        for i, row in items_code.iterrows():
            new_stack_needed = False

            # Check that the item is not already been used
            if not row.id_item in removed:
                stack_added = stack.addItem(row, constraints)

                # Returned code 0 means that the max stackability code is reached
                if stack_added == 0:
                    new_stack_needed = True

                # Returned code -1 means max_height reached
                if stack_added == -1:
                    new_stack_needed = True

                    # If other item with different height exist then another iteam is searched to be fitted in this stack
                    if len(unique_height) > 1:
                        fit = False
                        h = 0
                        while h < len(unique_height) and not fit:
                            # If an item respect the height constrain is found, all the other constrain are also checked
                            if (
                                unique_height[h] + stack.height
                                <= constraints["max_height"]
                            ):
                                valid_items = items_code[
                                    items_code.height == unique_height[h]
                                ]
                                i = 0
                                while i < len(valid_items) and not fit:
                                    # If all the constrain are respected the item is added to the stack
                                    if (
                                        stack.addItem(
                                            valid_items.iloc[i], constraints
                                        )
                                        == 1
                                    ):
                                        fit = True
                                        id_item = valid_items.iloc[i].id_item
                                        items_code = items_code[
                                            items_code.id_item
                                            != valid_items.iloc[i].id_item
                                        ]
                                        removed.append(id_item)
                                    i += 1
                            h += 1

                # Returned code -2 means max_weight reached(the following procedure is the same as the height)
                if stack_added == -2:
                    new_stack_needed = True
                    if len(unique_weight) > 1:
                        fit = False
                        w = 0
                        while w < len(unique_weight) and not fit:
                            if (
                                unique_weight[w] + stack.weight
                                <= constraints["max_weight_stack"]
                            ):
                                valid_items = items_code[
                                    items_code.weight == unique_weight[w]
                                ]
                                i = 0
                                while i < len(valid_items) and not fit:
                                    if (
                                        stack.addItem(
                                            valid_items.iloc[i], constraints
                                        )
                                        == 1
                                    ):
                                        fit = True
                                        id_item = valid_items.iloc[i].id_item
                                        items_code = items_code[
                                            items_code.id_item
                                            != valid_items.iloc[i].id_item
                                        ]
                                        removed.append(id_item)
                                    i += 1
                            w += 1

                # Returned code -3 means max_density reached(another item is searched)
                if stack_added == -3:
                    new_stack_needed = True
                    if len(unique_weight) > 1:
                        fit = False
                        w = 0
                        while w < len(unique_weight) and not fit:
                            density = (
                                unique_weight[w] + stack.weight
                            ) / stack.area
                            if density <= constraints["max_density"]:
                                valid_items = items_code[
                                    items_code.weight == unique_weight[w]
                                ]
                                i = 0
                                while i < len(valid_items) and not fit:
                                    if (
                                        stack.addItem(
                                            valid_items.iloc[i], constraints
                                        )
                                        == 1
                                    ):
                                        fit = True
                                        id_item = valid_items.iloc[i].id_item
                                        items_code = items_code[
                                            items_code.id_item
                                            != valid_items.iloc[i].id_item
                                        ]
                                        removed.append(id_item)
                                    i += 1
                            w += 1

                # Returned code 1 means item correctly added to the stack
                if stack_added == 1:
                    items_code = items_code[items_code.id_item != row.id_item]

                # When the stack il ready must be added to the stackList
                if new_stack_needed and stack.n_items != 0:
                    n_items_insert += stack.n_items
                    stack.updateHeight()
                    stack_lst.append(stack)
                    stack_quantity[code] += 1

                    stack = Stack(
                        int(code), stack_feat[0], stack_feat[1], stack_feat[3]
                    )
                    stack.addItem(row, constraints)
                    items_code = items_code[items_code.id_item != row.id_item]

        # After the loop if the last stack created have some items must be added
        if stack.n_items > 0:
            n_items_insert += stack.n_items
            stack.updateHeight()
            stack_lst.append(stack)
            stack_quantity[code] += 1

    assert len(df_items) == n_items_insert

    return stack_lst, stack_quantity


def sol_to_df(dataset_name, bestAnt, vehicle):
    """
    sol_to_df
    -----
    ### Input parameters:
        - bestAnt: the best ant output of the ACO solution of the 2D bin packing problem
        - vehicle: the vehicle of the solution
    """
    sol = {
        "dataset": [],
        "type_vehicle": [],
        "id_stack": [],
        "id_item": [],
        "x_origin": [],
        "y_origin": [],
        "z_origin": [],
        "orient": [],
    }

    for i, stack in enumerate(bestAnt):
        z_origin = 0
        # Saving all the item with their information in the dictionary solution
        for j in range(len(stack.items)):
            sol["dataset"].append(dataset_name)
            sol["type_vehicle"].append(vehicle["id_truck"])
            sol["id_stack"].append(f"S{i}")
            sol["id_item"].append(stack.items[j])
            sol["x_origin"].append(stack.vertexes[0][0])
            sol["y_origin"].append(stack.vertexes[0][1])
            sol["z_origin"].append(z_origin)
            sol["orient"].append(stack.orient)
            z_origin += stack.h_items[j]

    return pd.DataFrame.from_dict(sol)


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
        "orientation": item.forced_orientation,
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


def removeRandStacks(stack_lst, stack_quantity, perc=0.10):
    elem_to_rem = int(len(stack_lst) * perc)
    for i in range(elem_to_rem):
        rand_ind = random.randrange(len(stack_lst))
        stack_code = stack_lst[rand_ind].stack_code
        stack_lst.pop(rand_ind)
        stack_quantity[stack_code] -= 1


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
    df_vehicles["volume"] = (
        df_vehicles["length"] * df_vehicles["width"] * df_vehicles["height"]
    )
    df_vehicles["area"] = df_vehicles["length"] * df_vehicles["width"]
    df_vehicles["eff_ratio"] = (
        df_vehicles["volume"] * df_vehicles["max_weight"] / df_vehicles["cost"]
    )
    df_vehicles["tot_eff_ratio"] = (
        df_vehicles["eff_ratio"] * df_vehicles["max_weight_stack"]
    )
    df_vehicles["weight_eff"] = (
        df_vehicles["max_weight"]
        * df_vehicles["max_weight_stack"]
        / df_vehicles["cost"]
    )

    return df_vehicles


def stackInfo_creation_weight(df_items):
    """
    stackInfo_creation
    ------------------
    Creation of a dataframe containing the carachteristics of items per
    stackability code.

    This function adds a column with the stack weight (`classWeight`) that can
    be used to group the stacks based on their weight.

    Args:
        df_items:
    """
    # Retrieve information about the features of the items given their stackability code,
    # then a sorting is needed for the correct performance of states creation
    df_items = map_items_weight(df_items)
    stackInfo_App = df_items[
        [
            "length",
            "width",
            "stackability_code",
            "forced_orientation",
            "classWeight",
        ]
    ].drop_duplicates()
    stackInfo_App = stackInfo_App.sort_values(
        by=["stackability_code"]
    ).reset_index(drop=True)
    stackInfo = (
        stackInfo_App[["length", "width", "stackability_code", "classWeight"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    stackInfo["forced_orientation"] = ""
    # In case of stackability codes with different orientation, they are divided into
    # different stackability codes for mantaining the standard ACO solution
    for i in range(len(stackInfo_App)):
        stackInfo.at[i, "forced_orientation"] = stackInfo_App.at[
            i, "forced_orientation"
        ]
        stackInfo.at[i, "length"] = stackInfo_App.at[i, "length"]
        stackInfo.at[i, "width"] = stackInfo_App.at[i, "width"]
        stackInfo.at[i, "stackability_code"] = i
        stackInfo.at[i, "classWeight"] = stackInfo_App.at[i, "classWeight"]
    # If items with same stack code have different characteristics raise an exception
    if len(stackInfo) != len(
        stackInfo[["stackability_code", "forced_orientation"]].drop_duplicates()
    ):
        raise ValueError(
            "Items with the same stackability code have different lenght or width!"
        )

    # Only when some modification happen to the stackability code the mapping must be done
    if not stackInfo.equals(stackInfo_App):
        df_items = df_items.reset_index(drop=True)
        df_items = map_items(df_items, stackInfo_App)

    return df_items, stackInfo


def map_items_weight(df_items):
    """
    Group the items into different weight classes for each stackability code.

    Args:
        df_items: dataframe of items

    Returns:
        The same dataframe, with an additional column `classWeight` indicating
        the weight cluster it belongs to.
    """
    # FIXME: this function takes a long time to run! The issue is the use of
    # `searchsorted`
    codes = df_items.stackability_code.drop_duplicates()
    data = []
    for code in codes:
        items_code = df_items[
            df_items.stackability_code == code
        ]  # Get items with current code
        maxIt_W = max(items_code["weight"])  # Get maximum weight of these items
        cols = np.linspace(
            0, maxIt_W, num=N_WEIGHT_CLUSTERS
        )  # Define the weight clusters (by value)
        classWeight = np.searchsorted(
            cols, items_code.loc[:, "weight"]
        )  # Separate the items in the corresponding weight classes
        items_code.loc[:, ["classWeight"]] = classWeight

        for _, row in items_code.iterrows():
            data.append(row)
    df = pd.DataFrame(data)
    return df


def loadingBar(
    current_iter: int,
    tot_iter: int,
    n_chars: int = 10,
    ch: str = "=",
    n_ch: str = " ",
) -> str:
    """
    loadingBar
    ---
    Produce a loading bar string to be printed.

    ### Input parameters
    - current_iter: current iteration, will determine the position
    of the current bar
    - tot_iter: total number of iterations to be performed
    - n_chars: total length of the loading bar in characters
    - ch: character that makes up the loading bar (default: =)
    - n_ch: character that makes up the remaining part of the bar
    (default: blankspace)
    """
    n_elem = int(current_iter * n_chars / tot_iter)
    prog = str("".join([ch] * n_elem))
    n_prog = str("".join([" "] * (n_chars - n_elem - 1)))
    return "[" + prog + n_prog + "]"


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
    for i, code in enumerate(df_items.stackability_code):
        orientation = df_items.at[i, "forced_orientation"]
        classWeight = df_items.at[i, "classWeight"]

        # The new code is the corresponding changed from stackInfo_App to stackInfo
        new_code = np.where(
            (stackInfo_App.stackability_code == code)
            & (stackInfo_App.forced_orientation == orientation)
            & (stackInfo_App.classWeight == classWeight)
        )[0]
        df_items.at[i, "stackability_code"] = new_code[0]

    return df_items


def map_items_old(df_items, stackInfo_App):
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
    for i, code in enumerate(df_items.stackability_code):
        orientation = df_items.at[i, "forced_orientation"]

        # The new code is the corresponding changed from stackInfo_App to stackInfo
        new_code = np.where(
            (stackInfo_App.stackability_code == code)
            & (stackInfo_App.forced_orientation == orientation)
        )[0]
        df_items.at[i, "stackability_code"] = new_code[0]

    return df_items


def stackInfo_creation(df_items):
    """
    stackInfo_creation
    ------------------
    Creation of a dataframe containing the carachteristics of items per
    stackability code
    """
    # Retrieve information about the features of the items given their stackability code,
    # then a sorting is needed for the correct performance of states creation
    stackInfo_App = df_items[
        ["length", "width", "stackability_code", "forced_orientation"]
    ].drop_duplicates()
    stackInfo_App = stackInfo_App.sort_values(
        by=["stackability_code"]
    ).reset_index(drop=True)
    stackInfo = (
        stackInfo_App[["length", "width", "stackability_code"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    stackInfo["forced_orientation"] = ""
    # In case of stackability codes with different orientation, they are divided into
    # different stackability codes for mantaining the standard ACO solution
    for i in range(len(stackInfo_App)):
        stackInfo.at[i, "forced_orientation"] = stackInfo_App.at[
            i, "forced_orientation"
        ]
        stackInfo.at[i, "length"] = stackInfo_App.at[i, "length"]
        stackInfo.at[i, "width"] = stackInfo_App.at[i, "width"]
        stackInfo.at[i, "stackability_code"] = i
    # If items with same stack code have different characteristics raise an exception
    if len(stackInfo) != len(
        stackInfo[["stackability_code", "forced_orientation"]].drop_duplicates()
    ):
        raise ValueError(
            "Items with the same stackability code have different lenght or width!"
        )

    # Only when some modification happen to the stackability code the mapping must be done
    if not stackInfo.equals(stackInfo_App):
        df_items = map_items_old(df_items, stackInfo_App)

    return df_items, stackInfo
