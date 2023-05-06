from solver.group22.stack import Stack
import numpy as np

DEBUG = False

def create_stack_cs(df_items, truck):
    """
    create_stack_cs
    ---
    Given an object dataframe and a truck, create stacks which 
    can be placed into the truck.

    ### Input parameters
    - df_items: pandas Dataframe of usable items.
    - truck: pandas Series object containing the truck information.

    ### Output parameters
    - stacks_list: list containing the created stack objects

    ---
    
    FIXME: *Approach is to be reviewed*

    Having isolated all stackability codes, iterate on all items for each code value.
    Place each item in a stack, until an item does not pass the checks for being added.
    Once this happens, close current stack and add it to the list of stacks, then 
    start a new stack by placing the current item in a new one.

    This method makes use of the Stack class and its method(s) 'add_item' (and 
    'add_item_override').

    Stacks can be created only for items with the same stackability code.
    """
    stack_codes = df_items.stackability_code.unique()
    stacks_list = []        # Outcome of this function
    
    for code in stack_codes:
        if DEBUG:
            print(f"CODE: {code}")
        new_stack_needed = False
        other_constraints = {           # Enforce constraints at stack creation rather than after
            "max_height": truck["height"],
            "max_weight": truck["max_weight_stack"],
            "max_dens": truck["max_density"]
        }

        new_stack = Stack()
        curr_items_code = df_items[df_items.stackability_code == code].reset_index()
        curr_items_code["used_flag"] = 0            # Used to keep track of used items
        all_weights = np.sort(list(curr_items_code.weight.unique()))
        all_weights = all_weights[::-1]
        all_heights = np.sort(list(curr_items_code.height.unique()))
        all_heights = all_heights[::-1]
        
        for i, row in curr_items_code.iterrows():
            # Check the item was not added to a stack already
            if int(row.used_flag) == 0:
                was_added = new_stack.add_item_override(row, other_constraints)
                
                # The value of 'new_stack_needed' can be: 
                # 1: success
                # 0: cannot add item as it won't satisfy max stackability constraint
                # -1: cannot add item as it won't satisfy max height constraint
                # -2: cannot add item as it won't satisfy max weight constraint
                # -3: cannot add item as it won't satisfy max density constraint

                if was_added == -1:
                    # Max height was violated
                    remaining_height = other_constraints['max_height'] - new_stack.tot_height

                    # Find max value of the height which can be placed in the stack
                    # This approach is optimized for dataframes, as it avoids 'explicit' iteration
                    j = 0
                    new_stack_needed = True
                    while j < len(all_heights):
                        if all_heights[j] <= remaining_height:
                            # Try to use an item whose height is the biggest possible, if not available
                            valid_df = curr_items_code[curr_items_code.height == all_heights[j]]
                            
                            k = 0
                            while k < len(valid_df.index):
                                possib_elem = valid_df.iloc[k]
                                if possib_elem.used_flag == 0:
                                    # Try adding it
                                    if new_stack.add_item_override(possib_elem, other_constraints) == 1:
                                        curr_items_code.loc[curr_items_code.id_item == possib_elem.id_item, "used_flag"] = 1
                                        new_stack_needed = False
                                        # If success, break cycle
                                        j = len(all_heights)
                                        k = len(valid_df.index)
                                    # Else: keep on iterating
                                # TODO: try to iterate in a 'smart' way - avoid '+= 1' and try adding more 
                                # if many elements in 'possib_elem'
                                k += 1
                        j += 1

                elif was_added == -2 or was_added == -3:
                    # Max weight was violated
                    if was_added == -2:
                        remaining_weight = other_constraints['max_height'] - new_stack.tot_weight
                    else:
                        # If constraint on the density
                        remaining_weight = other_constraints['max_dens'] * new_stack.area - new_stack.tot_weight

                    j = 0
                    while j < len(all_weights):
                        if all_weights[j] <= remaining_weight:
                            valid_df = curr_items_code[curr_items_code.weight == all_weights[j]]
                            
                            new_stack_needed = True
                            k = 0
                            while k < len(valid_df.index):
                                possib_elem = valid_df.iloc[k]
                                if possib_elem.used_flag == 0:
                                    # Try adding it
                                    if new_stack.add_item_override(possib_elem, other_constraints) == 1:
                                        curr_items_code[curr_items_code.id_item == possib_elem.id_item, "used_flag"] = 1
                                        new_stack_needed = False
                                        # If success, break cycle
                                        j = len(all_weights)
                                        k = len(valid_df.index)
                                    # Else: keep on iterating
                                k += 1
                        j += 1
   
                elif was_added == 0:
                    # Max stackability was violated, stop

                    # if DEBUG:
                    #     print("-> Reached max stackability!")
                    new_stack_needed = True
                elif was_added == 1:
                    # SUCCESS - mark the item as used
                    used_flags = 1

                # if a new stack is needed (unable to add elements):
                if new_stack_needed:
                    stacks_list.append(new_stack)
                    # Open new stack (with current element as first)
                    new_stack = Stack(row, other_constraints)
                    new_stack_needed = False
        
        # Need to add last stack to the list (if not empty)
        if len(new_stack.items) > 0:
            stacks_list.append(new_stack)
    
    for i in range(len(stacks_list)):
        stacks_list[i].assignID(i)

    return stacks_list