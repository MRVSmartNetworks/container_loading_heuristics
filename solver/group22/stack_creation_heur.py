from solver.group22.stack import Stack


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
        tot_high = 0
        tot_weight = 0
        new_stack_needed = False
        other_constraints = {           # Enforce constraints at stack creation rather than after
            "max_height": truck["height"],
            "max_weight": truck["max_weight_stack"],
            "max_dens": truck["max_density"]
        }

        new_stack = Stack()
        for i, row in df_items[df_items.stackability_code == code].iterrows():
            # FIXME: the procedure is not ideal! If an item is not added because too heavy it does 
            # not mean that we need to start a new stack...

            was_added = new_stack.add_item_override(row, other_constraints)
            # The value of 'new_stack_needed' can be: 
            # 0: cannot add item as it won't satisfy constraint (weight, height, density, stackability)
            # 1: success
            # NOT HERE - {-1: cannot add item since it would lock the orientation property}
            if was_added == 0:
                new_stack_needed = True
                # In all other cases can still try to add elements to the stack
                # FIXME: it may happen that one element cannot be added because too tall/heavy
                # need to allow for a search for compatible items
                # IDEA: only stop if max_stackability was violated, else act as in the 

            # if a new stack is needed (current element was not added):
            if new_stack_needed:
                stacks_list.append(new_stack)
                # Open new stack (with current element as first)
                new_stack = Stack(row, other_constraints)
                new_stack_needed = False

                # NOTE: this approach also works when we end the loop for the current 
                # stackability code value, as next item will not be added (won't pass 
                # checks in add_item[_override])... 
            else:
                # The item was already added
                pass
        
        # Last stack is probably not appended to the list, since it does 
        # not trigger new_stack_needed
        stacks_list.append(new_stack)
    
    
    for i in range(len(stacks_list)):
        stacks_list[i].assignID(i)

    return stacks_list