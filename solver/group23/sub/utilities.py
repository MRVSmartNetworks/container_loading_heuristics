import numpy as np
from copy import deepcopy

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
    return [item.length,
            item.width, 
            item.max_stackability,
            item.forced_orientation] 

def popStack(stack_lst, code, n_code):
    """ 
    popStack
    --------
    Pop the first stack with the specified code
    Note:   - the stack is removed form stack_lst
            - if the stack need to be widthwise length 
            and width are swapped
    #### INPUT PARAMETERS:
        - stack_lst: full list containing all the stacks created
        - code: stackability code of the selected item
        - n_code: number of different codes
    """ 
    widthwise = False
    if code >= n_code:
        code = code - 7
        widthwise = True

    for i,stack in enumerate(stack_lst):
        if stack.stack_code == code:
            stack = stack_lst.pop(i)
            stack_copy = deepcopy(stack)
            stack_copy.state = code
            if widthwise:
                stack_copy.state += 7
                stack_copy.orient = 'w'
                stack_copy.length, stack_copy.width = stack_copy.width, stack_copy.length
            else:
                stack_copy.orient = 'l'
            
            #NOTE: cerca una soluzione migliore
            return stack_copy, stack_lst
    raise Exception("No more stacks with specified code")


