def getStackFeatures(df_items, code):
    """  
    Get the shared features of the stack given the code
    """
    item = df_items[df_items.stackability_code == code].iloc[0]
    return [item.length,
            item.width, 
            item.max_stackability,
            item.forced_orientation] 

def popStack(stack_lst, code):
    """ 
    Pop the first stack with the specified code
    Note: the stack is removed form stack_lst
    """
    for i,stack in enumerate(stack_lst):
        if stack.stack_code == code:
            stack = stack_lst.pop(i)
            return stack, stack_lst
    raise Exception("No more stacks with specified code")