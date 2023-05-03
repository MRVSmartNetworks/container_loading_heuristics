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
            stack.state = code
            if widthwise:
                stack.orient = 'w'
                stack.length, stack.width = stack.width, stack.length
            return stack, stack_lst
    raise Exception("No more stacks with specified code")


def statesCreation(code_orientation):
    """ 
    statesCreation
    --------------
    Create and initialize the probability of a move matrix

    Note: 
          - matrix[0:N_code] are lengthwise
          - matrix[N_code:2*N_code] are widthwise
          - matrix last state is the empty truck
          
    #### INPUT PARAMETERS:
        - code_orientation: dataframe containing all the stackability codes
                            and their forced orientation
    """
    len_matrix = 0
    code_sub = 1
    N_code = len(code_orientation.stackability_code)
    len_matrix = (2*N_code) + 1     # length of the final matrix, the +1 is for adding the state of the empty truck
    mult_array = np.zeros(len_matrix)
    for i,code in enumerate(code_orientation.stackability_code):
        if (code_orientation.iloc[code]["forced_orientation"]) == 'w':    # widthwise constrain
            mult_array[i] = 0
            mult_array[N_code + i] = 1
            code_sub += 1
        else:                                                             # no constrain, so 2 different orientation
            mult_array[i] = 1
            mult_array[N_code + i] = 1
               
    matrix = np.full((len_matrix,len_matrix), 1./(len_matrix-code_sub)) * mult_array
    return matrix  


