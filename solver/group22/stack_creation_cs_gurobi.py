import gurobipy as gp
from gurobipy import GRB
from solver.group22.stack import Stack
import os


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
        
        # Isolate the items with the current code
        current_items = df_items[df_items.stackability_code == code]

        # Evaluate most bounding weight constraint
        # Since for each stack. code the base dimensions have to be the same, 
        # max density is actually a weight constraint
        max_supported_wt = min(float(truck.max_weight_stack), float(truck.max_density) * float(current_items.surface.iloc[0]))

        # For each stackability code, launch optimization with OR Tools
        stacks_list = solve_cutting_stock(current_items, truck.height, max_supported_wt)

    for i in range(len(stacks_list)):
        stacks_list[i].assignID(i)

    return stacks_list

def map_data(items, truck_height, weight):
    """
    map_data
    ---
    Map the data used in the stack creation into the required format.

    ### Input parameters
    - items: pandas dataframe containing the items used for the current 
    optimization (same stackability code)
    - truck_height: height of the current truck - max bin size for the 
    problem.

    ### Output variable
    - data: dictionary containing the info necessary to perform the optimization
    """
    data = {}
    heights = list(items.height)
    weights = list(items.weight)
    data["heights"] = heights
    data["weights"] = weights
    # Will be able to access the items by performing '.iloc[]' on the df 
    # with current stack code
    data["items"] = range(len(heights))
    data["bins"] = data["items"]
    data["bin_height"] = truck_height
    data["bin_weight"] = weight
    return data

def solve_cutting_stock(items, truck_height, weight):
    """
    solve_cutting_stock
    ---
    Solution of the cutting stock problem associated with stack creation 
    by means of OR Tools.

    The method returns the obtained variables values.
    """
    data = map_data(items, truck_height, weight)

    solver = gp.Model("cuttingStock")
    
    ## Variables
    # x[i, j] = 1 if item i is contained in bin j
    x = solver.addVars(data['items'], data['bins'], vtype=GRB.BINARY, name="x")
    
    # y[j] = 1 if stack j is 'used'
    y = solver.addVars(data['bins'], vtype=GRB.BINARY, name="y")
    
    ## Constraints
    # Each item should be in 1 stack only
    for i in data['items']:
        solver.addConstr(sum(x[i, j] for j in data['bins']) == 1, "Use items once")
    
    # Max weight of stacks
    for j in data['bins']:
        solver.addConstr(
            sum(x[i, j] * data['weights'][i] for i in data['items']) <= y[j] * data['bin_weight'],
            "Respect max weight"
        )

    # Max height of stacks
    for j in data['bins']:
        solver.addConstr(
            sum(x[i, j] * data['heights'][i] for i in data['items']) <= y[j] * data['bin_height'],
            "Respect max height"
        )
    
    ## Objective: minimization of the total number of stacks produced
    obj_expr = gp.quicksum([y[j] for j in data['bins']])
    solver.setObjective(obj_expr, GRB.MINIMIZE)

    ## Setting stopping criteria (avoid excessive runtimes)
    solver.Params.TimeLimit = 180       # 3 min time limit for each stack
    solver.Params.MIPGap = 0.1          # 10% gap...
    solver.Params.LogToConsole = 0
    solver.Params.LogFile = os.path.join('.', 'solver', 'group22', 'logs', 'stack_logs.txt')

    print("Here")
    solver.optimize()
    print("Here (2)")
    
    stacks_list = []
    for j in range(len(y)):
        if y[j].X == 1:
            new_stack = Stack()
            stack_items_list = []
            for i in data['items']:
                if x[i, j].X > 0:
                    # Store the positional index of the item to be added to the current stack
                    stack_items_list.append(i)
                    
            # Sort the items of the current stack according to descending max_stackability
            # to avoid as much as possible to violate max_stackability
            curr_stack_items = items.iloc[stack_items_list]
            curr_stack_items = curr_stack_items.sort_values(by=['max_stackability'], ascending=False)

            for k in range(len(stack_items_list)):
                # Add items to the stack
                val = new_stack.add_item_override(curr_stack_items.iloc[k])
                # TODO: check return value - what if cannot add?
                if val == 0:
                    print("-> MAX STACKABILITY TRIGGERED")
                    # Append the current stack, open new one... May not be the best choice
                    stacks_list.append(new_stack)
                    new_stack = Stack()
                    new_stack.add_item_override(curr_stack_items.iloc[k])

            # Add the stack to the list if not empty
            if len(new_stack.items) > 0:
                stacks_list.append(new_stack)
    
    assert len(stacks_list) != 0

    return stacks_list
