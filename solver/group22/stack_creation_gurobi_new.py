import gurobipy as gp
from gurobipy import GRB
from solver.group22.stack import Stack
import os

from solver.group22.stack_creation_heur import checkValidStacks

VERB = False
MORE_VERB = False

N_DEBUG = False


def create_stack_gurobi(df_items, truck, id):
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

    ### Approach

    Having isolated items with the same stackability code,
    create stacks by solving the maximization problem of the
    stack height and the truck height, until all items have
    been used.
    """
    # If the path exists, delete the log file (avoid building up large file)
    if os.path.exists(os.path.join(".", "logs", "stack_logs_g22.txt")):
        os.remove(os.path.join(".", "logs", "stack_logs_g22.txt"))

    stack_codes = df_items.stackability_code.unique()
    stacks_list = []  # Outcome of this function

    # Use a copy of the dataframe in order not to delete the original items
    tmp_items = df_items.copy()

    for code in stack_codes:
        if True:
            print("Code: ", code)
            # if code == 1:
            #     print("Code is 1")
        # TODO - review these:
        tot_high = 0
        tot_weight = 0
        new_stack_needed = False
        other_constraints = {  # Enforce constraints at stack creation rather than after
            "max_height": truck["height"],
            "max_weight": truck["max_weight_stack"],
            "max_dens": truck["max_density"],
        }

        # Isolate the items with the current code
        current_items = tmp_items[tmp_items.stackability_code == code]
        nest_h = current_items.nesting_height.values[0]

        if N_DEBUG:
            assert all(
                current_items.nesting_height.values == nest_h
            ), "Nesting heights are not always the same!"

        # Evaluate most bounding weight constraint
        # Since for each stack. code the base dimensions have to be the same,
        # max density is actually a weight constraint
        max_supported_wt = min(
            float(truck.max_weight_stack),
            float(truck.max_density) * float(current_items.volume.iloc[0]),
        )

        while len(current_items.index) > 0:
            # For each stackability code, launch optimization with OR Tools
            new_stack = solve_knapsack_stack(
                current_items, truck.height, max_supported_wt, other_constraints
            )
            stacks_list.append(new_stack)
            current_items = updateItemsList(new_stack, current_items)

            if VERB:
                print(f"Remaining items, type {code}: {len(current_items.index)}")

    for j in range(len(stacks_list)):
        stacks_list[j].assignID(id)
        id += 1

    # Check validity of stacks
    if checkValidStacks(stacks_list, truck, df_items, compareItems=True):
        return stacks_list, id
    else:
        raise ValueError("Invalid stacks have been created!")


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
    heights = list(items.height)
    weights = list(items.weight)
    # Upper bound for stack height (n. of items)
    min_h = min(heights)
    big_m_items = round(truck_height / min_h)

    data = {}
    data["max_items"] = big_m_items
    data["heights"] = heights
    data["weights"] = weights
    # Will be able to access the items by performing '.iloc[]' on the df
    # with current stack code
    data["items"] = range(len(heights))
    data["bins"] = range(big_m_items)
    data["bin_height"] = truck_height
    data["bin_weight"] = weight
    data["nesting_height"] = list(items.nesting_height)
    data["max_stack"] = list(items.max_stackability)
    return data


def solve_knapsack_stack(items, truck_height, weight, other_constraints):
    """
    solve_cutting_stock
    ---
    Solution of the cutting stock problem associated with stack creation
    by means of Gurobi.

    The method returns a single stack object
    """
    data = map_data(items, truck_height, weight)

    solver = gp.Model("knapsackRevisited")

    ## Variables
    # x[i, j] = 1 if item i is contained at position j (0: top)
    x = solver.addVars(data["items"], data["bins"], vtype=GRB.BINARY, name="x")

    ## Constraints
    # Total height should be less than truck's
    tot_height_constr = solver.addConstr(
        sum(x[i, j] * data["heights"][i] for i in data["items"] for j in data["bins"])
        - sum(
            x[i, j] * data["nesting_height"][i]
            for i in data["items"]
            for j in data["bins"][1:]
        )
        <= data["bin_height"],
        "Respect max height",
    )

    # Each item should appear at most once in its column (not appearing if not used)
    for i in data["items"]:
        solver.addConstr(
            sum(x[i, j] for j in data["bins"]) <= 1, "Item can be in 1 position"
        )

    # For each position there should be one object only
    for j in data["bins"]:
        solver.addConstr(
            sum(x[i, j] for i in data["items"]) <= 1, "Item can be in 1 position"
        )

    # No empty positions
    for j in data["bins"][1:]:
        solver.addConstr(
            sum(x[i, j] for i in data["items"])
            <= sum(x[i, j - 1] for i in data["items"])
        )

    # Max weight of stacks
    solver.addConstr(
        sum(x[i, j] * data["weights"][i] for i in data["items"] for j in data["bins"])
        <= data["bin_weight"],
        "Respect max weight",
    )

    # Max stackability
    # NOTE: gurobi does not allow for strict inequality constraints - had to use <= ...
    for j in data["bins"]:
        for i in data["items"]:
            solver.addConstr(j * x[i, j] <= data["max_stack"][i] - 1)

    ## Objective: minimization of the total number of stacks produced
    solver.setObjective(
        gp.quicksum(
            x[i, j] * data["heights"][i] for i in data["items"] for j in data["bins"]
        )
        - gp.quicksum(
            x[i, j] * data["nesting_height"][i]
            for i in data["items"]
            for j in data["bins"][1:]
        ),
        GRB.MAXIMIZE,
    )

    ## Setting stopping criteria (avoid excessive runtimes)
    solver.Params.LogToConsole = 0
    solver.Params.TimeLimit = 5  # 5 s time limit for each stack
    solver.Params.MIPGap = 0.1  # 10% gap...
    solver.Params.LogFile = os.path.join(".", "logs", "stack_logs_g22.txt")

    solver.optimize()

    if MORE_VERB:
        print(f"  Obj. val: {solver.getObjective().getValue()}")
        print(
            f"  Height constraint value: {solver.getRow(tot_height_constr).getValue()}"
        )

    if N_DEBUG:
        assert solver.getRow(tot_height_constr).getValue() <= data["bin_height"]

    new_stack = Stack()
    for j in data["bins"][::-1]:
        found_in_row = False
        for i in data["items"]:
            if not found_in_row and x[i, j].X == 1:
                # Add element to the stack
                if N_DEBUG:
                    assert (
                        data["heights"][i] == items.iloc[i].height
                    ), "The item heights don't match"

                add_attempt = new_stack.add_item_override(
                    newitem=items.iloc[i], other_constraints=other_constraints
                )
                if add_attempt != 1:
                    print("  Obj. value:", solver.getObjective().getValue())
                    print(
                        "  Height of stack would be: ",
                        new_stack.tot_height
                        - new_stack.next_nesting
                        + items.iloc[i].height,
                    )
                    raise ValueError(
                        f"Cannot add the item in the stack despite it appearing in solution - constraints violated with code {add_attempt}"
                    )

                found_in_row = True
            elif found_in_row and x[i, j].X == 1:
                raise ValueError(f"Another element has been placed in position {j}")

    return new_stack


def updateItemsList(new_stack, df_items):
    """
    updateItemsList
    ---
    Remove the items just placed in the last stack from the initial set of items.
    """
    for it in new_stack.items:
        df_items = df_items[df_items.id_item != it.id_item]

    return df_items
