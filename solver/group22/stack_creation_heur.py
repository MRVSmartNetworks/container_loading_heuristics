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
    The creation of the stack follows what is called the 'Cutting Stock' problem.
    This function implements an heuristic approach to the solution of this problem.

    Having isolated all stackability codes, iterate on all items for each code value.
    Place each item in a stack, until an item does not pass the checks for being added.
    Once this happens, depending on the reason why the item was not added, choose how
    proceed:

    - If the item violated the max height constraint of the truck, look for the biggest
    valid height in the list containing all the unique height values of the items.
    Having found the viable height, try to add items of this height to the stack, else
    iterate on smaller height values, until either all items are analyzed or a valid
    item was found. If no items were added, the stack needs to be closed and a new one
    is started
    - If the item violated max weight stack, do the same as for max height, but
    considering the items weight
    - If the item violated max density, find the value of the weight associated with
    the max density (given the stack surface) and treat the event as a max weight
    violation
    - If max stackability was violated, it is not possible to add any more items.
    The stack is then closed and a new one is started.

    Stacks can be created only for items with the same stackability code.
    """
    stack_codes = df_items.stackability_code.unique()
    stacks_list = []  # Outcome of this function

    for code in stack_codes:
        if DEBUG:
            print(f"CODE: {code}")
        new_stack_needed = False
        other_constraints = {  # Enforce constraints at stack creation rather than after
            "max_height": truck["height"],
            "max_weight": truck["max_weight_stack"],
            "max_dens": truck["max_density"],
        }

        new_stack = Stack()
        curr_items_code = df_items[df_items.stackability_code == code].reset_index()
        curr_items_code["used_flag"] = 0  # Used to keep track of used items
        all_weights = np.sort(list(curr_items_code.weight.unique()))
        all_weights = all_weights[::-1]
        all_heights = np.sort(list(curr_items_code.height.unique()))
        all_heights = all_heights[::-1]

        # Make sure there is 1 item per ID
        all_items_ids, counts_current_items = np.unique(
            curr_items_code.id_item, return_counts=True
        )
        assert all(counts_current_items == 1), "Duplicate items are present!"

        if code == 4:
            print("Code is 4")
        for i, row in curr_items_code.iterrows():
            # Check the item was not added to a stack already
            if row.id_item == "I0076":
                print(".")

            # If the item has not been used yet
            if int(row.used_flag) == 0:
                if row.id_item == "I0076":
                    print("a")
                was_added = new_stack.add_item_override(row, other_constraints)

                # The value of 'new_stack_needed' can be:
                # 1: success
                # 0: cannot add item as it won't satisfy max stackability constraint
                # -1: cannot add item as it won't satisfy max height constraint
                # -2: cannot add item as it won't satisfy max weight constraint
                # -3: cannot add item as it won't satisfy max density constraint

                if was_added == -1:
                    # Max height was violated
                    remaining_height = (
                        other_constraints["max_height"] - new_stack.tot_height
                    )

                    # Find max value of the height which can be placed in the stack
                    # This approach is optimized for dataframes, as it avoids 'explicit' iteration
                    j = 0
                    new_stack_needed = True
                    while j < len(all_heights):
                        if all_heights[j] <= remaining_height:
                            # Try to use an item whose height is the biggest possible, if not available
                            valid_df = curr_items_code[
                                curr_items_code.height == all_heights[j]
                            ]

                            k = 0
                            while k < len(valid_df.index):
                                possib_elem = valid_df.iloc[k]
                                if possib_elem.used_flag == 0:
                                    # Try adding it
                                    if row.id_item == "I0076":
                                        print("a")
                                    if (
                                        new_stack.add_item_override(
                                            possib_elem, other_constraints
                                        )
                                        == 1
                                    ):
                                        curr_items_code.loc[
                                            curr_items_code.id_item
                                            == possib_elem.id_item,
                                            "used_flag",
                                        ] += 1
                                        new_stack_needed = False
                                        # If success, break cycle
                                        j = len(all_heights)
                                        k = len(valid_df.index)
                                    # Else: keep on iterating
                                # TODO: try to iterate in a 'smart' way - avoid '+= 1' and try adding more
                                # if many elements in 'possib_elem'
                                k += max(1, round(np.log(len(valid_df.index))))
                        j += 1

                elif was_added == -2 or was_added == -3:
                    # Max weight was violated
                    if was_added == -2:
                        remaining_weight = (
                            other_constraints["max_height"] - new_stack.tot_weight
                        )
                    else:
                        # If constraint on the density
                        remaining_weight = (
                            other_constraints["max_dens"] * new_stack.area
                            - new_stack.tot_weight
                        )

                    j = 0
                    new_stack_needed = True
                    while j < len(all_weights):
                        if all_weights[j] <= remaining_weight:
                            valid_df = curr_items_code[
                                curr_items_code.weight == all_weights[j]
                            ]

                            k = 0
                            while k < len(valid_df.index):
                                possib_elem = valid_df.iloc[k]
                                if possib_elem.used_flag == 0:
                                    # Try adding it
                                    if row.id_item == "I0076":
                                        print("a")
                                    if (
                                        new_stack.add_item_override(
                                            possib_elem, other_constraints
                                        )
                                        == 1
                                    ):
                                        curr_items_code.loc[
                                            curr_items_code.id_item
                                            == possib_elem.id_item,
                                            "used_flag",
                                        ] += 1
                                        new_stack_needed = False
                                        # If success, break cycle
                                        j = len(all_weights)
                                        k = len(valid_df.index)
                                    # Else: keep on iterating (on both weight values and items)
                                k += max(1, round(np.log(len(valid_df.index))))
                        j += 1

                elif was_added == 0:
                    # Max stackability was violated, stop

                    # if DEBUG:
                    #     print("-> Reached max stackability!")
                    new_stack_needed = True
                elif was_added == 1:
                    # SUCCESS - mark the item as used
                    curr_items_code.loc[
                        curr_items_code.id_item == row.id_item, "used_flag"
                    ] += 1

                # if a new stack is needed (unable to add elements):
                if new_stack_needed:
                    if "I0076" in [it.id_item for it in new_stack.items]:
                        print("Item I0076 is in new stack")
                    if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
                        stacks_list.append(new_stack)
                    # Open new stack (with current element as first)
                    new_stack = Stack(row, other_constraints)
                    if len(new_stack.items) > 0:
                        curr_items_code.loc[
                            curr_items_code.id_item == row.id_item, "used_flag"
                        ] += 1
                    new_stack_needed = False

        # Need to add last stack to the list (if not empty)
        if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
            stacks_list.append(new_stack)

        assert all(
            curr_items_code.used_flag < 2
        ), "Some items have been used more than once!"

    for j in range(len(stacks_list)):
        stacks_list[j].assignID(j)

    assert all([s.tot_weight for s in stacks_list]) > 0

    # Check validity of stacks
    if checkValidStacks(stacks_list, df_items, compareItems=True):
        return stacks_list


def checkValidStacks(stacks_list, df_items, compareItems=False):
    """
    checkValidStacks
    ---
    Check the created stacks are valid.

    The requirements are that, if the stacks contain objects from 'items',
    each item has exactly been used once.
    """
    # Check that the provided items list does not contain duplicates
    items_ids = np.array(list(df_items.id_item))
    unique_ids, counts = np.unique(items_ids, return_counts=True)
    assert all(counts == 1), "The items list contains duplicates!"

    # Extract the used item ids from the stacks
    stack_items_ids = []  # IDs of all items in the stacks which have been used
    for st in stacks_list:
        stack_items_ids += [it.id_item for it in st.items]

    stack_items_ids = np.array(stack_items_ids)

    used_unique_ids, used_counts = np.unique(stack_items_ids, return_counts=True)

    if compareItems:
        # Compare the number of used items with the number of provided ones
        assert len(used_unique_ids) <= len(
            unique_ids
        ), "The stacks contain more elements than provided"

        # Check that of the provided elements each one has exactly been used once

        for i in range(len(items_ids)):
            if items_ids[i] in used_unique_ids:
                assert (
                    used_counts[used_unique_ids == unique_ids[i]] <= 1
                ), f"[i = {i}] -> Element {unique_ids[i]} has been used {used_counts[used_unique_ids == items_ids[i]]} times in the solution!"

    # Check that each item appearing in the used IDs is present only once
    if not all(used_counts == 1):
        for j in range(len(used_counts)):
            assert (
                used_counts[j] == 1
            ), f"Item {used_unique_ids[j]} was used {used_counts[j]} times"
        return False

    return True
