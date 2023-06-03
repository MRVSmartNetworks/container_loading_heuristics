from solver.group22.stack import Stack
import numpy as np

DEBUG = False


def create_stack_heur(df_items, truck, id):
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
    n_items_init = len(df_items.index)

    stack_codes = df_items.stackability_code.unique()
    stacks_list = []  # Outcome of this function

    if len(df_items.index) == 0:
        return [], id

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
        all_weights = np.sort(list(curr_items_code.weight.unique()))
        all_weights = all_weights[::-1]  # Sorted in decreasing order
        all_heights = np.sort(list(curr_items_code.height.unique()))
        all_heights = all_heights[::-1]  # Sorted in decreasing order

        # Keep track of used items:
        used_items_arr = np.zeros((len(curr_items_code.index),))

        # Make sure there is 1 item per ID
        all_items_ids, counts_current_items = np.unique(
            curr_items_code.id_item, return_counts=True
        )
        assert all(counts_current_items == 1), "Duplicate items are present!"

        for i, row in curr_items_code.iterrows():
            # Check the item was not added to a stack already

            # If the item has not been used yet
            if used_items_arr[i] == 0:
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
                    while j < len(all_heights) and all_heights[j] <= remaining_height:
                        # Try to use an item whose height is the biggest possible, if not available
                        valid_df = curr_items_code[
                            curr_items_code.height == all_heights[j]
                        ]

                        k = 0
                        while k < len(valid_df.index):
                            possib_elem = valid_df.iloc[k]
                            if used_items_arr[list(valid_df.index)[k]] == 0:
                                # Try adding it
                                if (
                                    new_stack.add_item_override(
                                        possib_elem, other_constraints
                                    )
                                    == 1
                                ):
                                    used_items_arr[list(valid_df.index)[k]] += 1
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
                    while j < len(all_weights) and all_weights[j] <= remaining_weight:
                        valid_df = curr_items_code[
                            curr_items_code.weight == all_weights[j]
                        ]

                        k = 0
                        while k < len(valid_df.index):
                            possib_elem = valid_df.iloc[k]
                            if used_items_arr[list(valid_df.index)[k]] == 0:
                                # Try adding it
                                if (
                                    new_stack.add_item_override(
                                        possib_elem, other_constraints
                                    )
                                    == 1
                                ):
                                    used_items_arr[list(valid_df.index)[k]] += 1
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
                    used_items_arr[i] += 1

                # if a new stack is needed (unable to add elements):
                if new_stack_needed:
                    if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
                        stacks_list.append(new_stack)
                    # Open new stack (with current element as first)
                    new_stack = Stack(row, other_constraints)
                    used_items_arr[i] += 1
                    new_stack_needed = False

            else:
                # The item has been used already
                pass

        # Need to add last stack to the list (if not empty)
        if len(new_stack.items) > 0 and new_stack.tot_weight > 0:
            stacks_list.append(new_stack)

        assert all(
            used_items_arr == 1
        ), f"For code {code}, {sum(1-used_items_arr)} items have not been used\nUnused: {np.argwhere(used_items_arr == 0)}"

    for j in range(len(stacks_list)):
        stacks_list[j].assignID(id)
        id += 1

    assert all([s.tot_weight for s in stacks_list]) > 0

    n_items_post = len(df_items.index)
    assert (
        n_items_init == n_items_post
    ), f"The number of items before was {n_items_init}, but after it is {n_items_post}"

    # Check validity of stacks
    if checkValidStacks(stacks_list, truck, df_items, compareItems=True):
        return stacks_list, id
    else:
        raise ValueError("Invalid stacks have been created!")


def refill_stacks(stacks_list, df_items, truck, id, stack_creation):
    """
    refill_stacks
    ---
    Refill the stacks with the items that were previously discarded.

    The idea is that during slice creation, some stacks are broken and
    it may be more effective to take the removed items and try to fill
    up stacks which had been created in a non-efficient manner before
    (e.g., they would allow for some other items on top).
    This way it is possible to achieve better stacks without having
    to rebuild all stacks from scratch at each truck.

    The method will first try to refill the current stacks, and then
    it will feed the remaining stacks to 'create_stacks_cs', meaning
    that all items of the dataframe will be used.

    ### Input parameters:
    - stacks_list: list of stacks to be filled
    - df_items: dataframe containing discarded items
    - truck: Series object containing truck information
    - id: ID of last stack (to avoid duplicate IDs)
    - stack_creation: method used to build the stacks (e.g.,
    create_stacks_cs)

    ### Output parameters:
    - stacks_list: updated list of stacks
    - id: id of the last stack
    """
    other_constraints = {  # Enforce constraints at stack creation rather than after
        "max_height": truck["height"],
        "max_weight": truck["max_weight_stack"],
        "max_dens": truck["max_density"],
    }

    it_added = []

    assert len(np.unique(df_items["id_item"].values)) == len(
        df_items.index
    ), "The recycled item df contains duplicate"

    for i in range(len(stacks_list)):
        # Make sure it is still possible to add items to the stack
        if not stacks_list[i].isMaxStack():
            st_code = stacks_list[i].stack_code

            # Isolate items with that stackability code
            curr_items = df_items[df_items.stackability_code == st_code]

            if len(curr_items.index) > 0:
                min_wt = min(curr_items["weight"].values)
                min_ht = min(curr_items["height"].values)

                wt_left = (
                    min(
                        truck["max_weight_stack"],
                        stacks_list[i].area * truck["max_density"],
                    )
                    - stacks_list[i].tot_weight
                )
                ht_left = truck["height"] - stacks_list[i].tot_height

                n_added = 0
                if min_wt < wt_left and min_ht < ht_left:
                    for j, row in curr_items.iterrows():
                        # Can try to add items to the
                        result_code = stacks_list[i].add_item_override(
                            row, other_constraints
                        )
                        if result_code == 1:
                            # Item was added
                            it_added.append(row["id_item"])
                        elif result_code == 0:
                            # Can stop the loop before
                            break
                        else:
                            # If here, constraints were violated during item addition
                            pass
        # Here, can remove used items from the items list
        for it_id in it_added:
            df_items = df_items[df_items.id_item != it_id]

    # With the remaining items, build new stacks
    other_stacks, id = stack_creation(df_items, truck, id)

    stacks_list += other_stacks

    if checkValidStacks(stacks_list, truck, compareItems=False):
        return stacks_list, id
    else:
        raise ValueError("Invalid stacks have been created!")


def checkValidStacks(stacks_list, truck, df_items=None, compareItems=False):
    """
    checkValidStacks
    ---
    Check the created stacks are valid.

    The requirements are that, if the stacks contain objects from 'items',
    each item has exactly been used once.
    """
    if df_items is not None:
        # print(df_items.columns)
        # Check that the provided items list does not contain duplicates
        items_ids = np.array(list(df_items["id_item"]))
        unique_ids, counts = np.unique(items_ids, return_counts=True)
        assert all(counts == 1), "The provided items list contains duplicates!"

    # Extract the used item ids from the stacks
    stack_items_ids = []  # IDs of all items in the stacks which have been used
    for st in stacks_list:
        stack_items_ids += [it.id_item for it in st.items]

    stack_items_ids = np.array(stack_items_ids)

    used_unique_ids, used_counts = np.unique(stack_items_ids, return_counts=True)

    if compareItems:
        # Compare the number of used items with the number of provided ones
        assert len(used_unique_ids) == len(
            unique_ids
        ), f"The stacks contain {len(used_unique_ids)} elements, while {len(unique_ids)} have been provided"

        # Check that of the provided elements each one has exactly been used once

        for i in range(len(items_ids)):
            if items_ids[i] in used_unique_ids:
                assert (
                    used_counts[used_unique_ids == items_ids[i]] <= 1
                ), f"[i = {i}] -> Element {unique_ids[i]} has been used {used_counts[used_unique_ids == items_ids[i]]} times in the solution!"

    # Check that each item appearing in the used IDs is present only once
    if not all(used_counts <= 1):
        for j in range(len(used_counts)):
            assert (
                used_counts[j] <= 1
            ), f"Item {used_unique_ids[j]} was used {used_counts[j]} times"
        return False

    # Check that each stack is lower than the truck height
    truck_height = truck.height
    assert all(
        st.tot_height <= truck_height for st in stacks_list
    ), "Some stacks are higher than allowed!"

    return True
