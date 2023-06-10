import os
import math
import pandas as pd
import numpy as np

from solver.group22.config import N_DEBUG


class Stack:
    """
    Stack
    ---
    Class used to model stacks in the 3D bin packing problem.
    Each stack is composed of multiple items sharing the same stackability
    code, placed one on top of the other.

    This class also contains a set of methods used to handle basic operations,
    like item addition, removal and estimation of parameters.
    """

    def __init__(self, item=None, other_constraints=None, stack_id=None):
        """
        Stack
        ---
        Class used to model stacks in the 3D bin packing problem.
        Each stack is composed of multiple items sharing the same stackability
        code, placed one on top of the other.

        This class also contains a set of methods used to handle basic operations,
        like item addition, removal and estimation of parameters.

        ### Input parameters
        - item: pandas Series object containing the first item in the stack; if None
        the stack is initialized empty and it will be needed to call the method
        'add_item' on it with a new item to initialize the parameters
        - other_constraints (default None): dict containing the additional
        constraints, which may be ''max_weight', 'max_height' and 'max_dens'
        """
        self.n_digits_id = 10
        self.items = []  # Elements on top are APPENDED

        self.id = None
        if stack_id is not None:
            self.id = f"S{str(stack_id).zfill(self.n_digits_id)}"

        self.length = 0
        self.width = 0
        self.area = 0
        self.perimeter = 0
        self.stack_code = -1
        self.max_stack = 100000

        self.next_nesting = 0
        self.tot_height = 0
        self.tot_weight = 0
        self.tot_dens = 0
        self.forced_orientation = "n"

        # Some other parameters which are used
        self.price = 0

        if item is not None:
            # If an item is provided, it is added
            # Can already check for constraints
            self.add_item_override(item, other_constraints)

            # To see if it was added, look at the length of
            # the 'items' list

    def add_item(self, newitem, other_constraints=None):
        """
        add_item
        ---
        NOTE: this method is unused (see 'add_item_override')

        Add a new item to the stack.
        The method checks for compatibility of the new object, both concerning the stackability
        and other possible constraints (see input parameter 'other_constraints').

        In this method, if the stack has no forced orientation, but the new
        item has, the object is NOT added.

        This method also works if the stack was empty.
        It is also possible to add further constraints to be checked here, e.g.,
        max_weight, max_height, max_density.

        ### Input parameters
        - newitem: Pandas Series containing the item
        - other_constraints (default None): dict containing the additional
        constraints, which may be ''max_weight', 'max_height' and 'max_dens'

        ### Output values
        - 1: item was correctly added
        - 0: unable to add item (constraints were violated)
        - -1: item was not added because it would have constrained rotation (but other items may be added...)
        """
        if other_constraints is not None:
            # Check other constraints - look for valid keys
            if isinstance(other_constraints, dict):
                # These checks work fine even if the stack is currently empty
                # Max_height:
                if "max_height" in list(other_constraints.keys()):
                    tmp_new_h = self.tot_height + newitem["height"] - self.next_nesting
                    if tmp_new_h > other_constraints["max_height"]:
                        return 0

                if "max_weight" in list(other_constraints.keys()):
                    tmp_new_w = self.tot_weight + newitem["weight"]
                    if tmp_new_w > other_constraints["max_weight"]:
                        return 0

                if "max_dens" in list(other_constraints.keys()):
                    if len(self.items) > 0:
                        # TODO: review this; the if was added to prevent
                        # division by 0 when checking empty stack
                        tmp_new_w = self.tot_weight + newitem["weight"]
                        tmp_new_d = tmp_new_w / (self.area)
                        if tmp_new_d > other_constraints["max_dens"]:
                            return 0

        if (newitem["forced_orientation"] != "n") and self.forced_orientation == "n":
            # This condition is only valid if the stack is not empty
            if len(self.items) > 0:
                # The new item would override free orientation of the stack
                return -1

        # Check stack_code, max_stack and previous flag
        if (
            newitem["stackability_code"] == self.stack_code
            and len(self.items) + 1 <= self.max_stack
        ):
            # Here ONLY IF STACK WAS NOT EMPTY
            # Can add
            self.items.append(newitem)
            # Update parameters
            self.tot_height = self.tot_height + newitem["height"] - self.next_nesting
            self.tot_weight += newitem["weight"]
            self.next_nesting = newitem["nesting_height"]
            self.tot_dens = self.tot_weight / (self.area)

            return 1

        elif len(self.items) == 0:
            # HERE ONLY IF STACK WAS EMPTY
            # Need to initialize the stack parameters
            self.length = newitem["length"]
            self.width = newitem["width"]
            self.area = self.length * self.width
            self.perimeter = 2 * self.length + 2 * self.width
            self.stack_code = newitem["stackability_code"]
            self.max_stack = newitem["max_stackability"]

            # The following need to be changed
            self.next_nesting = newitem[
                "nesting_height"
            ]  # Nesting height of the topmost element
            self.tot_height = newitem["height"]
            self.tot_weight = newitem["weight"]
            self.tot_dens = self.tot_weight / (self.area)
            self.forced_orientation = newitem["forced_orientation"]

            self.items.append(newitem)

            return 1

        return 0

    def add_item_override(self, newitem, other_constraints=None):
        """
        add_item_override
        ---
        Add a new item to the stack.
        The method checks for compatibility of the new object, both concerning the stackability
        and other possible constraints (see input parameter 'other_constraints').

        In this method, if the stack has no forced orientation, but the new
        item has, the object IS added!

        It is also possible to add further constraints to be checked here, e.g.,
        max_weight, max_height, max_density.

        ### Input parameters
        - newitem: Pandas Series containing the item
        - other_constraints (default None): dict containing the additional
        constraints, which may be ''max_weight', 'max_height' and 'max_dens'

        ### Return values (ordered)
        - 0: stackability code exceeded - cannot add any more elements to this stack
        - -1: adding item would exceed max height
        - -2: adding item would exceed max weight
        - -3: adding item would exceed max density
        - 1: success
        """
        if self.isMaxStack():
            return 0

        if other_constraints is not None:
            # Check other constraints - look for valid keys
            if isinstance(other_constraints, dict):
                # Max_height:
                if "max_height" in list(other_constraints.keys()):
                    tmp_new_h = self.tot_height + newitem["height"] - self.next_nesting
                    if tmp_new_h > other_constraints["max_height"]:
                        if N_DEBUG:
                            print("Max_height violated!")
                        return -1

                if "max_weight" in list(other_constraints.keys()):
                    tmp_new_w = self.tot_weight + newitem["weight"]
                    if tmp_new_w > other_constraints["max_weight"]:
                        if N_DEBUG:
                            print("Max_weight_stack violated!")
                        return -2

                if "max_dens" in list(other_constraints.keys()):
                    if len(self.items) > 0:
                        # TODO: review this; the if was added to prevent
                        # division by 0 when checking empty stack
                        tmp_new_w = self.tot_weight + newitem["weight"]
                        tmp_new_d = tmp_new_w / (self.area)
                        if tmp_new_d > other_constraints["max_dens"]:
                            if N_DEBUG:
                                print("Max_density violated!")
                            return -3
            else:
                raise ValueError("Variable 'other_constraints' should be a dictionary")

        # Check stack_code, max_stack and previous flag
        if (
            newitem["stackability_code"] == self.stack_code
            and len(self.items) + 1 <= self.max_stack
        ):
            # Can add
            self.items.append(newitem)
            assert len(self.items) > 0
            # Update parameters
            self.tot_height = self.tot_height + newitem["height"] - self.next_nesting
            self.tot_weight += newitem["weight"]
            self.next_nesting = newitem["nesting_height"]
            self.tot_dens = self.tot_weight / (self.area)

            if (
                newitem["forced_orientation"] != "n"
            ) and self.forced_orientation == "n":
                self.forced_orientation = newitem["forced_orientation"]

            return 1

        elif len(self.items) == 0:
            # HERE ONLY IF STACK WAS EMPTY
            # Need to initialize the stack parameters
            self.length = newitem["length"]
            self.width = newitem["width"]
            self.area = self.length * self.width
            self.perimeter = 2 * self.length + 2 * self.width
            self.stack_code = newitem["stackability_code"]
            self.max_stack = newitem["max_stackability"]

            # The following need to be changed
            self.next_nesting = newitem[
                "nesting_height"
            ]  # Nesting height of the topmost element
            self.tot_height = newitem["height"]
            self.tot_weight = newitem["weight"]
            self.tot_dens = self.tot_weight / (self.area)
            self.forced_orientation = newitem["forced_orientation"]

            self.items.append(newitem)
            assert len(self.items) > 0

            return 1

        return 0

    def assignPrice(self, val):
        """
        assignPrice
        ---
        Assign the price specified in 'val' to the stack.
        """
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, np.int64):
            self.price = val
        else:
            raise ValueError(
                f"The price of the stack has to be either an int or a float! {val} is not valid!"
            )

    def rot90deg(self):
        """
        rot90deg
        ---
        If possible, rotate the stack by 90 degrees (i.e., swap width and length).

        This method returns true if the operation was successful.
        """
        if self.forced_orientation == "n":
            tmp = self.width
            self.width = self.length
            self.length = tmp
            return True
        return False

    def getZList(self):
        """
        getZList
        ---
        Get the list of the z coordinates of the origins of the items
        contained in the stack.

        ### Output value
        - z_lst: list of the z coordinates, ordered from lowest to
        highest (bottom to top)
        """
        z_lst = [0]

        if len(self.items) > 1:
            for i in range(1, len(self.items)):
                last_z = z_lst[-1]
                z_lst.append(
                    self.items[i - 1]["height"] - self.items[i - 1]["nesting_height"]
                )

        return z_lst

    def assignID(self, id_int):
        """
        assignID
        ---
        Method used to assign the ID to the stack.
        The format of the ID is "S__...__", where the '_' represent
        a number which is left-side padded with zeros to reach the
        length specified in 'self.n_digits_id'

        ### Input parameter:
        - id_int: value to be used as ID.
        """
        self.id = f"S{str(id_int).zfill(self.n_digits_id)}"

    def isMaxStack(self):
        """
        isMaxStack
        ---
        Used to check whether a stack has reached max stackability or not.
        It returns True if it has.
        """
        return len(self.items) == self.max_stack

    def removeTopItem(self):
        pass

    def removeBottomItem(self):
        pass

    def removeHeaviestItem(self):
        """
        removeHeaviestItem
        ---
        Remove the heaviest element in the stack.

        The removed element is returned as a Pandas.Series object.
        """
        if len(self.items) > 0:
            wt_vec = [it["weight"] for it in self.items]

            assert len(wt_vec) == len(
                self.items
            ), f"The vector of weights has a wrong length ({len(len(wt_vec))} vs {len(self.items)})"

            ind_max_wt = np.argmax(wt_vec)
            rem_elem = self.items[ind_max_wt]

            if ind_max_wt == len(wt_vec) - 1 and ind_max_wt > 0:
                self.next_nesting = self.items[ind_max_wt - 1]["nesting_height"]
                self.tot_height = self.tot_height - rem_elem.height + self.next_nesting
            else:
                self.tot_height = (
                    self.tot_height - rem_elem.height + rem_elem.nesting_height
                )

            self.tot_weight = self.tot_weight - rem_elem.weight
            if self.tot_height == 0:
                assert (
                    self.tot_weight < 1e-5
                ), "The stack has height = 0, but nonzero weight"
                self.tot_dens = 0
            else:
                self.tot_dens = self.tot_weight / (self.area)

            old_n = len(self.items)

            del self.items[ind_max_wt]

            new_n = len(self.items)
            assert new_n + 1 == old_n, "Item was not removed from stack"

            if all([it.forced_orientation for it in self.items]) == "n":
                self.forced_orientation = "n"

            return rem_elem

        else:
            raise ValueError("Cannot remove element - stack is empty!")
