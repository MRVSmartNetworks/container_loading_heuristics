import os
import math
import pandas as pd
import numpy as np

class Stack:

    def __init__(self, item):
        """
        Stack
        ---
        Class used to model stacks in the 3D bin packing problem.
        Each stack is composed as multiple items sharing the same 
        stackability code placen one onto the other.

        ### Input parameters
        - item: pandas Series object containing the first item in the stack
        """
        self.items = [item]     # Elements on top are APPENDED

        # TODO: assign ID - look at prof code
        self.id = None

        # The following are not changed
        self.length = item["length"]
        self.width = item["width"]
        self.area = self.length*self.width
        self.perimeter = 2*self.length + 2*self.width
        self.stack_code = item["stackability_code"]
        self.max_stack = item["max_stackability"]
        
        # The following need to be changed
        self.next_nesting = item["nesting_height"]  # Nesting height of the topmost element
        self.tot_height = item["height"]
        self.tot_weight = item["weight"]
        self.tot_dens = self.tot_weight/(self.length*self.width)
        self.forced_orientation = item["forced_orientation"]    # A new element may have a different forced orientation

        # Some other parameters which are used
        self.price = 0

################## FIXME: the following methods (add_items) can be built around the returned values
################## As soon as 1 constraint is not respected, return 0

    def add_item(self, newitem, other_constraints=None):
        """
        add_item
        ---
        Add a new item to the stack.
        The method also checks for compatibility of the new object.

        In this method, if the stack has no forced orientation, but the new 
        item has, the object is not added!

        It is also possible to add further constraints to be checked here, e.g., 
        max_weight, max_height, max_density.

        ### Inputs
        - newitem: Pandas Series containing the item
        - other_constraints (default None): dict containing the additional 
        constraints, which may be ''max_weight', 'max_height' and 'max_dens'

        ### Output
        - 1: item was correctly added
        - 0: unable to add item (constraints were violated)
        - -1: item was not added because it would have constrained rotation (but other items may be added...)
        """
        if other_constraints is not None:
            # Check other constraints - look for valid keys
            if isinstance(other_constraints, dict):
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
                    tmp_new_w = self.tot_weight + newitem["weight"]
                    tmp_new_d = tmp_new_w/(self.area)
                    if tmp_new_d > other_constraints["max_dens"]:
                        return 0

        if (newitem["forced_orientation"] != "n") and self.forced_orientation == "n":
            # The new item would override free orientation of the stack
            return -1

        # Check stack_code, max_stack and previous flag
        if newitem["stackability_code"] == self.stack_code and len(self.items)+1 <= self.max_stack:
            # Can add
            self.items.append(newitem)
            # Update parameters
            self.tot_height = self.tot_height + self.newitem["height"] - self.next_nesting
            self.tot_weight += newitem["weight"]
            self.next_nesting = newitem["nesting_height"]
            self.tot_dens = self.tot_weight/(self.length*self.width)
            
            return 1
        
        return 0

    def add_item_override(self, newitem, other_constraints=None):
        """
        add_item_override
        ---
        Add a new item to the stack.
        The method also checks for compatibility of the new object.

        In this method, if the stack has no forced orientation, but the new 
        item has, the object IS added!

        It is also possible to add further constraints to be checked here, e.g., 
        max_weight, max_height, max_density.

        Inputs:
        - newitem: Pandas Series containing the item
        - other_constraints (default None): dict containing the additional 
        constraints, which may be ''max_weight', 'max_height' and 'max_dens'
        """
        if other_constraints is not None:
            # Check other constraints - look for valid keys
            if isinstance(other_constraints, dict):
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
                    tmp_new_w = self.tot_weight + newitem["weight"]
                    tmp_new_d = tmp_new_w/(self.area)
                    if tmp_new_d > other_constraints["max_dens"]:
                        return 0

        # Check stack_code, max_stack and previous flag
        if newitem["stackability_code"] == self.stack_code and len(self.items)+1 <= self.max_stack:
            # Can add
            self.items.append(newitem)
            # Update parameters
            self.tot_height = self.tot_height + self.newitem["height"] - self.next_nesting
            self.tot_weight += newitem["weight"]
            self.next_nesting = newitem["nesting_height"]
            self.tot_dens = self.tot_weight/(self.length*self.width)

            if (newitem["forced_orientation"] != "n") and self.forced_orientation == "n":
                self.forced_orientation = newitem["forced_orientation"]

            return 1
        
        return 0
            
    def assignPrice(self, val):
        self.price = val

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
            for it in self.items[1:]:
                z_lst.append(z_lst[-1]+self.items["height"])
        
        return z_lst
