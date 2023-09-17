#!/usr/bin/python3
# -*- coding: utf-8 -*-
class Stack:
    """  
    Class used to stock the and modify the information of each stack
    ----------------------------------------------------------------

    #### INPUT PARAMETERS:
        - stack_code: stackability code of the stack
        - length: length of the stack
        - width: width of the stack
        - orient: orientation of the stack
    -----------------------------------------------------------------
    """
    def __init__(self, stack_code, length, width, orient):
        self.stack_code = stack_code
        self.length = length
        self.width = width
        self.max_stack = None
        self.orient = orient
        self.items = []
        self.h_items = []
        #self.max_stack_code = []
        self.weight = 0
        self.height = 0
        self.area = self.length * self.width
        self.state = None
        self.n_items = 0
        self.old_Nest_Height = 0
        self.vertexes = []

    def addItem(self, item, constraints):
        """ 
        addItem
        -------
        Method to check the constrain of the current stack and add the item to it 

        #### INPUT PARAMETERS:
            - item: the item to be added
            - constraints: the constrain that must be respected to add the item to the stack
        """

        if self.max_stack == None:
            self.max_stack = item.max_stackability
        elif item.max_stackability < self.max_stack:
            self.max_stack = item.max_stackability
        
        if self.max_stack == self.n_items: 
            return 0
        
        if item.height + self.height > constraints["max_height"]:
            return -1
        
        if item.weight + self.weight > constraints["max_weight_stack"]:
            return -2
        
        density = (item.weight + self.weight) / self.area
        if density > constraints["max_density"]:
            return -3
        
        # Nesting height is always removed in the stack because i can not know if this is the last item of the stack
        # If this item is the last of the stack, the nest height will be summed at the end of the procedure
        h_item = item.height - item.nesting_height
        self.old_Nest_Height = item.nesting_height

        # 
        #self.max_stack_code.append(item.max_stackability)

        # Each item is a tuple with ID and item's height
        self.items.append(item.id_item)
        self.h_items.append(h_item) #NOTE: è giusto h_item oppure item.height
        self.height += h_item
        self.weight += item.weight
        self.n_items += 1
        return 1
    
    def updateWeight(self, item_weight):
        self.weight += item_weight
    
    def updateHeight(self):
        # Used to add the last nesting height removed from the last item of a stack
        self.height += self.old_Nest_Height
    
    def position(self, x_origin, y_origin):
        """ 
        position
        --------
        Method to give a spacial position to a stack by defining its vertexes

        #### INPUT PARAMETERS:
            - x_origin, y_origin: position of the left-down vertex of the stack
        """
        self.vertexes = [
            (x_origin, y_origin),
            (x_origin + self.length, y_origin),
            (x_origin, y_origin + self.width), 
            (x_origin + self.length, y_origin + self.width)
        ]