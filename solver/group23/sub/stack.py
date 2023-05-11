class Stack:
    def __init__(self, stack_code, length, width, max_stack, orient):
        self.stack_code = stack_code
        self.length = length
        self.width = width
        self.max_stack = max_stack
        self.orient = orient
        self.items = []
        self.weight = 0
        self.height = 0
        self.state = None
        self.n_items = len(self.items)

    def addItem(self, id_item, h_item):
        # each item is a tuple with ID and item's height
        self.items.append((id_item, h_item))
        self.n_items += 1
    
    def updateWeight(self, item_weight):
        self.weight += item_weight
    
    def updateHeight(self, item_height):
        self.height += item_height
    
    def position(self, x_origin, y_origin):
        self.x_origin = x_origin
        self.y_origin = y_origin