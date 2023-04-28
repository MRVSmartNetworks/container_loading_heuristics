import numpy as np
from utilities import *
class ACO:
    """  
    Ant Colony Optimization
    ---------

    #### INPUT PARAMETERS:
        - stack_lst: list of stacks (format: [[I001, I002, NaN], [I003, I005, NaN]])
        - alpha: realative trail importance (default 1)
        - beta: relative attractivness importance (default 1)
        - n_ants: number of ants
    #### ACO PARAMETERS:
        - attr(η): matrix of attractiveness from state i to j
        - trail(τ): matrix of trails from state i to j
        - pr_move: 8x8 matrix of probabilities of moves from i to j 
            (state 7 is related to empty vehicle)
    """
    def __init__(self, stack_lst, vehicle, alpha=1, beta=1, n_ants=10):
        self.stack_lst = stack_lst
        self.vehicle = vehicle
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants

        self.pr_move = pr_move = np.full((8,8), 1./7) * np.array([1, 1, 1, 1, 1, 1, 1, 0])
        
    
    def aco_2D_bin(self):
        """ 
        aco_2D_bin

        Method to solve 2D bin packing problem  
        """
        #TODO: outer loop contaning a termination condition (no. of iterations, solution's goodness???)
        ants = []
        for k in range(self.n_ants):
            free_space = True 
            prev_s_code = 6 # empty vehicle state
            # initialize solution of ant k
            sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
            }
            x_pos = 0
            y_pos = y_max = 0
            ant_k = []
            while(free_space):  # loop until free space available in vehicle
                next_s_code = self.choose_move(self.pr_move, prev_s_code) 
                # ants[ant_k].append()
                new_stack, stack_lst = popStack(stack_lst, next_s_code) #TODO: what if no more stacks with this stack code??
                #TODO: se widthwise mettere come length la width
                
                toAddStack, x_pos, y_max = self.addStack(new_stack, x_pos, y_pos, y_max)
                ant_k.append(toAddStack)
                prev_s_code = next_s_code
                #TODO: controllo se free space
    
    def addStack(self, toAddStack, x_pos, y_pos, y_max):
        """  
        addStack
        ---------

        Method to add a new stuck in vehicle.
        Initially the stacks are added on the X axis until the max length
        of the vehicle is reached.

        Parameters
        - toAddStack: the stack to be added
        - x_pos: update position on X axis considering previously added stacks
        - y_pos: update position on Y axis considering previously added stacks
        - y_max: the max width of a stack in a row in order to build the next
                row staring from y_max
        """
        if x_pos + toAddStack.length < self.vehicle['length']:
            toAddStack.position(x_pos, y_pos)
            x_pos += toAddStack.length
            if toAddStack.width > y_max:
                y_max = toAddStack.width
        else:
            x_pos = 0
            y_pos = y_max
            y_max = 0
            if y_pos + toAddStack.width < self.vehicle['width']:
                toAddStack.position(x_pos, y_pos)
                x_pos += toAddStack.length
                if toAddStack.width > y_max:
                    y_max = toAddStack.width
            else:
                #TODO: spazio nel camion esaurito
                pass
        return toAddStack, x_pos, y_max