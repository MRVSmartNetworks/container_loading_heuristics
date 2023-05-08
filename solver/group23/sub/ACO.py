import numpy as np
from numpy.random import choice
from sub.utilities import *

class ACO:
    """  
    Ant Colony Optimization
    -----------------------

    #### INPUT PARAMETERS:
        - stack_lst: list of stacks (format: [[I001, I002, NaN], [I003, I005, NaN]])
        - alpha: realative trail importance (default 1)
        - beta: relative attractivness importance (default 1)
        - n_ants: number of ants
    #### ACO PARAMETERS:
        - attr(η): matrix of attractiveness from state i to j
        - trail(τ): matrix of trails from state i to j
        - pr_move: nxn matrix of probabilities of moves from i to j 
                    (ultimate state is related to empty vehicle)
    """
    def __init__(self, alpha=1, beta=1, n_ants=40, n_iter=20, evaporationCoeff = 0.5):
        
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants 
        self.evaporationCoeff = evaporationCoeff
        self.n_iter = n_iter
        self.ants = []               


    def choose_move(self, prev_state):
        """ 
        choose_move
        -----------

        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - prev_state: stackability code of the last stack placed into the truck
        #### OUTPUT PARAMETERS:
            - next_state: stackability code of the next stack to be placed into the truck
        """
        #BUG: bug probability not equal one
        row_to_choose = self.pr_move[prev_state][:] # select the row from the stack the ant is moving
        next_state = int(choice(range(len(row_to_choose)), p=row_to_choose))
        
        return next_state 
    
    def prMoveUpdate(self):
        """
        prMoveUpdate
        ------------

        Method used to update the probability to move matrix.

        Parameters
        - 
        """
        #TODO: add alpha and beta
        for i in range(len(self.trailMatrix)):
            mul = np.power(self.trailMatrix[i, :], self.alpha) * np.power(self.attractiveness[i, :], self.beta)
            _sum = sum(mul)
            if _sum == 0:   
                _sum = 1    # to not divide by zero
            self.pr_move[i, :] = mul/_sum
        



