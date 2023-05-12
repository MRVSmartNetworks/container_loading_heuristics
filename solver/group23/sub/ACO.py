import numpy as np
from numpy.random import choice
from sub.utilities import *

class ACO:
    """  
    Ant Colony Optimization
    --------------------------------------------------------------------------------
    
    #### INPUT PARAMETERS:
        - stack_lst: list of stacks (format: [[I001, I002, NaN], [I003, I005, NaN]])
        - alpha: realative trail importance (default 1)
        - beta: relative attractivness importance (default 1)
        - n_ants: number of ants
        - n_iter: number of iteration, if a good results is obtained not all the 
                    n_iter are done
        - evaporationCoeff: evaporation coefficient of the trail matrix, smaller 
                            it is more the new ant solution will have importance
    #### ACO PARAMETERS:
        - attractiveness (η): N x N matrix of attractiveness from state i to j 
                                (N is the total number of states)
        - trailMatrix (τ): N x N matrix of trails from state i to j
        - pr_move: N x N matrix of probabilities of the moves from i to j 
                    (ultimate state is related to empty vehicle)
    --------------------------------------------------------------------------------
    """
    def __init__(self, alpha=1, beta=1, n_ants=40, n_iter=20, evaporationCoeff = 0.5):
        
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants 
        self.evaporationCoeff = evaporationCoeff
        self.n_iter = n_iter
        self.ants = []               


    def choose_move(self, prev_state, pr_move=None):
        """ 
        choose_move
        -----------
        Function used for choose the next state where the ants will move

        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - prev_state: state of the ant
        #### OUTPUT PARAMETERS:
            - next_state: state where the ant will move
        """
        if pr_move is None: # to run on local pr_move inside aco_bin_packing 
            pr_move = self.pr_move
        row_to_choose = pr_move[prev_state][:] # select the row from the state the ant was
        next_state = int(choice(range(len(row_to_choose)), p=row_to_choose)) # selecting the next state where the ant will move
        
        return next_state 
    
    def prMoveUpdate(self):
        """
        prMoveUpdate
        ------------

        Method used to update the probability to move matrix.
        """
        for i in range(len(self.trailMatrix)):
            mul = np.power(self.trailMatrix[i, :], self.alpha) * np.power(self.attractiveness[i, :], self.beta)
            _sum = sum(mul)
            if _sum == 0:   
                _sum = 1    # to not divide by zero
            self.pr_move[i, :] = mul/_sum
        
        



