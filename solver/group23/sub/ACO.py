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
    def __init__(self, stack_lst, vehicle, alpha=1, beta=1, n_ants=10, n_iter=10, evaporationCoeff = 1):
        self.stack_lst = stack_lst
        self.vehicle = vehicle
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants 
        self.evaporationCoeff = evaporationCoeff
        self.n_iter = n_iter
        
        self.ants = []       
    
    def aco_2D_bin(self):
        """ 
        aco_2D_bin
        ----------

        Method to solve 2D bin packing problem.
        """
        n_code = (len(self.pr_move) - 1)/2  # no. of different stackability codes
        self.trailMatrix = np.zeros([len(self.pr_move), len(self.pr_move)]) # initialization of the trail matrix
        antsArea = []
        #TODO: outer loop contaning a termination condition (no. of iterations, solution's goodness???)
        for iter in range(self.n_iter):
            for k in range(self.n_ants):
                free_space = True 
                prev_s_code = len(self.pr_move)-1 # empty vehicle state
                # initialize solution of ant k
                
                x_pos = 0
                y_pos = y_max = 0
                totArea = 0
                ant_k = []
                stack_lst = self.stack_lst.copy()
                while(free_space):  # loop until free space available in vehicle
                    next_s_code = self.choose_move(prev_s_code) 
                    # ants[ant_k].append()
                    new_stack, stack_lst = popStack(stack_lst, next_s_code, n_code) #TODO: what if no more stacks with this stack code??
                    toAddStack, x_pos, y_max = self.addStack(new_stack, x_pos, y_pos, y_max)
                    if toAddStack is not None:
                        ant_k.append(toAddStack)
                        totArea += (toAddStack.length*toAddStack.width)
                        prev_s_code = next_s_code
                    else:
                        free_space = False
                
                self.ants.append(ant_k)
                antsArea.append(totArea)
                
            # valutare la bontà tra tutte le soluzioni -> migliore = max     peggiore = min
            deltaTrail = self.trailUpdate(antsArea)
            #if iter == 0:
            self.trailMatrix = self.evaporationCoeff*self.trailMatrix + deltaTrail
            #else :
            #    trailMatrix = self.evaporationCoeff*trailMatrix + deltaTrail*(1 - self.evaporationCoeff)
        self.pr_move= self.prMoveUpdate()
                

    def choose_move(self, prev_s_code):
        """ 
        choose_move
        -----------

        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - prev_s_code: stackability code of the last stack placed into the truck
        #### OUTPUT PARAMETERS:
            - next_s_code: stackability code of the next stack to be placed into the truck
        """
        row_to_choose = self.pr_move[prev_s_code][:] # select the row from the stack the ant is moving
        next_s_code = int(choice(range(len(row_to_choose)), p=row_to_choose))
        
        return next_s_code 
    

    def addStack(self, toAddStack, x_pos, y_pos, y_max):
        """  
        addStack
        ---------

        Method to add a new stuck in vehicle. \n
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
                # no more space in vehicle
                toAddStack = None
        return toAddStack, x_pos, y_max

    
    def trailUpdate(self, _antsArea):
        """
        trailUpdate
        -----------

        Method used to update the trail matrix. \n
        The previous trail matrix is multiplied by the pheromone evaporation \n
        coefficient and is added to the trail variation derived from the sum \n
        of the contribution of all ants that used move to construct their solution.

        Parameters
        - _antsArea: list of the area of all the ants
        """
        vehicleArea = self.vehicle['length'] * self.vehicle['width'] 
        deltaTrail = np.zeros([len(self.pr_move), len(self.pr_move)])
        for i,ant in enumerate(self.ants):
            x = len(self.pr_move)-1         # the first state to start is always the empty truck for all the ants
            trailApp = np.zeros([len(self.pr_move), len(self.pr_move)])
            for stack in ant:  # x and y are the position in the state matrix
                y = stack.state
                trailApp[x,y] += 1 #NOTE: forse il +1 qua non va bene, da verificare a programma completo
                x = y
            deltaTrail += trailApp * _antsArea[i] / vehicleArea # more is the area covered, more is the quality of the solution
        return deltaTrail
    
    def prMoveUpdate(self):
        """
        prMoveUpdate
        ------------

        Method used to update the probability to move matrix.

        Parameters
        - 
        """
        """ buff = self.trailMatrix * self.attractiveness
        for i in range(len(buff)):
            _sum = np.sum(buff[i], axis=1)
        vect = 1./np.sum(buff, axis=1)
        pr_move = buff*vect
        pass      """   

    def statesCreation(self, code_orientation):
        """ 
        statesCreation
        --------------
        Create and initialize the probability of a move matrix

        Note: 
            - matrix[0:N_code] are lengthwise
            - matrix[N_code:2*N_code] are widthwise
            - matrix last state is the empty truck
            
        #### INPUT PARAMETERS:
            - code_orientation: dataframe containing all the stackability codes
                                and their forced orientation
        """
        len_matrix = 0
        code_sub = 1
        N_code = len(code_orientation.stackability_code)
        len_matrix = (2*N_code) + 1     # length of the final matrix, the +1 is for adding the state of the empty truck
        mult_mat = np.ones((len_matrix,len_matrix))
        mult_mat[:,len_matrix-1] = 0
        for i,code in enumerate(code_orientation.stackability_code):
            if (code_orientation.iloc[code]["forced_orientation"]) == 'w':    # widthwise constrain
                mult_mat[i,:] = 0
                mult_mat[:,i] = 0
                code_sub += 1

        self.pr_move = np.full((len_matrix,len_matrix), 1./(len_matrix-code_sub)) * mult_mat
        self.attractiveness = np.full((len_matrix,len_matrix), 1) * mult_mat

    
