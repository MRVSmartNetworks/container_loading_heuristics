import os
import numpy as np
from sub.utilities import *
from sub.stack import Stack

class ACO:
    """  
    Ant Colony Optimization for bin packing problem
    --------------------------------------------------------------------------------

    #### INPUT PARAMETERS:
        - stackInfo: information related to the items given
            their stackability code
        - alpha: realative trail importance (default 1)
        - beta: relative attractivness importance (default 1)
        - n_ants: number of ants
        - n_iter: number of iteration, if a good results is obtained not all the 
                    n_iter are done
        - evaporationCoeff: evaporation coefficient of the trail matrix, smaller 
                            it is more the new ant solution will have importance
    #### ACO PARAMETERS:
        - attractiveness (η): N x N matrix of attractiveness from state i to j 
                                (N is the total number of states plus the empyty
                                veichle state)
        - trailMatrix (τ): N x N matrix of trails from state i to j
        - pr_move: N x N matrix of probabilities of the moves from i to j 
                    (ultimate state is related to empty vehicle)
    #### OUTPUT PARAMETERS:
        - bestAnt: the best ant solution in terms of area occupied
    --------------------------------------------------------------------------------
    """
    def __init__(self, stackInfo,alpha=1, beta=1, n_ants=50, n_iter=20, evaporationCoeff = 0.2):
        
        self.stackInfo = stackInfo.sort_values(by=['stackability_code']) # sorting needed for states creation
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants 
        self.evaporationCoeff = evaporationCoeff
        self.n_iter = n_iter

        self.n_code = len(self.stackInfo.stackability_code) # number of different stackability codes
        self.dim_matr = 2 * self.n_code + 1 # the state added is fot the empty vehicle
        self.trailMatrix = np.zeros([self.dim_matr, self.dim_matr]) # initialization of the trail matrix
        self.pr_move = np.zeros([self.dim_matr, self.dim_matr]) # initialization of the probability matrix
        self.attractiveness = np.zeros([self.dim_matr, self.dim_matr]) # initialization of the attractivness matrix
        self.ants = []
        self.vehicle = None
        self.stack_lst = []

    #####################################################################################################
    ######### ACO function
     
    def aco_2D_bin(self): 
        """ 
        aco_2D_bin
        ----------

        Method to solve 2D bin packing problem.
        """
        self.statesCreation()
        vehicleArea = self.vehicle['length'] * self.vehicle['width']
        vehicleVolume = vehicleArea * self.vehicle['height']
        bestArea = 0
        good_sol = False
        _iter = 0
        while _iter < self.n_iter and not good_sol:
            self.ants = []
            antsArea = []
            antsWeight = []
            for _ in range(self.n_ants):
                stack_lst_ant = self.stack_lst.copy() # [ele for ele in stack_lst] better????
                stack_quantity_ant = self.stack_quantity.copy()
                pr_move = self.pr_move.copy()
                
                # Variables initialization for each ant
                free_space = True # bool to check if free space available in vehicle
                prev_s_code = 2*self.n_code # empty vehicle state
                x_pos= y_pos = y_max = 0    # position initialization
                totArea = 0
                totWeight = 0
                totVolume = 0
                ant_k = []

                while(free_space):  # loop until free space available in vehicle
                    next_s_code = self.choose_move(prev_s_code, pr_move)    # choose next state
                    new_stack, stack_lst_ant, stack_quantity_ant = popStack(stack_lst_ant, stack_quantity_ant, next_s_code, self.n_code)
                    toAddStack, x_pos, y_pos, y_max = self.addStack(new_stack, x_pos, y_pos, y_max)
                                
                    # Check if a stack can be added
                    if toAddStack is not None and (totWeight + toAddStack.weight <= self.vehicle["max_weight"]):
                        ant_k.append(toAddStack)
                        totArea += (toAddStack.length*toAddStack.width)
                        totVolume +=(toAddStack.length*toAddStack.width*toAddStack.height)
                        totWeight += toAddStack.weight
                        prev_s_code = next_s_code
                    else:
                        free_space = False
                    # Check if there are staks left
                    if sum(stack_quantity_ant.values()) > 0:
                        code = next_s_code 
                        
                        if next_s_code >= self.n_code:
                            code = code - self.n_code

                        # if there are no more stacks of a certain code then set the
                        # pr_move to zero and distribute the probability over rows
                        if stack_quantity_ant[code] == 0: 
                            prob_to_distr = pr_move[:,code] + pr_move[:,code+self.n_code]
                            pr_move[:,[code, code + self.n_code]] = 0
                            if np.any(pr_move):
                                prob_to_distr = prob_to_distr/pr_move[:, pr_move.any(0)].shape[1]
                                pr_move[:, pr_move.any(0)] +=  prob_to_distr.reshape(-1,1)
                    else:
                        free_space = False

                self.ants.append(ant_k)
                antsArea.append(totArea)
                antsWeight.append(totWeight)
            
            # Evaluate the trail update  
            deltaTrail = self.trailUpdate(antsArea, vehicleArea)
            self.trailMatrix = self.evaporationCoeff*self.trailMatrix + deltaTrail
            # Update moves probabilities 
            self.prMoveUpdate()

            # Find the best solution in terms of area ratio with vehicle size
            area_ratio = max(antsArea)/vehicleArea
            if area_ratio > bestArea:   # best solution during all the iteration
                bestAnt = self.ants[np.argmax(antsArea)]
                bestArea = area_ratio 
                weightRatio = antsWeight[np.argmax(antsArea)]/self.vehicle["max_weight"]

            # Change evaportaion coefficient dynamically given the area ratio
            if area_ratio >= 0.9:
                if _iter >= 3:
                    good_sol = True
                self.evaporationCoeff = 0.9
            elif area_ratio >= 0.8:
                self.evaporationCoeff = 0.7
            elif area_ratio >= 0.6:
                self.evaporationCoeff = 0.4
            
            _iter += 1
            
        print(f"Area ratio: {bestArea},\n Weight ratio: {weightRatio} vehicle: {self.vehicle['id_truck']}")
        return bestAnt

    def addStack(self, toAddStack, x_pos, y_pos, y_max):
        """  
        addStack
        ---------

        Method to add a new stuck in vehicle. \n
        Initially the stacks are added on the X axis until the max length 
        of the vehicle is reached.

        #### INPUT PARAMETERS:
        - toAddStack: the stack to be added
        - x_pos: update position on X axis considering previously added stacks
        - y_pos: update position on Y axis considering previously added stacks
        - y_max: the max width of a stack in a row in order to build the next
                row staring from y_max
        #### OUTPUT PARAMETERS:
        - toAddStack: the stack to be added, if none the stack will not fit
                        in the truck
        - x_pos: X position where to put the stack into the truck
        - y_pos: Y position where to put the stack into the truck
        - y_max: the max width of a stack in a row in order to build the next
                row staring from y_max
        """
        if x_pos + toAddStack.length <= self.vehicle['length'] and y_pos + toAddStack.width <= self.vehicle['width']:
            toAddStack.position(x_pos, y_pos)
            x_pos += toAddStack.length
            if toAddStack.width > y_max:
                y_max = toAddStack.width
        else:
            x_pos = 0
            y_pos = y_max + y_pos
            y_max = 0
            if y_pos + toAddStack.width < self.vehicle['width']:
                toAddStack.position(x_pos, y_pos)
                x_pos += toAddStack.length
                if toAddStack.width > y_max:
                    y_max = toAddStack.width
            else:
                # no more space in vehicle
                toAddStack = None
        return toAddStack, x_pos, y_pos, y_max
    

    #####################################################################################################
    ######### Utilities

    def statesCreation(self):
        """ 
        statesCreation
        --------------
        Create and initialize the probability of a move matrix
        and the matrix of attractiveness.

        Note: 
            - matrix[0:n_code] are lengthwise (lengthwise in respect to the length of the truck)
            - matrix[n_code:2*n_code] are widthwise
            - matrix last state is the empty truck state
            
        #### INPUT PARAMETERS:
            - stackInfo: dataframe containing all the stackability codes, their forced 
                        orientation, length and width
        #### OUTPUT PARAMETERS:
            - self.pr_move: N x N matrix of probabilities of the moves from i to j, 
                            if there are no more stack of a specific stackability 
                            code or a forced orientatio is present, their respective 
                            row and columns will be set to 0               
            - self.attractiveness: N x N matrix of attractiveness from state i to j, 
                                states that fill widthwise the truck are privileged
        """
        # shared parameters
        code_sub = 1

        # attractiveness parameter
        best_code1 = 0
        best_code2 = 0
        find = False
        attr_mat = np.ones((self.dim_matr,self.dim_matr))
        
        #pr_move parameters
        # used to put at 0 the row in respect to the stack no more available and the ones with orientation constrain
        pr_mat = np.ones((self.dim_matr,self.dim_matr))
        pr_mat[:,self.dim_matr-1] = 0 #last state is the empty truck, no ants can go in this state apart from the start

        # loop over all the stackability code for checking the orientation and the presence of stack
        for i,code in enumerate(self.stackInfo.stackability_code):
            find == False
            if (self.stackInfo.iloc[code]["forced_orientation"]) == 'w' or self.stack_quantity[code] == 0:    # widthwise constrain
                pr_mat[i,:] = 0
                pr_mat[:,i] = 0
                code_sub += 1
                if self.stack_quantity[code] == 0: #if no more stack are present i must set to 0 the prMove columns and rows
                    pr_mat[i+self.n_code,:] = 0
                    pr_mat[:,i+self.n_code] = 0
                    code_sub += 1
                    find == True

            # Find the stacks that have the best fit in the trucks
            app = 0 # Updated every time a best fit into the truck is find 
            j = 0
            y = 0
            while((j < (len(self.stackInfo) - code)) and (find == False) and (self.stack_quantity[code] != 0)): 
                if (self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[j+code]["length"] > app) and (self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[j+code]["length"] <= self.vehicle["width"]) and (self.stack_quantity[j+code] != 0): 
                    app = self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[j+code]["length"]
                    best_code1 = code + self.n_code
                    best_code2 = j+code + self.n_code
                    if(app == self.vehicle["width"]): #perfect solution for truck width, instant attractiveness
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                        find = True
                if (self.stackInfo.iloc[code]["width"] + self.stackInfo.iloc[j+code]["width"] > app) and (self.stackInfo.iloc[code]["width"] + self.stackInfo.iloc[j+code]["width"] <= self.vehicle["width"]) and (self.stack_quantity[j+code] != 0): 
                    app = self.stackInfo.iloc[code]["width"] + self.stackInfo.iloc[j+code]["width"]
                    best_code1 = code
                    best_code2 = j+code
                    if(app == self.vehicle["width"]): # if a perfect fit is found attractiveness matrix must be adjusted
                        find = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                j += 1

            find = False
            
            while((y < (len(self.stackInfo))) and (find == False) and (self.stack_quantity[code] != 0)):
                if (self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[y]["width"] > app) and (self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[y]["width"] <= self.vehicle["width"]) and (self.stack_quantity[y] != 0): 
                    app = self.stackInfo.iloc[code]["length"] + self.stackInfo.iloc[y]["width"]
                    best_code1 = code + self.n_code
                    best_code2 = y
                    if(app == self.vehicle["width"]):  # if a perfect fit is found attractiveness matrix must be adjusted
                        find = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                y += 1

            find = False
            
        attr_mat[:,best_code1] = 2
        attr_mat[:,best_code2] = 2
        
        self.pr_move = np.full((self.dim_matr,self.dim_matr), 1./(self.dim_matr-code_sub)) * pr_mat
        self.attractiveness = np.full((len(self.pr_move),len(self.pr_move)), 0.5) * attr_mat * pr_mat 
        
        self.attractiveness[:,:7] = self.attractiveness[:,:7]*1.5 #NOTE: metodo tappabuchi per farlo funzionare fino in fondo, si preferisce indiscrinatamente la posizione di stack lengthwise
        
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
        next_state = int(np.random.choice(range(len(row_to_choose)), p=row_to_choose)) # selecting the next state where the ant will move
        
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

    def trailUpdate(self, _antsArea, vehicleArea):
        """
        trailUpdate
        -----------

        Method used to update the trail matrix. \n
        The previous trail matrix is multiplied by the pheromone evaporation \n
        coefficient and is added to the trail variation derived from the sum \n
        of the contribution of all ants that used move to construct their solution.

        #### INPUT PARAMETERS:
            - _antsArea: list of the area of all the ants
            - vehicleArea: area of the vehicle taken into account
        #### OUTPUT PARAMETERS:
            - deltaTrail: trail matrix of one iteration that is summed to the
                        old trail matrix multiplied by the evaporation coefficient
        """
        deltaTrail = np.full((len(self.pr_move), len(self.pr_move)), 0.01)
        for i,ant in enumerate(self.ants):
            x = len(self.pr_move)-1         # the first state to start is always the empty truck for all the ants
            trailApp = np.zeros([len(self.pr_move), len(self.pr_move)])
            for stack in ant:  # x and y are the position in the state matrix
                y = stack.state
                trailApp[x,y] += 1
                x = y
                
            deltaTrail += trailApp * _antsArea[i] / vehicleArea # more the area covered, the better the solution
        return deltaTrail
    
    def getVehicle(self, vehicle):
        """ 
        getVehicle
        -----
        Get the vehicle for which the 2D bin packing
        problem must be solved
        ### Input parametes:
            - vehicle: the vehicle for which the sol must
                be found  
        """
        self.vehicle = vehicle
    
    def getStacks(self, stacks):
        """ 
        getStacks
        -----
        Get the stacks information need to fill
        the vehicle
        ### Input parameters
            - stacks: is a list where the first element is
                stack list and the second is stack quantity
        """
        self.stack_lst = stacks[0]
        self.stack_quantity = stacks[1]
