from sub.ACO import ACO
from sub.utilities import *
from sub.stack import Stack


class aco_bin_packing(ACO):
    """  
    Ant Colony Optimization for bin packing problem
    --------------------------------------------------------------------------------

    #### INPUT PARAMETERS:
        - stack_lst: stacks list, during the simulation will decrease until empty
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
    #### OUTPUT PARAMETERS:
        - self.sol: solution dictionary containing all the stack and their 
                    information contained by the truck choosen to be filled
    --------------------------------------------------------------------------------
    """
    def __init__(
            self, alpha=1, beta=1, 
            n_ants=40, n_iter=40, evaporationCoeff=0.2
            ):
        self.vehicle = None
        self.stack_lst = []
        self.stack_quantity = [] #TODO: cercare miglior modo di inizializzare (guarda in build stacks)
        super().__init__(alpha, beta, n_ants, n_iter, evaporationCoeff)


    #####################################################################################################
    ######### ACO function
     
    def aco_2D_bin(self): 
        """ 
        aco_2D_bin
        ----------

        Method to solve 2D bin packing problem.
        """

        self.sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": []
        }

        n_code = int((len(self.pr_move) - 1)/2)  # no. of different stackability codes
        self.trailMatrix = np.zeros([len(self.pr_move), len(self.pr_move)]) # initialization of the trail matrix
        vehicleArea = self.vehicle['length'] * self.vehicle['width']
        bestArea = 0
        good_sol = False
        _iter = 0
        while _iter < self.n_iter and not good_sol:
            self.ants = []
            antsArea = []
            for _ in range(self.n_ants):
                stack_lst_ant = self.stack_lst.copy()
                stack_quantity_ant = self.stack_quantity.copy()
                pr_move = self.pr_move.copy()

                # Variables initialization for each ant
                free_space = True # bool to check if free space available in vehicle
                prev_s_code = len(self.pr_move)-1 # empty vehicle state
                x_pos= y_pos = y_max = 0    # position initialization
                totArea = 0
                totWeight = 0
                ant_k = []

                while(free_space):  # loop until free space available in vehicle
                    next_s_code = self.choose_move(prev_s_code, pr_move)    # choose next state
                    new_stack, stack_lst_ant, stack_quantity_ant = popStack(stack_lst_ant, stack_quantity_ant, next_s_code, n_code)
                    toAddStack, x_pos, y_pos, y_max = self.addStack(new_stack, x_pos, y_pos, y_max)
                    
                    # Check if a stack can be added
                    if toAddStack is not None and (totWeight + toAddStack.weight <= self.vehicle["max_weight"]):
                        ant_k.append(toAddStack)
                        totArea += (toAddStack.length*toAddStack.width)
                        totWeight += toAddStack.weight
                        prev_s_code = next_s_code
                    else:
                        free_space = False
                    # Check if there are staks left
                    if sum(stack_quantity_ant) > 0:
                        code = next_s_code 
                        
                        if next_s_code >= n_code:
                            code = code - n_code

                        # if there are no more stacks of a certain code then set the
                        # pr_move to zero and distribute the probability over rows
                        if stack_quantity_ant[code] == 0: 
                            prob_to_distr = pr_move[:,code] + pr_move[:,code+n_code]
                            pr_move[:,[code, code + n_code]] = 0
                            prob_to_distr = prob_to_distr/pr_move[:, pr_move.any(0)].shape[1]
                            pr_move[:, pr_move.any(0)] +=  prob_to_distr.reshape(-1,1)
                    else:
                        free_space = False
                
                self.ants.append(ant_k)
                antsArea.append(totArea)
            
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
            

        print(f"Area ratio: {bestArea}, vehicle: {self.vehicle['id_truck']}")
        if bestArea < 0.7:
            print("#########", self.vehicle['id_truck'])

        # Create the solution format
        self.solCreation(bestAnt)
        return self.sol
    
    
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

    def statesCreation(self, stackInfo):
        """ 
        statesCreation
        --------------
        Create and initialize the probability of a move matrix
        and the matrix of attractiveness.

        Note: 
            - matrix[0:N_code] are lengthwise (lengthwise in respect to the length of the truck)
            - matrix[N_code:2*N_code] are widthwise
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
        N_code = len(stackInfo.stackability_code)
        len_matrix = (2*N_code) + 1     # length of the final matrix, the +1 is for adding the state of the empty truck
        code_sub = 1

        # attractiveness parameter
        best_code1 = 0
        best_code2 = 0
        find = False
        attr_mat = np.ones((len_matrix,len_matrix))
        
        #pr_move parameters
        pr_mat = np.ones((len_matrix,len_matrix))  # used to put at 0 the row in respect to the stack no more available and the ones with orientation constrain
        pr_mat[:,len_matrix-1] = 0

        for i,code in enumerate(stackInfo.stackability_code):
            find == False
            if (stackInfo.iloc[code]["forced_orientation"]) == 'w' or self.stack_quantity[code] == 0:    # widthwise constrain
                pr_mat[i,:] = 0
                pr_mat[:,i] = 0
                code_sub += 1
                if self.stack_quantity[code] == 0:
                    pr_mat[i+N_code,:] = 0
                    pr_mat[:,i+N_code] = 0
                    code_sub += 1
                    find == True

            # Find the stacks that have the best fit in the trucks
            app = 0 # Updated every time a best fit into the truck is find 
            j = 0
            y = 0
            while((j < (len(stackInfo) - code)) and (find == False)): 
                if (stackInfo.iloc[code]["length"] + stackInfo.iloc[j+code]["length"] > app) and (stackInfo.iloc[code]["length"] + stackInfo.iloc[j+code]["length"] <= self.vehicle["width"]): 
                    app = stackInfo.iloc[code]["length"] + stackInfo.iloc[j+code]["length"]
                    best_code1 = code + N_code
                    best_code2 = j+code + N_code
                    if(app == self.vehicle["width"]):
                        find = True
                if (stackInfo.iloc[code]["width"] + stackInfo.iloc[j+code]["width"] > app) and (stackInfo.iloc[code]["width"] + stackInfo.iloc[j+code]["width"] <= self.vehicle["width"]): 
                    app = stackInfo.iloc[code]["width"] + stackInfo.iloc[j+code]["width"]
                    best_code1 = code
                    best_code2 = j+code
                    if(app == self.vehicle["width"]): # if a perfect fit is found attractiveness matrix must be adjusted
                        find = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                j += 1

            find = False
            
            while((y < (len(stackInfo))) and (find == False)):
                if (stackInfo.iloc[code]["length"] + stackInfo.iloc[y]["width"] > app) and (stackInfo.iloc[code]["length"] + stackInfo.iloc[y]["width"] <= self.vehicle["width"]): 
                    app = stackInfo.iloc[code]["length"] + stackInfo.iloc[y]["width"]
                    best_code1 = code + N_code
                    best_code2 = y
                    if(app == self.vehicle["width"]):  # if a perfect fit is found attractiveness matrix must be adjusted
                        find = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                y += 1

            find = False
            
        attr_mat[:,best_code1] = 2
        attr_mat[:,best_code2] = 2
        
        self.pr_move = np.full((len_matrix,len_matrix), 1./(len_matrix-code_sub)) * pr_mat
        self.attractiveness = np.full((len(self.pr_move),len(self.pr_move)), 0.5) * attr_mat * pr_mat 
        
        self.attractiveness[:,:7] = self.attractiveness[:,:7]*1.5 #NOTE: metodo tappabuchi per farlo funzionare fino in fondo, si preferisce indiscrinatamente la posizione di stack lengthwise
        
        

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
                trailApp[x,y] += 1 #NOTE: forse il +1 qua non va bene, da verificare a programma completo
                x = y
                
            deltaTrail += trailApp * _antsArea[i] / vehicleArea # more is the area covered, more is the quality of the solution
        return deltaTrail
    
    
    def buildStacks(self, vehicle, df_items):
        """"
        buildStacks
        -----------
        Function used to create the stack with all the specificity of the selected truck.

        #### INPUT PARAMETERS:
            - vehicle: vehicle type, needed to check the height, weight, max density and max 
                        stack weight for creating the stacks for this specific truck
            - df_items: dataframe containing all the items that will be put into the trucks
        #### OUTPUT PARAMETERS:
            - self.stackList: list of all the stack created 
        """
        
        # stack creation: adesso è fatta in modo molto semplice ma dato che item con lo stesso
        # stackability code possono avere diverse altezze probabilmente si può ottimizzare molto 
        # date le diverse altezze dei trucks
        self.vehicle = vehicle
        stackability_codes = df_items.stackability_code.unique()
        self.stack_lst = []
        self.stack_quantity = [0] * len(stackability_codes)
        maxStackDensity = (self.vehicle["length"] * self.vehicle["width"]) * self.vehicle["max_density"] #area stack * vehicle max density = maximum stack weight
        for code in stackability_codes:
            stack_feat = getStackFeatures(df_items, code)
            
            stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
            
            new_stack_needed = False
            iter_items = df_items[df_items.stackability_code == code].head(200)
            for i, row in iter_items.iterrows():
                stack.updateHeight(row.height - row.nesting_height)
                stack.updateWeight(row.weight)
                if stack.height > vehicle['height']:
                    new_stack_needed = True
                if stack.weight > vehicle['max_weight_stack'] or stack.weight > maxStackDensity:
                    new_stack_needed = True
                if stack.n_items == row.max_stackability:
                    new_stack_needed = True
                # if a new stack is needed:
                if new_stack_needed:
                    self.stack_lst.append(stack)
                    self.stack_quantity[code] += 1 # number of the stack with this precise stackability code
                    stack = Stack(code, stack_feat[0], 
                          stack_feat[1], stack_feat[2], stack_feat[3])
                    stack.addItem(row.id_item, row.height - row.nesting_height)
                    stack.updateHeight(row.height - row.nesting_height)
                    stack.updateWeight(row.weight)
                    new_stack_needed = False
                else:
                    # else add the item
                    stack.addItem(row.id_item, row.height - row.nesting_height)

    #####################################################################################################
    ######### Solution creation
                    
    def solCreation(self, bestAnt):
        """
        solCreation
        -----------

        Function used to create the truck solution that is saved in file.

        #### INPUT PARAMETERS:
            - bestAnt: bestAnt is the ant that have obtained the best solution during the ACO bin_packing.
        #### OUTPUT PARAMETERS:
            - self.sol: solution dictionary containing all the stack and their 
                    information contained by the truck choosen to be filled
        """
        for i,stack in enumerate(bestAnt):
            z_origin = 0
            for item in stack.items:
                self.sol['type_vehicle'].append(self.vehicle['id_truck'])
                self.sol['idx_vehicle'].append(0) #TODO: need a way to update the number of that vehicle
                self.sol['id_stack'].append(f"S{i}")    #BUG: doing so the id overwrite
                self.sol['id_item'].append(item[0])
                self.sol['x_origin'].append(stack.x_origin)
                self.sol['y_origin'].append(stack.y_origin)
                self.sol['z_origin'].append(z_origin)
                self.sol['orient'].append(stack.orient)
                z_origin += item[1]