from sub.ACO import ACO
from sub.utilities import *
from sub.stack import Stack


class aco_bin_packing(ACO):
    
    
    def __init__(
            self, alpha=1, beta=1, 
            n_ants=70, n_iter=10, evaporationCoeff=0.2
            ):
        self.vehicle = None
        self.stack_lst = []
        self.stack_quantity = [] #TODO: cercare miglior modo di inizializzare (guarda in build stacks)
        super().__init__(alpha, beta, n_ants, n_iter, evaporationCoeff)
     
    def aco_2D_bin(self): #################################################### NON PRENDE LA SOLUZIONE MIGLIOREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
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
        
        bestArea = 0
        for iter in range(self.n_iter):
            self.ants = []
            antsArea = []
            for _ in range(self.n_ants):
                free_space = True 
                prev_s_code = len(self.pr_move)-1 # empty vehicle state
                # initialize solution of ant k
                
                x_pos = 0
                y_pos = y_max = 0
                totArea = 0
                ant_k = []
                stack_lst_ant = self.stack_lst.copy()
                stack_quantity_ant = self.stack_quantity.copy()
                pr_move = self.pr_move.copy()
                while(free_space):  # loop until free space available in vehicle
                    next_s_code = self.choose_move(prev_s_code, pr_move)
                    code = next_s_code 
                    new_stack, stack_lst_ant, stack_quantity_ant = popStack(stack_lst_ant, stack_quantity_ant, next_s_code, n_code) #TODO: what if no more stacks with this stack code??
                    
                    if next_s_code >= n_code:
                        code = code - n_code

                    if stack_quantity_ant[code] == 0: # when the stack with this stackability code are 0 the probability to move 
                        #self.pr_move[next_s_code,:] = 0  # to this state will become also equal to 0
                
                        if next_s_code >= n_code: #TODO: redistribuire le probabilità che si cancellano sulle altre ancora esistenti.
                            prob_to_distr = pr_move[:,next_s_code] + pr_move[:,next_s_code-n_code]
                            pr_move[:,next_s_code] = 0
                            pr_move[:,next_s_code-n_code] = 0
                            prob_to_distr = prob_to_distr/pr_move[:, pr_move.any(0)].shape[1]
                            pr_move[:, pr_move.any(0)] +=  prob_to_distr.reshape(-1,1)
                            
                        else:
                            prob_to_distr = pr_move[:,next_s_code] + pr_move[:,next_s_code+n_code]
                            pr_move[:,next_s_code] = 0
                            pr_move[:,next_s_code+n_code] = 0
                            prob_to_distr = prob_to_distr/pr_move[:, pr_move.any(0)].shape[1]
                            pr_move[:, pr_move.any(0)] +=  prob_to_distr.reshape(-1,1)

                    toAddStack, x_pos, y_pos, y_max = self.addStack(new_stack, x_pos, y_pos, y_max)
                    if toAddStack is not None:
                        ant_k.append(toAddStack)
                        totArea += (toAddStack.length*toAddStack.width)
                        prev_s_code = next_s_code
                    if toAddStack is None or sum(stack_quantity_ant) == 0:
                        free_space = False
                
                self.ants.append(ant_k)
                antsArea.append(totArea)

            # valutare la bontà tra tutte le soluzioni -> migliore = max     peggiore = min
            deltaTrail = self.trailUpdate(antsArea)
            #zero_indexes =  np.where(~self.pr_move.any(0))[0]
            self.trailMatrix = self.evaporationCoeff*self.trailMatrix + deltaTrail
            #self.trailMatrix[:,zero_indexes] = 0
            self.prMoveUpdate()

            area_ratio = max(antsArea)/(self.vehicle['length'] * self.vehicle['width'])
            if area_ratio > bestArea:   # best solution during all the iteration
                bestAnt = self.ants[np.argmax(antsArea)]
                bestArea = area_ratio
            
            if area_ratio >= 0.9:
                if iter >= 3:
                    break
                self.evaporationCoeff = 0.9
            elif area_ratio >= 0.8:
                self.evaporationCoeff = 0.7
            elif area_ratio >= 0.6:
                self.evaporationCoeff = 0.4
            

        print(bestArea)
        if bestArea < 0.7:
            print("#########", self.vehicle['id_truck'])

        self.solCreation(bestAnt)
        return self.sol
    
    
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
                #BUG: toAddStack non deve essere rimosso da stack_lst
        return toAddStack, x_pos, y_pos, y_max

    def statesCreation(self, stackInfo):
        """ 
        statesCreation
        --------------
        Create and initialize the probability of a move matrix

        Note: 
            - matrix[0:N_code] are lengthwise
            - matrix[N_code:2*N_code] are widthwise
            - matrix last state is the empty
            
        #### INPUT PARAMETERS:
            - stackInfo: dataframe containing all the stackability codes
                                and their forced orientation
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

        - vehicle: vehicle type, needed to check the height for
                   creating the stacks for this specific truck
        """
        
        # stack creation: adesso è fatta in modo molto stupido ma dato che item con lo stesso
        # stackability code possono avere diverse altezze probabilmente si può ottimizzare molto 
        # date le diverse altezze dei trucks
        #TODO: controllo max density
        self.vehicle = vehicle
        stackability_codes = df_items.stackability_code.unique()
        self.stack_lst = []
        self.stack_quantity = [0] * len(stackability_codes)
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
                if stack.weight > vehicle['max_weight_stack']:
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
                    
    
    def solCreation(self, bestAnt):
        """
        solCreation
        -----------

        Function used to create the truck solution that is saved in file.

        Parameters
        - bestAnt: bestAnt is the ant that have obtained the best \n
                            solution during the ACO bin_packing.
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
            #TODO: return sol without calling aco.sol (fox disapproves)