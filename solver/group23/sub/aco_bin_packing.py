#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from solver.group23.sub.utilities import *
from solver.group23.sub.stack import Stack
from solver.group23.config import *
from copy import deepcopy, copy

class ACO:
    """  
    Ant Colony Optimization for bin packing problem
    --------------------------------------------------------------------------------

    #### INPUT PARAMETERS:
        - stackInfo: information related to the items given their stackability code
        - alpha: realative trail importance
        - beta: relative attractivness importance
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
    def __init__(self, stackInfo, alpha=1, beta=1, n_ants=50, n_iter=20, evaporationCoeff = 0.2):
        
        self.stackInfo = stackInfo
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants 
        self.evaporationCoeff = evaporationCoeff
        self.n_iter = n_iter
        self.index_code = {i:code for i,code in enumerate(stackInfo.stackability_code)}

        # Number of different stackability codes
        self.n_code = len(self.stackInfo.stackability_code) 
        # The state added is fot the empty vehicle
        self.dim_matr = 2 * self.n_code + 1 

        # First initialization of the most iportant matrix used by the ACO
        self.trailMatrix = np.ones([self.dim_matr, self.dim_matr])  
        self.pr_move = np.zeros([self.dim_matr, self.dim_matr]) 
        self.attractiveness = np.zeros([self.dim_matr, self.dim_matr]) 

        self.ants = []
        self.vehicle = None
        self.stack_lst = []

    #####################################################################################################
    ######### ACO function
     
    def aco_2D_bin(self, last_iter): 
        """ 
        aco_2D_bin
        ----------

        Method to solve 2D bin packing problem.

        #### INPUT PARAMETERS:
            - last_iter: When True the left items can be stored all in the selected vehicle.\n
                            In this way ACO parameters are changed to boost performance.
        """

        # When True the parameters of the ACO are changed.
        # The number of iteration and ants are highly increased the probability for the ACO
        # to complete the simulation with the filling of the last truck.
        if last_iter:
            self.n_ants += int(0.5 * self.n_iter)
            self.n_iter += int(0.5 * self.n_iter)
        
        self.statesCreation()
        bestArea = 0
        good_sol = False
        _iter = 0

        # All the ACO is done n_iter time or until a good solution is found.
        # The good solution is found when the filled area ratio of the truck reach a certain treshold
        while _iter < self.n_iter and not good_sol:
            self.ants = []
            antsArea = []
            antsWeight = []

            # Iteration over all the ants. Every ants will find its own solution putting the stack as they want.
            # At the end only the best solution of all the ants will be taken into account.
            for _ in range(self.n_ants):
                stack_lst_ant = [stack for stack in self.stack_lst] #NOTE: [ele for ele in stack_lst] better????
                stack_quantity_ant = self.stack_quantity.copy()
                pr_move = self.pr_move.copy()
                
                # Variables initialization for each ant
                # Bool to check if free space available in vehicle
                free_space = True 
                # The first code is the empty vehicle state
                prev_s_code = 2*self.n_code 
                # Position initialization at all 0
                x_pos= y_pos = y_max = 0    
                totArea = 0
                totWeight = 0
                totVolume = 0
                ant_k = []  # It will contain the list of stacks used by current ant
                first_line = True
                n_1_line = 0

                """
                Slice definition:
                It is a list of stacks that are placed along the y direction.
                """
                bound = [[0, 0], [0, self.vehicle["width"]]]
                                

                # Loop until free space available in vehicle
                while(free_space):  
                    # Choose of the next state and taking the stack selected:
                    # next_s_code = self.choose_move(prev_s_code, pr_move)    
                    # new_stack, stack_lst_ant, stack_quantity_ant = self.popStack(stack_lst_ant, stack_quantity_ant, next_s_code)
                    # toAddStack, first_line, x_pos = self.addStack(new_stack, ant_k[:n_1_line], first_line, x_pos)

                    ###

                    # TODO:
                    new_slice, stack_lst_ant, stack_quantity_ant, next_s_code = self.buildSlice(prev_s_code, pr_move, bound, stack_lst_ant, stack_quantity_ant, totWeight)

                    if new_slice != []:
                        # Some element was placed!
                        for s in new_slice:
                            st = s[0]
                            ant_k.append(st)

                            # Update the measurements
                            totArea += (st.length*st.width)
                            totVolume +=(st.length*st.width*st.height)
                            totWeight += st.weight
                            
                        # Push the slice
                        bound = self.pushSlice(bound, new_slice)
                    else:
                        # No element could be placed - there is no space left
                        free_space = False

                    ####

                    # Check if there are stacks left
                    if sum(stack_quantity_ant.values()) > 0:
                        code = next_s_code 
                        
                        if next_s_code >= self.n_code:
                            code = code - self.n_code

                        # If there are no more stacks of a certain code then set the pr_move to that specific 
                        # code to zero and distribute the probability over the others rows(stackability codes)
                        if stack_quantity_ant[self.state_to_code(code)] == 0: 
                            prob_to_distr = pr_move[:,code] + pr_move[:,code+self.n_code]
                            pr_move[:,[code, code + self.n_code]] = 0
                            if np.any(pr_move):
                                prob_to_distr = prob_to_distr/pr_move[:, pr_move.any(0)].shape[1]
                                pr_move[:, pr_move.any(0)] +=  prob_to_distr.reshape(-1,1)
                    else:
                        free_space = False

                # Save the ant solution
                self.ants.append(ant_k)
                antsArea.append(totArea)
                antsWeight.append(totWeight)
            
            # Evaluate the trail update  
            deltaTrail = self.trailUpdate(antsArea)
            self.trailMatrix = self.evaporationCoeff*self.trailMatrix + deltaTrail

            # Updating the moves probabilities 
            self.prMoveUpdate()

            # Find the best solution in terms of area ratio with vehicle size
            i_max = np.argmax(antsArea)
            area_ratio = antsArea[i_max]/self.vehicle['area']

            # Saving the best area solution during all the iteration
            if area_ratio > bestArea:   
                bestAnt = self.ants[i_max]
                bestArea = area_ratio 
                weightRatio = antsWeight[i_max]/self.vehicle["max_weight"]

            # Change evaportaion coefficient dynamically given the area ratio
            # At the start the evaporation coefficien will start with a low value (low importance of the ants trail)
            # Only when good solutions will be found the evaporation coefficien will grow increasing
            # the importance of the trail choosen by the ants, increasing the probability of the next
            # ants to follow the already known good solution path with only few variation
            if area_ratio >= AREA_RATIO or weightRatio >= 0.98:
                if _iter >= int(self.n_iter/4):
                    good_sol = True
                self.evaporationCoeff = 0.9
            elif area_ratio >= 0.8:
                self.evaporationCoeff = 0.7
            elif area_ratio >= 0.6:
                self.evaporationCoeff = 0.4
            
            _iter += 1

        if PRINT:    
            print(f"Area ratio: {bestArea},\n Weight ratio: {weightRatio} vehicle: {self.vehicle['id_truck']}")
        return bestAnt, area_ratio, weightRatio
    
    def buildSlice(self, prev_code: int, prob_move_mat, boundary: list, stack_list, stack_quantity_ant, tot_weight) -> [list, list, int]:
        """
        buildSlice
        ---
        Build the new slice to be placed in the truck for the current ant.

        ### Procedure
        - Choose the next code for the stack (choose_move)
        - Extract the first stack with the selected code
        - Add the stack to the slice

        ### Input parameters:
        - prev_code: last selected stackability code
        - prob_move_mat: matrix of transition probabilities
        - boundary: boundary for filling truck (in this case: lay stacks on 'y' axis)
        - stack_list: list of stacks that the current ant can use (stack_lst_ant in aco_2D_bin)
        - tot_weight: total sum of weights of placed stacks (before new slice)

        ### Output parameters:
        - new_slice: list of pairs [Stack, y_coordinate] resulting in the 
        - stack_list: [to be assigned to stack_lst_ant] - the updated list of stacks available to 
        the ant
        - next_s_code: code of the last stack in the slice
        """
        new_slice = []
        y_low = boundary[0][0] # Starting y coordinate
        assert y_low == 0, f"The first element of the boundary is not at y=0 (y={y_low})"
        max_width = boundary[-1][1] # The truck width 
        assert max_width == self.vehicle["width"], f"The last bound point has not y = vehicle['width'] ({max_width} vs {self.vehicle['width']})"

        x_0 = max([p[0] for p in boundary]) # Starting coordinate of slice (before pushing) - sub-optimal
        # Maximum available length (TODO: find some room for improvement... 
        # the bound can be used to find 'holes')
        max_length = self.vehicle["length"] - x_0

        slice_full = False  # Considering: y dimension, total weight of added stacks
        while not slice_full:
            # Used to track the chosen codes - 0 if not already chosen, 1 if chosen
            toAddStack = None
            track_codes = np.ones((len(prob_move_mat[0]),), dtype=np.int32)
            no_more_ok_stacks = False
            next_s_code = 0
            while toAddStack is None and sum(track_codes) > 0:
                # Choice of the next state (stackability code)
                next_s_code = self.choose_move(prev_code, prob_move_mat, track_codes)

                # Extract the first stack with the chosen code - without removing it from the list
                new_stack = self.getFirstStack(stack_list, stack_quantity_ant, next_s_code) # OK

                # Now, make sure that the stack can be added to the current slice
                toAddStack, y_upd = self.addStack2Slice(new_stack, y_low, max_length)

                if toAddStack is None:
                    track_codes[next_s_code] = 0

            # Check if a stack can be added due tue the vehicle weight constrain
            if toAddStack is not None and (tot_weight + toAddStack.weight <= self.vehicle["max_weight"]):
                # Can add stack to the slice
                new_stack, stack_list, stack_quantity_ant = self.popStack(stack_list, stack_quantity_ant, next_s_code)
                new_slice.append([toAddStack, y_upd])                
            else:
                # If not, no more stack can be added to the vehicle (either stack is none, or it would exceed wt.)
                slice_full = True

        return new_slice, stack_list, stack_quantity_ant, next_s_code

    def addStack_simple(self, toAddStack, x_pos, y_pos, y_max):
        """  
        addStack
        --------

        Old method to add a new stuck in vehicle. \n
        Initially the stacks are added on the X axis until the max length 
        of the vehicle is reached. Then on the next line the stacks are postioned
        by looking at the vertexes the first line stacks to not overlap them.

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
            if y_pos + toAddStack.width <= self.vehicle['width']:
                toAddStack.position(x_pos, y_pos)
                x_pos += toAddStack.length
                if toAddStack.width > y_max:
                    y_max = toAddStack.width
            else:
                # No more space in vehicle
                toAddStack = None
        return toAddStack, x_pos, y_pos, y_max
    
    def addStack(self, toAddStack, stack_1_line, first_line, x_pos):
        """  
        addStack
        --------

        Method to add a new stack in vehicle. \n
        Initially the stacks are added on the X axis until the max length 
        of the vehicle is reached. Then on the next line the stacks are postioned
        by looking at the vertexes the first line stacks to not overlap them.

        #### INPUT PARAMETERS:
        - toAddStack: the stack to be added
        - stack_1_line: list with the stacks in the first line
        - first_line: boolean that is true until there is enough space in the
                     first line
        - x_pos: update position on X axis considering previously added stacks
    
        #### OUTPUT PARAMETERS:
        - toAddStack: the stack to be added, if none the stack will not fit
                        in the truck
        - first_line: boolean to keep track if the first line is complete
        - x_pos: X position where to put the stack into the truck
        """
        if first_line:
        
            # Check if the stack does not exceed the vehicle constraint
            if x_pos + toAddStack.length <= self.vehicle['length'] and toAddStack.width <= self.vehicle['width']:
                toAddStack.position(x_pos, 0)
                x_pos += toAddStack.length
                first_line = True
            else:
                first_line = False
                x_pos = 0

        if not first_line:
            
            k = 0
            stack_added = False

            # Loop until the stack is added or there is no more space for it
            # The loop is on the stacks of contained in the first line
            while not stack_added and toAddStack != None and k <len(stack_1_line):
                
                # Check if the possible origin of the new stack is in the middle of the extremes 
                # of the k-th stack
                if x_pos <= stack_1_line[k].vertexes[3][0] and x_pos >= stack_1_line[k].vertexes[2][0]:
                    
                    y_pos_tmp = stack_1_line[k].vertexes[3][1]
                    x_pos_right = x_pos + toAddStack.length
                    
                    j = k
                    # Loop on the stacks which are positioned before the possible right extreme
                    # of the new stack 
                    while j < len(stack_1_line) and stack_1_line[j].vertexes[2][0] <= x_pos_right:
                        
                        # Check if there is an overlap
                        if x_pos_right <= stack_1_line[j].vertexes[3][0] and\
                           x_pos_right >= stack_1_line[j].vertexes[2][0] and\
                           stack_1_line[j].vertexes[3][1] >= y_pos_tmp:
                            
                            # Update the y position of the origin of the new stack
                            y_pos_tmp = stack_1_line[j].vertexes[3][1]
                        
                        j += 1
                    
                    if x_pos + toAddStack.length <= self.vehicle['length'] and y_pos_tmp + toAddStack.width <= self.vehicle['width']:
                        toAddStack.position(x_pos, y_pos_tmp)
                        x_pos += toAddStack.length
                        stack_added = True
                    else:
                        toAddStack = None

                k +=1

                if k == len(stack_1_line) and not stack_added:
                    toAddStack = None
    
        return toAddStack, first_line, x_pos

    def addStack2Slice(self, toAddStack: Stack, y_pos: float, x_avail: float):
        """
        addStack2Slice
        ---
        Try to add the newly selected stack to the current slice.
        The function returns the stack and its upper y coordinate if successful.

        Notice that this function does not check for the weight constraint (this 
        has to be done after).

        If the stack could not fit inside the slice, the function returns Null, y_0.

        ### Input parameters
        - toAddStack: stack that needs to be added to the slice (chosen with 
        transition matrix).
        - y_pos: y coordinate where to add the stack
        - x_avail: available length along the x direction

        ### Output parameters
        - toAddStack/None: if the stack can fit in the slice, return it, else return None
        - y_up/y_pos: if the stack can fit, return the new upper bound for the y coordinate
        """
        if toAddStack.length <= x_avail and y_pos + toAddStack.width <= self.vehicle['width']:
            return toAddStack, y_pos + toAddStack.width
        else:
            return None, y_pos

    def pushSlice(self, bound, new_slice):
        """
        pushSlice
        ---

        [Adapted from solver22]

        Perform the 'push' operation on the new slice and assign the used stacks the 
        coordinates in the solution.

        ### Input parameters
        - bound: current bound - will be updated by the function (NOTE: property of Python
        language - modifying strings in a method also modifies them outside, for how they
        are referenced)
        - new_slice: slice to be pushed; the format is:
          - new_slice[i][0]: Stack object
          - new_slice[i][1]: y coordinate of the object

        ### Return values
        - new_bound: updated bound

        ### Push operation
        - For each new stack 'i':
          - Isolate the points in the current bound which have y coordinates in the range
          [y_origin[i], y_origin[i] + y_dim[i]], being y_origin the y coordinate of the origin
          of the stack (fixed at slice creation) and y_dim the dimension of the stack along
          the y direction (it is the width if not rotated, the length if rotated)
          - The x coordinate of the origin in the stack will be the max value of x for the
          isolated points

        ### Updating the bound
        The new boundary is obtained by determining the vertices of all elements which have been
        placed in last slice.
        Since by definition the boundary has to have as last item a point having as y coordinate
        the truck width, to prevent missing points, a check is performed to possibly add points
        to the new bound to fill the gap.
        """
        new_bound = []

        # Store the index of the first element in the bound which is valid
        for new_stack in new_slice:
            y_i = new_stack[1]
            w_i = new_stack[0].width
            
            # Find lower bound starting from 0
            ind_bound = 0
            while ind_bound < len(bound) and bound[ind_bound][1] <= y_i:
                ind_bound += 1

            if ind_bound < len(bound):
                assert bound[ind_bound][1] > y_i
                ind_bound -= 1
                # This point has the same x coordinate as the one at which
                # the loop was broken and it is for sure not 'above' the
                # current stack
            else:
                raise ValueError("Out of boundary array bounds!")

            # Search for valid points
            ind_top = ind_bound + 0
            # (Needed to prevent to just copy the reference and update both indices...)
            while ind_top < len(bound) and bound[ind_top][1] < y_i + w_i:
                ind_top += 1
            # When the loop finishes, the element bound[ind_top] contains the upper end

            if ind_top >= len(bound):
                assert (
                    bound[ind_top - 1][1] == y_i + w_i
                ), f"The truck width is {bound[-1][1]}, but the item would reach width {y_i + w_i}"
                ind_top -= 1

            # This could happen, e.g., at the beginning
            # assert (
            #     len(bound[ind_bound : ind_top + 1]) > 1
            # ), "The considered elements of the bound are less than 2! Something went wrong"

            # The x coordinate is the max between the x coord of the elements of
            # index between ind_bound and ind_top
            x_i = max([p[0] for p in bound[ind_bound : ind_top + 1]])

            # Update position of stacks
            new_stack.position(x_i, y_i)

            # Update the bound
            # Simply add the points of the 'rightmost' points of the current stack
            l_i = new_stack[0].length
            x_br = x_i + l_i
            y_br = y_i

            x_tr = x_i + l_i
            y_tr = y_i + w_i
            new_bound.append([x_br, y_br])
            new_bound.append([x_tr, y_tr])

        # Fill the bound if the current slice does not reach the full width
        if new_bound[-1][1] < bound[-1][1]:
            # Increase the index from 0 until the element of the old bound is bigger
            ind_extra = 0

            while bound[ind_extra][1] < new_bound[-1][1] and ind_extra < len(bound):
                ind_extra += 1

            # ind_extra locates the 1st corner in the old bound which has y bigger
            # than the current last element in the new bound

            # Add adjustment point:
            # x is the one of the old bound
            # y is the same as the last element in the current bound
            if ind_extra < len(bound):
                new_bound.append([bound[ind_extra][0], new_bound[-1][1]])

                for p in bound[ind_extra:]:
                    new_bound.append(p)

                # if N_DEBUG:
                #     assert (
                #         bound[-1][1] == new_bound[-1][1]
                #     ), f"The last y of the bound does not match - {bound[ind_extra - 1][1]} (old) vs. {new_bound[-1][1]}"

            elif bound[ind_extra - 1][1] < new_bound[-1][1]:
                raise ValueError("The last point of the bound was lost!")

            #else:
                # Ind_extra == len(bound)
                # if N_DEBUG:
                #     assert (
                #         bound[ind_extra - 1][1] == new_bound[-1][1]
                #     ), f"The last y of the bound should have been {bound[ind_extra - 1][1]}, it is instead {new_bound[-1][1]}"

        return new_bound


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
        self.trailMatrix = np.ones([self.dim_matr, self.dim_matr])  
        self.pr_move = np.zeros([self.dim_matr, self.dim_matr]) 
        self.attractiveness = np.zeros([self.dim_matr, self.dim_matr]) 
        # Shared parameters
        code_sub = 1

        # Attractiveness parameter
        best_code1 = 0
        best_code2 = 0
        No_Stack = False
        attr_mat = np.ones((self.dim_matr,self.dim_matr))
        
        # Pr_move parameters
        # Used to put at 0 the row in respect to the stack no more available and the ones with orientation constrain
        pr_mat = np.ones((self.dim_matr,self.dim_matr))
        pr_mat[:,self.dim_matr-1] = 0 #last state is the empty truck, no ants can go in this state apart from the start


        # loop over all the stackability code for checking the orientation and the presence of stack
        for i,code in enumerate(self.stackInfo.stackability_code):
            No_Stack == False

            """ if (self.stackInfo.iloc[i]["length"] * self.stackInfo.iloc[i]["width"]) < 1200000:
                attr_mat[:, self.code_to_state(code)] += 1
                attr_mat[:, self.code_to_state(code) + self.n_code] += 1 """
            
            # If no more stack are present i must set to 0 the prMove columns and rows
            if self.stack_quantity[code] == 0:
                pr_mat[i,:] = 0
                pr_mat[:,i] = 0 
                pr_mat[i+self.n_code,:] = 0
                pr_mat[:,i+self.n_code] = 0
                code_sub += 2
                No_Stack = True

            # Widthwise constrain, check only if some stack with this code are present
            if (self.stackInfo.iloc[i]["forced_orientation"]) == 'w' and No_Stack == False:    
                pr_mat[i,:] = 0
                pr_mat[:,i] = 0
                code_sub += 1

            # Lengthwise constrain, check only if some stack with this code are present
            if (self.stackInfo.iloc[i]["forced_orientation"]) == 'l' and No_Stack == False:    
                pr_mat[i+self.n_code,:] = 0
                pr_mat[:,i+self.n_code] = 0
                code_sub += 1

            # Find the stacks that have the best fit in the trucks width
            # App is updated every time a best fit into the truck is No_Stack 
            app = 0 
            j = 0
            y = 0
            while((j < (len(self.stackInfo) - i)) and (No_Stack == False) and (self.stack_quantity[code] != 0)): 

                l_stack = self.stackInfo.iloc[i]["length"] + self.stackInfo.iloc[j+i]["length"]
                if (l_stack > app) and \
                    (l_stack <= self.vehicle["width"]) and \
                    (self.stack_quantity[self.state_to_code(j+i)] != 0): 

                    app = self.stackInfo.iloc[i]["length"] + self.stackInfo.iloc[j+i]["length"]
                    best_code1 = i + self.n_code
                    best_code2 = j+i + self.n_code

                    # If a perfect fit is found attractiveness matrix must be adjusted
                    if(app == self.vehicle["width"]): 
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                        No_Stack = True # in this cases no more couple of best stack for this code must be found

                w_stack = self.stackInfo.iloc[i]["width"] + self.stackInfo.iloc[j+i]["width"]
                if (w_stack  > app) and (w_stack <= self.vehicle["width"]) and \
                    (self.stack_quantity[self.state_to_code(j+i)] != 0):

                    app = self.stackInfo.iloc[i]["width"] + self.stackInfo.iloc[j+i]["width"]
                    best_code1 = i
                    best_code2 = j+i

                    # If a perfect fit is found attractiveness matrix must be adjusted
                    if(app == self.vehicle["width"]): 
                        No_Stack = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                j += 1

            No_Stack = False
            
            while((y < (len(self.stackInfo))) and (No_Stack == False) and (self.stack_quantity[code] != 0)):

                if (self.stackInfo.iloc[i]["length"] + self.stackInfo.iloc[y]["width"] > app) and \
                    (self.stackInfo.iloc[i]["length"] + self.stackInfo.iloc[y]["width"] <= self.vehicle["width"]) and \
                    (self.stack_quantity[self.state_to_code(y)] != 0): 
                    
                    app = self.stackInfo.iloc[i]["length"] + self.stackInfo.iloc[y]["width"]
                    best_code1 = i + self.n_code
                    best_code2 = y

                    # If a perfect fit is found attractiveness matrix must be adjusted
                    if(app == self.vehicle["width"]):  
                        No_Stack = True
                        attr_mat[:,best_code1] = 2
                        attr_mat[:,best_code2] = 2
                y += 1

            
        attr_mat[:,best_code1] = 2
        attr_mat[:,best_code2] = 2
        
        # Self.pr_move and attractiveness creation with all the information obtained before
        self.pr_move = np.full((self.dim_matr,self.dim_matr), 1./(self.dim_matr-code_sub)) * pr_mat
        self.attractiveness = np.full((len(self.pr_move),len(self.pr_move)), 0.5) * attr_mat * pr_mat
        self.prMoveUpdate()
        #NOTE: Items preferred lengthwise (longest side in respect to the length of the truck)
        #self.attractiveness[:,:self.n_code] = self.attractiveness[:,:self.n_code]*1.5
        
    def choose_move(self, prev_state, pr_move=None, track_choice=None):
        """ 
        choose_move
        -----------
        Function used for choose the next state where the ants will move

        #### INPUT PARAMETERS:
            - pr_move: matrix of probabilities of moves from i to j 
            - prev_state: state of the ant
            - track_choice: [default None] if specified, it indicates which codes can 
            be considered (value = 1)
        #### OUTPUT PARAMETERS:
            - next_state: state where the ant will move
        """
        # At the start the first initialization 
        if pr_move is None: 
            pr_move = self.pr_move
        
        if track_choice is None:
            track_choice = np.ones((pr_move.shape[0],))

        # Select the row from the state the ant was
        row_to_choose = pr_move[prev_state][:] 

        # Selecting the next state where the ant will move
        # The 'track_choice' variable is 1 if the associated code (index) can be considered
        considered_prob = np.zeros((2, sum(track_choice)))  # elements - row 0: indices of nonzero values in track_choice
        j = 0
        for i in range(len(track_choice)):
            if track_choice[i] == 1:
                considered_prob[0, j] = i
                considered_prob[1, j] = row_to_choose[i]
                j += 1

        sum_sel_probs = sum(considered_prob[1, :])
        considered_prob[1, :] = considered_prob[1, :]/sum_sel_probs

        next_state = int(np.random.choice(considered_prob[0, :], p=considered_prob[1, :])) 
        
        return next_state
    
    def prMoveUpdate(self):
        """
        prMoveUpdate
        ------------
        Method used to update the probability to move matrix.
        """

        for i in range(len(self.trailMatrix)):

            # Updating of the central formula of ACO heuristic
            mul = np.power(self.trailMatrix[i, :], self.alpha) * np.power(self.attractiveness[i, :], self.beta)
            _sum = sum(mul)

            # To not divide by zero
            if _sum == 0:   
                _sum = 1   

            # Matrix updating row by row
            self.pr_move[i, :] = mul/_sum    

    def trailUpdate(self, _antsArea):
        """
        trailUpdate
        -----------

        Method used to update the trail matrix. \n
        The previous trail matrix is multiplied by the pheromone evaporation \n
        coefficient and is added to the trail variation derived from the sum \n
        of the contribution of all ants that used move to construct their solution.

        #### INPUT PARAMETERS:
            - _antsArea: list of the area of all the ants
        #### OUTPUT PARAMETERS:
            - deltaTrail: trail matrix of one iteration that is summed to the
                        old trail matrix multiplied by the evaporation coefficient
        """

        # DeltaTrail is filled with 0.01 to not have division by 0 in the pr_move update formula
        deltaTrail = np.full((len(self.pr_move), len(self.pr_move)), 0.01)

        for i,ant in enumerate(self.ants):
             # The first state to start is always the empty truck for all the ants
            x = len(self.pr_move)-1        
            trailApp = np.zeros([len(self.pr_move), len(self.pr_move)])

            # x and y are the position in the state matrix
            for stack in ant:  
                y = stack.state
                trailApp[x,y] += 1
                x = y
            
            # More is the area covered, the better is the solution
            deltaTrail += trailApp * _antsArea[i] / self.vehicle["area"]
        return deltaTrail
    
    def getVehicle(self, vehicle):
        """ 
        getVehicle
        -----
        Get the vehicle for which the 2D bin packing
        problem must be solved
        ### Input parametes:
            - vehicle: the vehicle for which the sol must be found  
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
    
    def popStack(self, stack_lst, stack_quantity, code):
        """ 
        popStack
        --------
        Pop the first stack with the specified code\n
        Note:   
            - the stack is removed form stack_lst
            - if the stack need to be widthwise length and width are swapped

        #### INPUT PARAMETERS:
            - stack_lst: full list containing all the stacks created
            - stack_quantity: dictionary containing all the numbers of the items divided in stackability codes 
            - code: stackability code of the selected item
        """ 

        widthwise = False
        if code >= self.n_code:
            code = code - self.n_code
            widthwise = True

        # Iterate until a stack with the correct code is found
        for i,stack in enumerate(stack_lst):
            if stack.stack_code == self.state_to_code(code):
                #FIXME: deepcopy
                stack_copy = stack_lst[i]
                stack_lst.pop(i)
                stack_quantity[self.state_to_code(code)] -= 1
                stack_copy = deepcopy(stack)
                stack_copy.state = code

                # If the orientation is widhwise length and width must be switched
                if widthwise:
                    stack_copy.state += self.n_code
                    stack_copy.orient = 'w'
                    stack_copy.length, stack_copy.width = stack_copy.width, stack_copy.length
                else:
                    stack_copy.orient = 'l'
                
                return stack_copy, stack_lst, stack_quantity
        raise Exception(f"[popStack]: No more stacks with specified code {code}")
    
    def getFirstStack(self, stack_lst, stack_quantity, code):
        """
        getFirstStack
        ---
        Retrieve the first stack in the list having the specified code.

        ### Input parameters:
            - stack_lst: full list containing all the stacks created
            - stack_quantity: dictionary containing all the numbers of the items divided in stackability codes 
            - code: stackability code of the selected item
        """
        widthwise = False
        if code >= self.n_code:
            code = code - self.n_code
            widthwise = True

        # Iterate until a stack with the correct code is found
        for i,stack in enumerate(stack_lst):
            if stack.stack_code == self.state_to_code(code):
                stack_copy = copy(stack)
                stack_copy.state = code
                if widthwise:
                    stack_copy.state += self.n_code
                    stack_copy.orient = 'w'
                    tmp = stack_copy.width
                    stack_copy.width = stack_copy.length
                    stack_copy.length = tmp
                else:
                    stack_copy.orient = 'l'
                return stack_copy
            
        raise Exception(f"[getStack]: No more stacks with specified code {code}")
    
    def state_to_code(self, index):
        return self.index_code[index]
    
    def code_to_state(self, code):
        return self.stackInfo.index[self.stackInfo['stackability_code'] == code][0]