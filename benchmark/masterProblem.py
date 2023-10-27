import numpy as np
import gurobipy as gp
from gurobipy import GRB
from configCG import N_INIT_COLS

class MasterProblem:
    """  
    MasterProblem
    ---------------------------------------------------
    Class used to define the master problem used in the
    implementation of the column generation.
    """
    def __init__(self, columns, df_vehicles, df_items):
        self.model = gp.Model("MasterProblem")
        self.obj = 0
        self.columns = columns
        self.df_vehicles = df_vehicles
        self.n_cols = self.columns.shape[1]
        self.items_type = self.columns.shape[0]
        # Evaluate the number of items per type in df_items
        self.n_items_type = np.zeros(self.items_type)
        for i in range(self.items_type):
            self.n_items_type[i] =  len(df_items.loc[df_items['stackability_code'] == i])
        self.constrs = []
        self.vars = []
    
    def buildModel(self):
        self.generateVars()
        self.generateConstrs()
        self.generateObj()
        self.model.update()
        

    def generateVars(self):
        self.x = self.model.addVars(
                self.n_cols,
                vtype=GRB.INTEGER,  #NOTE: defined as integer and then relax
                lb = 0,
                name=f"X_0"
            )
        for n in range(self.n_cols):
                self.vars.append(self.x[n])
    
    def generateConstrs(self):
        # Adding constraint to the model
        for i in range(self.items_type):
            constr_i = 0
            for n in range(self.n_cols):
                constr_i += self.columns[i, n] * self.x[n]
            self.constrs.append(
                self.model.addConstr(
                    constr_i >= self.n_items_type[i],
                    f"quantity_item{i}"
                )
            )

    def generateObj(self):
        j = 0
        for i in range(len(self.df_vehicles)):
            j = i * N_INIT_COLS
            cost = self.df_vehicles.iloc[i]["cost"]
            for n in range(j, j + N_INIT_COLS):
                self.obj += cost * self.x[n]
        
        self.model.setObjective(self.obj, GRB.MINIMIZE)
    
    def solveRelaxedModel(self, _iter):
        self.model.write(f"./benchmark/results/model_{_iter}.lp")
        self.relaxedModel = self.model.relax()
        self.relaxedModel.optimize()
        
        # Check if model is infeasible
        if self.relaxedModel.Status == 3:
            return 1
        else:
            return 0

    def getDuals(self):
        return self.relaxedModel.getAttr("Pi", self.model.getConstrs())
    
    def addColumns(self, objective, newColumns):
        for i in range(newColumns.shape[1]):
            newColumn = gp.Column(newColumns[:, i], self.model.getConstrs())
            self.model.addVar(vtype = gp.GRB.INTEGER,
                              lb = 0, 
                              obj = objective, 
                              column = newColumn)
        self.model.update()
        self.model.write(f"./benchmark/results/model_X.lp")