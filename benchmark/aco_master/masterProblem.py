import numpy as np
import gurobipy as gp
from gurobipy import GRB


class MasterProblem:
    """
    MasterProblem
    ---------------------------------------------------
    Class used to define the master problem used in the
    implementation of the column generation.
    """

    def __init__(self, patternList, df_vehicles, df_items):
        self.model = gp.Model("MasterProblem")
        self.obj = 0
        self.patternlist = patternList
        self.items_type = len(patternList[0]["pattern"])
        self.columns = np.zeros([self.items_type, len(patternList)])
        for i, pattern in enumerate(patternList):
            self.columns[:, i] = pattern["pattern"]

        self.df_vehicles = df_vehicles
        self.n_cols = self.columns.shape[1]

        # Evaluate the number of items per type in df_items
        self.n_items_type = np.zeros(self.items_type)
        for i in range(self.items_type):
            self.n_items_type[i] = len(df_items.loc[df_items["stackability_code"] == i])
        self.constrs = []
        self.vars = []

        # Build the model
        self.buildModel()

    def buildModel(self):
        self.generateVars()
        self.generateConstrs()
        self.generateObj()
        self.model.update()

    def generateVars(self):
        self.x = self.model.addVars(
            self.n_cols,
            vtype=GRB.INTEGER,  # NOTE: defined as integer and then relax
            lb=0,
            name=f"X_0",
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
                    constr_i >= self.n_items_type[i], f"quantity_item{i}"
                )
            )

    def generateObj(self):
        for n in range(self.n_cols):
            id_truck = self.patternlist[n]["vehicle"]
            cost = self.df_vehicles[self.df_vehicles["id_truck"] == id_truck][
                "cost"
            ].iloc[0]
            self.obj += cost * self.x[n]

        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def solveRelaxedModel(self, _iter):
        self.relaxedModel = self.model.relax()
        self.relaxedModel.Params.LogToConsole = 0
        self.relaxedModel.optimize()

        # Check if model is infeasible
        if self.relaxedModel.Status == 3:
            print(f"Iteration {_iter}: Model infeasible")
            return None
        else:
            print(
                f"Iteration {_iter}: Objective value {round(self.relaxedModel.objVal, 2)}"
            )
            return self.relaxedModel.objVal

    def getDuals(self):
        return self.relaxedModel.getAttr("Pi", self.model.getConstrs())

    def addColumns(self, objective, newColumns):
        for i in range(newColumns.shape[1]):
            newColumn = gp.Column(newColumns[:, i], self.model.getConstrs())
            self.model.addVar(
                vtype=gp.GRB.INTEGER, lb=0, obj=objective, column=newColumn
            )
        self.model.update()

    def solveModel(self, file_name=None):
        self.model.optimize()
        if file_name:
            self.model.write(file_name)

    def getVars(self):
        return self.model.getVars()
