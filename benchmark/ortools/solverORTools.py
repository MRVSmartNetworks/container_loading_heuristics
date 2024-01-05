import warnings
from ortools.sat.python import cp_model
from typing import Dict, List, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

DEBUG = True
VERB = False
TIME_LIMIT = 1000  # seconds

# Instance - Need to ensure all elements can fit in the bin, else the solution
# will be infeasible
# container: 0->length, 1->width, 2->max_weight, 3->cost
containers = [(40, 15, 20, 10), (35, 15, 10, 2)]

# Boxes:
#   "item_id":
#       Unique id of the item
#   "dim":
#       [0]: length (x dim)
#       [1]: width (y dim)
#       [2]: weight
boxes = [
    {"item_id": "I0001", "dim": (11, 3, 4), "orientation": "n"},
    {"item_id": "I0002", "dim": (13, 3, 2), "orientation": "l"},
    {"item_id": "I0003", "dim": (9, 2, 4), "orientation": "w"},
    {"item_id": "I0004", "dim": (7, 2, 6), "orientation": "n"},
    {"item_id": "I0005", "dim": (9, 3, 1), "orientation": "l"},
    {"item_id": "I0006", "dim": (7, 3, 3), "orientation": "w"},
    {"item_id": "I0007", "dim": (11, 2, 3), "orientation": "n"},
    {"item_id": "I0008", "dim": (13, 2, 2), "orientation": "n"},
    {"item_id": "I0009", "dim": (11, 4, 7), "orientation": "l"},
    {"item_id": "I0010", "dim": (13, 4, 6), "orientation": "w"},
    {"item_id": "I0010", "dim": (3, 5, 1), "orientation": "w"},
    {"item_id": "I0012", "dim": (11, 2, 2), "orientation": "l"},
    {"item_id": "I0007", "dim": (11, 2, 3), "orientation": "n"},
]


class solverORTools:
    def __init__(self) -> None:
        self.name = "solORTools"
        self.sol = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }

    def solve(
        self,
        items: pd.DataFrame,
        trucks: pd.DataFrame,
        max_truck_n: List[int] = [],
        sol_file_name=None,
    ):
        """
        Solve the problem using OR Tools' CP solver.

        Args:
            items: DataFrame containig one item per row
            trucks: DataFrame of available trucks with their features
            max_truck_n: list containing, for each truck type, the maximum units
            available. This variable is optional (default []), and it is meant
            to speed up the computation by reducing the variable dimensionality.
        """
        self.items = items
        self.trucks = trucks
        n_items = self.items.shape[0]
        n_trucks = self.trucks.shape[0]

        if max_truck_n == []:
            # Default value for the max. allowed number of trucks for each type
            # is n_items (1 item per truck)
            self.max_truck_n = [n_items] * n_trucks
        else:
            if len(max_truck_n) != n_trucks:
                raise ValueError(
                    "The maximum number of trucks per type should be a list with as many elements as trucks"
                )
            self.max_truck_n = max_truck_n

        self.model = cp_model.CpModel()

        # The maximum number of trucks that can be used for each type is given
        # by the elements of 'self.max_truck_n'

        # VARIABLES DEFINITION:
        # t_jk = 1 if truck k of type j is used (k is self.max_truck_n[j])
        t_vars = []
        for j in range(n_trucks):
            t_vars.append(
                [
                    self.model.NewBoolVar(name=f"t_{j},{k}")
                    for k in range(self.max_truck_n[j])
                ]
            )

        c_vars = []
        x_vars = []
        y_vars = []
        x_interval_vars = []
        y_interval_vars = []
        for i in range(n_items):
            c_vars.append([])
            x_vars.append([])
            y_vars.append([])
            x_interval_vars.append([])
            y_interval_vars.append([])
            # df = self.items.iloc[i]["length"]
            # print(self.items.loc[i, ["length"]])
            for j in range(n_trucks):
                # i: item index
                # j: truck (type) index

                # c_ijk: 1 if item i is in k-th truck of type j
                c_vars[i].append(
                    [
                        self.model.NewBoolVar(name=f"c_({i},{j},{k})")
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # x_ijk: x coordinate of the origin of item i in k-th truck j
                x_vars[i].append(
                    [
                        self.model.NewIntVar(
                            0,
                            self.trucks.loc[j, "length"] - self.items.iloc[i]["length"],
                            name=f"x_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # y_ijk: y coordinate of the origin of item i in k-th truck j
                y_vars[i].append(
                    [
                        self.model.NewIntVar(
                            0,
                            self.trucks.loc[j, "width"] - self.items.iloc[i]["width"],
                            name=f"y_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # Interval vars definition (x and y)
                x_interval_vars[i].append(
                    [
                        self.model.NewOptionalIntervalVar(
                            start=x_vars[i][j][k],
                            size=self.items.iloc[i]["length"],
                            end=x_vars[i][j][k] + self.items.iloc[i]["length"],
                            is_present=c_vars[i][j][k],
                            name=f"x_interval_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                y_interval_vars[i].append(
                    [
                        self.model.NewOptionalIntervalVar(
                            start=y_vars[i][j][k],
                            size=self.items.iloc[i]["width"],
                            end=y_vars[i][j][k] + self.items.iloc[i]["width"],
                            is_present=c_vars[i][j][k],
                            name=f"y_interval_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )

        # CONSTRAINTS DEFINITION
        # Each element should appear exactly 1 time (in 1 truck)
        for i in range(n_items):
            self.model.Add(
                sum(
                    [
                        c_vars[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 1
            )

        objective = 0
        for j in range(n_trucks):
            for k in range(self.max_truck_n[j]):
                # Big-M constraint on the number of items in each truck - if the
                # truck is not considered, no item can be placed inside it
                self.model.Add(
                    sum([c_vars[i][j][k] for i in range(n_items)]) <= t_vars[j][k]
                ).OnlyEnforceIf(t_vars[j][k].Not())

                x_interval_vars_jk = [x[j][k] for x in x_interval_vars]
                y_interval_vars_jk = [y[j][k] for y in y_interval_vars]
                self.model.AddNoOverlap2D(x_interval_vars_jk, y_interval_vars_jk)

                # Weight constraint
                self.model.Add(
                    sum(
                        c_vars[i][j][k] * int(self.items.iloc[i]["weight"])
                        for i in range(n_items)
                    )
                    <= self.trucks.loc[j, "max_weight"]
                )

                # OBJECTIVE FUNCTION: total trucks cost
                objective += t_vars[j][k] * self.trucks.loc[j, "cost"]

        self.model.Minimize(obj=objective)
        # Solve!
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = TIME_LIMIT
        self.solver.parameters.log_search_progress = True
        self.solver.log_callback = print
        # Enumerate all solutions.
        # solver.parameters.enumerate_all_solutions = True
        # Solve
        t_start = time.time()
        status = self.solver.Solve(self.model)
        print("+--------------------------------------------+")
        print(f"Time required: {round(time.time() - t_start, 2)} s")
        if status != cp_model.INFEASIBLE and status != cp_model.MODEL_INVALID:
            self.sol_found = True
            if status == cp_model.OPTIMAL:
                print("-> Optimal solution was found!")
            elif status == cp_model.FEASIBLE:
                print("-> Feasible solution found!")
            else:
                warnings.warn("Unknown solution status!")
        else:
            raise RuntimeError("No solution was found!")

        self.obj_val = self.solver.ObjectiveValue()
        print(f"> Objective Value (truck cost): {self.obj_val}")
        print("")

        # Display the solution (each truck k of type j)
        if VERB:
            for j in range(n_trucks):
                for k in range(self.max_truck_n[j]):
                    # Print
                    print(f"Truck {k + 1}, type {j + 1}:")
                    print(
                        f"> Number of items: {sum([self.solver.Value(c_vars[i][j][k]) for i in range(n_items)])}"
                    )
                    curr_tot_weight = sum(
                        [
                            self.items.iloc[i]["weight"]
                            for i in range(n_items)
                            if self.solver.Value(c_vars[i][j][k]) > 0
                        ]
                    )
                    print(f"> Total weight: {curr_tot_weight}")
            print("")

        self.used_trucks_sol = self.printSolution(c_vars, x_vars, y_vars)
        print(f"> {self.used_trucks_sol} trucks have been used")
        print("+--------------------------------------------+")

    def printSolution(self, c_vars, x_vars, y_vars) -> int:
        """
        Print the solution - only displaying the trucks that contain at least
        one item.
        The function returns the number of used trucks.

        Args:
            boxes: list of items (dict)
            trucks: list of truck types (dict)
            c_vars: variable 'c' from the model
            x_vars: variable 'x' from the model
            y_vars: variable 'y' from the model

        Returns:
            Integer number of trucks used (i.e., containing >0 elements); if no
            solution has been found yet, the returned value is -1.
        """
        n_items = self.items.shape[0]
        n_trucks = self.trucks.shape[0]
        if self.sol_found:
            n_used_trucks = 0
            for j in range(n_trucks):
                for k in range(self.max_truck_n[j]):
                    # Check curr. truck contains at least 1 element
                    if sum(
                        [self.solver.Value(c_vars[i][j][k]) for i in range(n_items)]
                    ):
                        n_used_trucks += 1
                        fig, ax = plt.subplots(1)
                        ax.set_xlim(0, self.trucks.loc[j, "length"])
                        ax.set_ylim(0, self.trucks.loc[j, "width"])
                        for i in range(n_items):
                            if self.solver.Value(c_vars[i][j][k]) > 0:
                                ax.add_patch(
                                    patches.Rectangle(
                                        (
                                            self.solver.Value(x_vars[i][j][k]),
                                            self.solver.Value(y_vars[i][j][k]),
                                        ),
                                        self.items.iloc[i]["length"],
                                        self.items.iloc[i]["width"],
                                        facecolor="blue",
                                        alpha=0.2,
                                        edgecolor="b",
                                    )
                                )
                        # uniform axis
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_title(f"Truck {j + 1} number {k + 1}")
                        fig.tight_layout()
                        plt.show()
            return n_used_trucks
        else:
            warnings.warn("No solution was found yet! Please run the model first")
            return -1


if __name__ == "__main__":
    # IDEA: decide beforehand which are the items to be considered among the
    # (long) list of ones that are provided - choose this by solving the
    # knapsack problem on the weight and on dimensions -> 3D knapsack
    df_items = {
        "id_item": [
            "I0001",
            "I0002",
            "I0003",
            "I0004",
            "I0005",
            "I0006",
            "I0007",
            "I0008",
            "I0009",
            "I0010",
        ],
        "length": [11, 13, 9, 7, 9, 7, 9, 7, 3, 10],
        "width": [3, 3, 4, 5, 3, 2, 1, 2, 3, 4],
        "weight": [4, 5, 6, 4, 2, 1, 4, 5, 5, 3],
    }
    df_vehicles = {
        "id_truck": ["V0"],
        "length": [40],
        "width": [15],
        "max_weight": [20],
        "cost": [10],
    }

    df_items = pd.DataFrame.from_dict(df_items)
    df_vehicles = pd.DataFrame.from_dict(df_vehicles)

    truck_loading = solverORTools()
    truck_loading.solve(df_items, df_vehicles)
