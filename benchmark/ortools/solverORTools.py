import os
import time
import warnings
import math
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ortools.sat.python import cp_model

DEBUG = False
VERB = False
TIME_LIMIT = 300  # seconds

# Instance - Need to ensure all elements can fit in the bin, else the solution
# will be infeasible


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
        time_limit=TIME_LIMIT,
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
        t1 = time.time()
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
                    self.model.new_bool_var(name=f"t_{j},{k}")
                    for k in range(self.max_truck_n[j])
                ]
            )

        # Initialize variables
        o_vars = []
        c_vars = []
        x_vars = []
        y_vars = []
        x_interval_vars = []
        y_interval_vars = []
        # Initialize "copies" for rotated items
        c_vars_rot = []
        x_vars_rot = []
        y_vars_rot = []
        x_interval_vars_rot = []
        y_interval_vars_rot = []
        for i in range(n_items):
            # o_i = 0 if item 'i' is placed length-wise, 1 if width-wise
            # (rotated)
            o_vars.append(self.model.new_bool_var(name=f"o_{i}"))

            c_vars.append([])
            x_vars.append([])
            y_vars.append([])
            x_interval_vars.append([])
            y_interval_vars.append([])
            c_vars_rot.append([])
            x_vars_rot.append([])
            y_vars_rot.append([])
            x_interval_vars_rot.append([])
            y_interval_vars_rot.append([])
            # df = self.items.iloc[i]["length"]
            # print(self.items.loc[i, ["length"]])
            for j in range(n_trucks):
                # i: item index
                # j: truck (type) index

                # Original items
                # c_ijk: 1 if item i is in k-th truck of type j
                c_vars[i].append(
                    [
                        self.model.new_bool_var(name=f"c_({i},{j},{k})")
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # x_ijk: x coordinate of the origin of item i in k-th truck j
                x_vars[i].append(
                    [
                        self.model.new_int_var(
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
                        self.model.new_int_var(
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
                        self.model.new_optional_interval_var(
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
                        self.model.new_optional_interval_var(
                            start=y_vars[i][j][k],
                            size=self.items.iloc[i]["width"],
                            end=y_vars[i][j][k] + self.items.iloc[i]["width"],
                            is_present=c_vars[i][j][k],
                            name=f"y_interval_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )

                # Rotated items - used if o_vars[i] == 1
                # c_ijk: 1 if item i is in k-th truck of type j, rotated
                c_vars_rot[i].append(
                    [
                        self.model.new_bool_var(name=f"c_rot_({i},{j},{k})")
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # x_ijk: x coordinate of the origin of item i in k-th truck j if
                # rotated
                x_vars_rot[i].append(
                    [
                        self.model.new_int_var(
                            0,
                            self.trucks.loc[j, "length"] - self.items.iloc[i]["width"],
                            name=f"x_rot_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                # y_ijk: y coordinate of the origin of item i in k-th truck j if
                # rotated
                y_vars_rot[i].append(
                    [
                        self.model.new_int_var(
                            0,
                            self.trucks.loc[j, "width"] - self.items.iloc[i]["length"],
                            name=f"y_rot_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )

                # Interval vars definition (x and y) - rotated items
                x_interval_vars_rot[i].append(
                    [
                        self.model.new_optional_interval_var(
                            start=x_vars_rot[i][j][k],
                            size=self.items.iloc[i]["width"],
                            end=x_vars_rot[i][j][k] + self.items.iloc[i]["width"],
                            is_present=c_vars_rot[i][j][k],
                            name=f"x_interval_rot_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )
                y_interval_vars_rot[i].append(
                    [
                        self.model.new_optional_interval_var(
                            start=y_vars_rot[i][j][k],
                            size=self.items.iloc[i]["length"],
                            end=y_vars_rot[i][j][k] + self.items.iloc[i]["length"],
                            is_present=c_vars_rot[i][j][k],
                            name=f"y_interval_rot_({i},{j},{k})",
                        )
                        for k in range(self.max_truck_n[j])
                    ]
                )

        # CONSTRAINTS DEFINITION
        # Each element should appear exactly 1 time (in 1 truck) either rotated
        # or not
        # NOTE: if c_vars[i j k] == 1, then c_vars_rot[i j k] == 0 and
        # vice-versa

        # Not rotated:
        for i in range(n_items):
            self.model.add(
                sum(
                    [
                        c_vars[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 1
            ).only_enforce_if(o_vars[i].Not())
            self.model.add(
                sum(
                    [
                        c_vars_rot[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 0
            ).only_enforce_if(o_vars[i].Not())

            # Rotated (width-wise):
            self.model.add(
                sum(
                    [
                        c_vars_rot[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 1
            ).only_enforce_if(o_vars[i])
            self.model.add(
                sum(
                    [
                        c_vars[i][j][k]
                        for j in range(n_trucks)
                        for k in range(self.max_truck_n[j])
                    ]
                )
                == 0
            ).only_enforce_if(o_vars[i])

            # self.model.add(
            #     sum(
            #         [
            #             c_vars[i][j][k]
            #             for j in range(n_trucks)
            #             for k in range(self.max_truck_n[j])
            #         ]
            #     )
            #     + sum(
            #         [
            #             c_vars_rot[i][j][k]
            #             for j in range(n_trucks)
            #             for k in range(self.max_truck_n[j])
            #         ]
            #     )
            #     == 1
            # )

            # Link orientation to the forced orientation
            self.model.add(o_vars[i] == 0).only_enforce_if(
                self.items.iloc[i]["forced_orientation"] == "l"
            )
            self.model.add(o_vars[i] == 1).only_enforce_if(
                self.items.iloc[i]["forced_orientation"] == "w"
            )

        objective = 0
        for j in range(n_trucks):
            for k in range(self.max_truck_n[j]):
                # Big-M constraint on the number of items in each truck - if the
                # truck is not considered, no item can be placed inside it
                self.model.add(
                    sum([c_vars[i][j][k] for i in range(n_items)])
                    + sum([c_vars_rot[i][j][k] for i in range(n_items)])
                    <= t_vars[j][k]
                ).only_enforce_if(t_vars[j][k].Not())

                # Overlapping items - orientation should be taken into account
                # by the definition of the interval variables
                x_interval_vars_jk = [x[j][k] for x in x_interval_vars] + [
                    x[j][k] for x in x_interval_vars_rot
                ]
                y_interval_vars_jk = [y[j][k] for y in y_interval_vars] + [
                    y[j][k] for y in y_interval_vars_rot
                ]
                self.model.AddNoOverlap2D(x_interval_vars_jk, y_interval_vars_jk)

                for i in range(n_items):
                    self.model.add(
                        c_vars[i][j][k] * int(np.ceil(self.items.iloc[i]["weight"]))
                        <= self.trucks.loc[j, "max_weight_stack"]
                    )
                for i in range(n_items):
                    self.model.add(
                        c_vars_rot[i][j][k] * int(np.ceil(self.items.iloc[i]["weight"]))
                        <= self.trucks.loc[j, "max_weight_stack"]
                    )
                # Weight constraint
                self.model.add(
                    sum(
                        c_vars[i][j][k] * int(np.ceil(self.items.iloc[i]["weight"]))
                        for i in range(n_items)
                    )
                    + sum(
                        c_vars_rot[i][j][k] * int(np.ceil(self.items.iloc[i]["weight"]))
                        for i in range(n_items)
                    )
                    <= self.trucks.loc[j, "max_weight"]
                )

                # OBJECTIVE FUNCTION: total trucks cost
                objective += t_vars[j][k] * self.trucks.loc[j, "cost"]

        self.model.Minimize(obj=objective)
        # Solve!
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit
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

        var_1 = [c_vars, x_vars, y_vars]
        var_2 = [c_vars_rot, x_vars_rot, y_vars_rot]
        if VERB:
            self.used_trucks_sol = self.printSolution(var_1, var_2)
            print(f"> {self.used_trucks_sol} trucks have been used")
        print("+--------------------------------------------+")

        # Store solution - converted to the usual format
        vars_all = [
            t_vars,
            o_vars,
            c_vars,
            c_vars_rot,
            x_vars,
            x_vars_rot,
            y_vars,
            y_vars_rot,
        ]
        sol_dict = self.assembleSolution(vars_all)

        df_sol = pd.DataFrame.from_dict(sol_dict)
        df_sol.to_csv(os.path.join("results", sol_file_name), index=False)

        t = round(time.time() - t1, 2)
        t_not_used = 0
        return t, t_not_used, self.obj_val

    def printSolution(self, var: List, var_rot: List) -> int:
        """
        Print the solution - only displaying the trucks that contain at least
        one item.
        The function returns the number of used trucks.

        Args:
            var: List with 3 elements:
                c_vars: variable 'c' from the model
                x_vars: variable 'x' from the model
                y_vars: variable 'y' from the model
            var_rot: List with 3 elements:
                c_vars_rot: variable 'c' from the model, for rotated items
                x_vars_rot: variable 'x' from the model, for rotated items
                y_vars_rot: variable 'y' from the model, for rotated items

        Returns:
            Integer number of trucks used (i.e., containing >0 elements); if no
            solution has been found yet, the returned value is -1.
        """
        c_vars = var[0]
        x_vars = var[1]
        y_vars = var[2]
        c_vars_rot = var_rot[0]
        x_vars_rot = var_rot[1]
        y_vars_rot = var_rot[2]
        n_items = self.items.shape[0]
        n_trucks = self.trucks.shape[0]
        if self.sol_found:
            n_used_trucks = 0
            for j in range(n_trucks):
                for k in range(self.max_truck_n[j]):
                    # Check curr. truck contains at least 1 element
                    if sum(
                        [self.solver.Value(c_vars[i][j][k]) for i in range(n_items)]
                    ) or sum(
                        [self.solver.Value(c_vars_rot[i][j][k]) for i in range(n_items)]
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
                            elif self.solver.Value(c_vars_rot[i][j][k]) > 0:
                                ax.add_patch(
                                    patches.Rectangle(
                                        (
                                            self.solver.Value(x_vars_rot[i][j][k]),
                                            self.solver.Value(y_vars_rot[i][j][k]),
                                        ),
                                        self.items.iloc[i]["width"],
                                        self.items.iloc[i]["length"],
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

    def assembleSolution(self, var_list: List) -> Dict:
        """
        Put together a Dict containing the obtained solution, given all the
        variables of the solved problem.

        The dict is both returned and stored as a class attribute.

        This function also performs some basic checks on the solution and used
        items

        Args:
            var_list: list containing the problem variables
        """
        # This function is ugly, but it works (hopefully)
        t_vars = var_list[0]
        o_vars = var_list[1]
        c_vars = var_list[2]
        c_vars_rot = var_list[3]
        x_vars = var_list[4]
        x_vars_rot = var_list[5]
        y_vars = var_list[6]
        y_vars_rot = var_list[7]

        """Attributes:
            type_vehicle,
            idx_vehicle,
            id_stack,
            id_item,
            x_origin,
            y_origin,
            z_origin,
            orient
        """
        # Initialize solution dict
        self.sol_dict = {
            "type_vehicle": [],
            "idx_vehicle": [],
            "id_stack": [],
            "id_item": [],
            "x_origin": [],
            "y_origin": [],
            "z_origin": [],
            "orient": [],
        }

        n_trucks = len(t_vars)
        n_items = len(o_vars)
        trucks_count = 0
        check_items = np.zeros((n_items,))  # Add 1 when item is encountered
        for j in range(n_trucks):
            # Iterate over the truck types and fill the solution
            curr_type = self.trucks.iloc[j]["id_truck"]

            for k in range(len(t_vars[j])):
                # Iterate over the trucks of the same type, looking for a truck
                # with more than 0 elements inside
                if sum(
                    [self.solver.Value(c_vars[i][j][k]) for i in range(n_items)]
                ) + sum(
                    [self.solver.Value(c_vars_rot[i][j][k]) for i in range(n_items)]
                ):
                    for i in range(n_items):
                        if (
                            self.solver.Value(c_vars[i][j][k]) == 1
                            or self.solver.Value(c_vars_rot[i][j][k]) == 1
                        ):
                            check_items[i] += 1

                            assert self.solver.Value(
                                c_vars[i][j][k]
                            ) != self.solver.Value(
                                c_vars_rot[i][j][k]
                            ), "Something is wron with the variables 'c'"

                            self.sol_dict["type_vehicle"].append(curr_type)
                            self.sol_dict["idx_vehicle"].append(trucks_count)
                            # Item ID will be the same as the stack ID
                            # (items == stacks)
                            curr_it_id = self.items.iloc[i]["id_item"]
                            self.sol_dict["id_stack"].append(curr_it_id)
                            self.sol_dict["id_item"].append(curr_it_id)
                            if self.solver.Value(o_vars[i]) == 0:
                                # Not rotated
                                self.sol_dict["x_origin"].append(
                                    self.solver.Value(x_vars[i][j][k])
                                )
                                self.sol_dict["y_origin"].append(
                                    self.solver.Value(y_vars[i][j][k])
                                )
                                self.sol_dict["orient"].append("l")
                            elif self.solver.Value(o_vars[i]) == 1:
                                # Rotated
                                self.sol_dict["x_origin"].append(
                                    self.solver.Value(x_vars_rot[i][j][k])
                                )
                                self.sol_dict["y_origin"].append(
                                    self.solver.Value(y_vars_rot[i][j][k])
                                )
                                self.sol_dict["orient"].append("w")
                            else:
                                raise ValueError("Invalid rotation")

                            self.sol_dict["z_origin"].append(0)

                    trucks_count += 1

        assert all(
            check_items > 0
        ), f"Not all items have been used! {sum(check_items == 0)} unused items"

        assert all(check_items == 1), "Some duplicate exist!"

        return self.sol_dict
