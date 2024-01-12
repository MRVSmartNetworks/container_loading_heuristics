# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import pickle
import gurobipy as grb
from exact.check.check_sol import check_above, check_3D
import numpy as np

### Config:
DEBUG = True
VERB = False
MINUTES = 10  # Maximum optimization time (in minutes)
TIME_LIMIT = 60 * MINUTES


class ExactSolver:
    """
    Solve exact model
    ---
    Constraint summary:
    - Elements fit entirely inside truck
    - No superimposed items (i, j s.t., one of the 2 comes before in at least 1 dim)
    - Elements separated by at least the dimension of the one coming before in
    that direction (no superposition)
    - If item is below another one then it comes before that along direction z
    - Item cannot be 'before' itself
    - Item cannot be 'above' itself
    - Items not in same stack cannot be placed one on top of the other
    - If one item is below the other: same x, y coordinates
    - Less elements above wrt max stackability
    - Orientation
    - Max stack weight
    - Max density (evaluated on each stack)
    """

    def __init__(
        self,
        df_items: pd.DataFrame,
        df_vehicles: pd.DataFrame,
        model_path: str = None,
        pickle_path: str = None,
    ):
        """
        ExactSolver
        ---
        ### inputs:
        - df_items: dataframe containing items
        - df_vehicles: dataframe containing vehicles
        - model_path (default: None): path where to store the '.lp' model
        """
        if VERB:
            print("Items:")
            print(df_items, "\n")

        self.df_items = df_items

        # Define sets
        self.n_items = len(df_items)
        self.items = range(self.n_items)
        self.n_vehicles = len(df_vehicles)
        self.vehicles = range(self.n_vehicles)

        self.orientations = ["widthwise", "lengthwise"]
        self.dimensions_names = ["length", "width", "height"]

        # The model assumes only 1 vehicle was provided (and all items can fit inside it)
        self.vehicle = df_vehicles.iloc[0].to_dict()
        self.df_vehicles = df_vehicles[["length", "width", "height"]].to_numpy()
        if VERB:
            print("Vehicle:")
            print(df_vehicles, "\n")

        dimensions_names = ["length", "width", "height"]

        w = df_items["width"].to_numpy()
        l = df_items["length"].to_numpy()
        h = df_items["height"].to_numpy()

        # Check for possible issues (e.g., missing data):
        assert len(w) == len(l) and len(l) == len(
            h
        ), "The items are missing data (dimensions don't match)"

        # Check: volume, surfaces, dimensions, weight
        # Tot volume - NOTE: the nesting height is not considered
        tot_item_vol = np.sum(w * l * h)
        assert tot_item_vol <= (
            self.vehicle["length"] * self.vehicle["width"] * self.vehicle["height"]
        ), "The item volume exceeds the volume of the truck"

        # Ensure surface is lower than total surface (FIXME - may be removed because of next one)
        all_item_surf = (
            w * l
        )  # Array of surfaces of all items (XY) - check each is lower than truck area
        assert all(
            all_item_surf < (self.df_vehicles[0, 0] * self.df_vehicles[0, 1])
        ), "Some items have bigger surface than the truck"

        # Ensure all dimensions are smaller than the the corresponding truck
        # dimensions and, if the item can be rotated, of the opposite ones
        for i in self.items:
            if df_items.iloc[i]["forced_orientation"] == "lengthwise":
                # If lengthwise, i.e., no rotation:
                assert (
                    w[i] <= self.vehicle["width"] and l[i] <= self.vehicle["length"]
                ), f"Item {i} cannot be placed in the truck"
            elif df_items.iloc[i]["forced_orientation"] == "widthwise":
                # If widthwise, i.e., rotation:
                assert (
                    w[i] <= self.vehicle["length"] and l[i] <= self.vehicle["width"]
                ), f"Item {i} cannot be placed in the truck"
            else:
                # If no imposed orientation, check item can fit either rotated or not
                assert (
                    w[i] <= self.vehicle["width"] and l[i] <= self.vehicle["length"]
                ) or (
                    w[i] <= self.vehicle["length"] and l[i] <= self.vehicle["width"]
                ), f"Item {i} cannot be placed in the truck"

        # Create a new model
        self.model = grb.Model("EasyModel")

        self.has_constraints = False  # true if the constraints have been initialized
        # Set output folder for the model
        if model_path is None:
            script_folder = os.path.dirname(__file__)
        else:
            script_folder = model_path
        self.model_folder = os.path.join(
            script_folder, "..", "logs"
        )  # The logs folder is 1 level above the 'exact' folder
        if not os.path.exists(self.model_folder):
            # Create folder if it does not exist
            os.makedirs(self.model_folder)
        self.model_path = os.path.join(self.model_folder, "model.lp")

        self.path_sol = pickle_path

        # Time for solution
        self.comp_time = None

        """ Define problem variables """
        # V_ij: 1 if item i is "immediately" below item j
        self.V = self.model.addVars(
            self.n_items, self.n_items, vtype=grb.GRB.BINARY, name="V"
        )

        # W_ij: 1 if item i is below item j
        # FIXME: not needed - redundant with B_ij2
        # W = self.model.addVars(n_items, n_items, vtype=grb.GRB.BINARY, name="W")

        # B_ijd: 1 if item i comes before item j on dimension d
        self.B = self.model.addVars(
            self.n_items, self.n_items, 3, vtype=grb.GRB.BINARY, name="B"
        )

        # R_i: 1 if item i is rotatated widthwise
        self.R = self.model.addVars(self.n_items, vtype=grb.GRB.BINARY, name="R")

        # X_d: coordinate (bottom) left point (as defined in project)
        self.X = self.model.addVars(self.n_items, 3, vtype=grb.GRB.CONTINUOUS, name="X")

        self.obj = grb.quicksum(self.X[i, d] for i in self.items for d in range(3))
        # NB: il modello con obj = 1 è più veloce, con 50 items però non trova sol feasible.
        # NB: this also imply a preference from (0,0) on
        self.model.setObjective(self.obj, grb.GRB.MINIMIZE)

    def defModel(self):
        """
        defModel
        ---
        Add constraints to Gurobi model.

        This method was created to isolate the definition of the constraints from the rest.
        """

        """ Constraints definition """
        for i in self.items:
            # Element i fits entirely along truck length
            # (Rotation determines which dimension is along x)
            size = (
                self.df_items.iloc[i][self.dimensions_names[0]] * (1 - self.R[i])
                + self.df_items.iloc[i][self.dimensions_names[1]] * self.R[i]
            )
            self.model.addConstr(
                (self.X[i, 0] + size <= self.vehicle[self.dimensions_names[0]]),
                f"lenght I{i}]vehicle_size_limit",
            )
            # Element i fits entirely along truck width
            size = (
                self.df_items.iloc[i][self.dimensions_names[1]] * (1 - self.R[i])
                + self.df_items.iloc[i][self.dimensions_names[0]] * self.R[i]
            )
            self.model.addConstr(
                (self.X[i, 1] + size <= self.vehicle[self.dimensions_names[1]]),
                f"width I{i}]vehicle_size_limit",
            )
            # Element i fits entirely along truck height
            self.model.addConstr(
                (
                    self.X[i, 2]
                    <= self.vehicle[self.dimensions_names[2]]
                    - self.df_items.iloc[i][self.dimensions_names[2]]
                ),
                f"height I{i}]vehicle_size_limit_height",
            )

        # Prevent 2 items from having the same coordinates
        self.model.addConstrs(
            (
                grb.quicksum(self.B[i, j, d] + self.B[j, i, d] for d in range(3)) >= 1
                for i in self.items
                for j in range(i + 1, self.n_items)
            ),
            "i_before_j_or_reverse_in_one_dim",
        )

        # LENGHT AND WIDTH - items need to be separated by (at least) the
        # len/width of the one which comes before:
        for i in self.items:
            for j in self.items:
                if i != j:
                    # For any couple of items
                    size = (
                        self.df_items.iloc[i][self.dimensions_names[0]]
                        * (1 - self.R[i])
                        + self.df_items.iloc[i][self.dimensions_names[1]] * self.R[i]
                    )
                    self.model.addConstr(
                        self.X[j, 0]
                        >= self.X[i, 0] + size - 10000 * (1 - self.B[i, j, 0]),
                        f"{self.dimensions_names[0]}-{i}-{j}] link_x_b",
                    )
                    size = (
                        self.df_items.iloc[i][self.dimensions_names[1]]
                        * (1 - self.R[i])
                        + self.df_items.iloc[i][self.dimensions_names[0]] * self.R[i]
                    )
                    self.model.addConstr(
                        self.X[j, 1]
                        >= self.X[i, 1] + size - 10000 * (1 - self.B[i, j, 1]),
                        f"{self.dimensions_names[1]}-{i}-{j}] link_x_b",
                    )
        # HEIGHT - Ensure separation, accounting for nesting height
        self.model.addConstrs(
            (
                self.X[j, 2]
                >= self.X[i, 2]
                + self.df_items.iloc[i][self.dimensions_names[2]]
                - self.df_items.iloc[i]["nesting_height"]
                - 10000 * (1 - self.V[i, j])
                for i in self.items
                for j in self.items
                if i != j
            ),
            f"{self.dimensions_names[2]}-{i}-{j}] link_x_v",
        )
        # Link constraints on V to ones on B
        self.model.addConstrs(
            (
                self.B[i, j, 2] <= self.V[i, j]
                for i in self.items
                for j in self.items
                if i != j
            ),
            "link_b_v",  # Updated!!!!
        )

        # Item cannot be 'before' itself
        self.model.addConstrs(
            (self.B[i, i, d] == 0 for i in self.items for d in range(3)), "null_B_ii"
        )
        # Same, but for V
        self.model.addConstrs((self.V[i, i] == 0 for i in self.items), "null_V_ii")

        # Stackability code:
        for i in self.items:
            for j in range(i + 1, self.n_items):
                if (
                    self.df_items.iloc[i]["stackability_code"]
                    != self.df_items.iloc[j]["stackability_code"]
                ):
                    self.model.addConstr(self.V[i, j] == 0, name="no_stack_v_1")
                    # self.model.addConstr(B[i, j, 2] == 0, name="no_stack_b_1")     # Review
                    self.model.addConstr(self.V[j, i] == 0, name="no_stack_v_2")
                    # self.model.addConstr(B[j, i, 2] == 0, name="no_stack_b_2")     # Review

        # If V_ij = 1 same X (x and y)
        # What happens when V_ij != 1?
        for i in self.items:
            for j in self.items:
                if i != j:
                    for d in range(2):  # x and y only!
                        # Suggested correction: maybe instead of 'height': dimension_names[d]
                        # or vehicle['height'] is just big M
                        self.model.addConstr(
                            self.X[i, d] - self.X[j, d]
                            <= self.vehicle[self.dimensions_names[d]]
                            * (1 - self.V[i, j]),
                            name=f"{self.dimensions_names[d]}]{i}-{j}_same_X_M",
                        )
                        self.model.addConstr(
                            self.X[i, d] - self.X[j, d]
                            >= -self.vehicle[self.dimensions_names[d]]
                            * (1 - self.V[i, j]),
                            name=f"{self.dimensions_names[d]}]{i}-{j}_same_X_m",
                        )
        self.model.addConstrs(
            (
                grb.quicksum(self.B[i, j, 2] for j in self.items)
                <= self.df_items.iloc[i]["max_stackability"] - 1
                for i in self.items
            ),
            name="max_stackability",
        )
        # TODO: CHECK BELOW: [fixed]
        # ROTATIONS:
        for i in self.items:
            if self.df_items.iloc[i]["forced_orientation"] == "widthwise":  # Rotated
                self.model.addConstr(self.R[i] == 1, name=f"I{i}]rotation_widthwise")
            elif self.df_items.iloc[i]["forced_orientation"] == "lenghtwise":
                self.model.addConstr(self.R[i] == 0, name=f"I{i}]rotation_lenghtwise")

        self.model.addConstrs(
            (
                self.df_items.iloc[i]["weight"]
                + grb.quicksum(
                    self.df_items.iloc[j]["weight"] * self.B[i, j, 2]
                    for j in self.items
                )
                <= self.vehicle["max_weight_stack"]
                for i in self.items
            ),
            name="stack_weight_limit",
        )
        # MAX density:
        self.model.addConstrs(
            (
                self.df_items.iloc[i]["weight"]
                + grb.quicksum(
                    self.df_items.iloc[j]["weight"] * self.B[i, j, 2]
                    for j in self.items
                )
                <= self.vehicle["max_density"]
                * self.df_items.iloc[i][self.dimensions_names[0]]
                * self.df_items.iloc[i][self.dimensions_names[1]]
                for i in self.items
            ),
            name="stack_weight_limit",
        )
        self.model.update()

        # Store the updated model
        self.model.write(self.model_path)
        self.has_constraints = True

    def solve(self, time_limit_s: float = TIME_LIMIT, plots: bool = True):
        """
        solve
        ---
        Solve the exact model using Gurobi.

        It is possible to specify a time limit (defaults to TIME_LIMIT)
        """
        if not self.has_constraints:
            self.defModel()

        # Set TIME LIMIT:
        self.model.setParam("TimeLimit", TIME_LIMIT)
        # -----
        start = time.time()
        self.model.optimize()
        end = time.time()
        self.comp_time = end - start
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Time for solution: {self.comp_time} seconds")
        print(
            "------------------------------------------------------------------------"
        )
        print(
            f"vehicle dimensions: {[self.vehicle[self.dimensions_names[d]] for d in range(3)]}"
        )
        for i in self.items:
            # Print:
            # - Item index
            # - Coordinates of bottom-left vertex
            # - Dimensions of the item
            print(
                f"{i}]{self.X[i,0].X}\t{self.X[i,1].X}\t{self.X[i,2].X} -- {[self.df_items.iloc[i][self.dimensions_names[d]] for d in range(3)]}"
            )
            upper_corner = [
                self.X[i, d].X + self.df_items.iloc[i][self.dimensions_names[d]]
                for d in range(3)
            ]
            print(f"  {upper_corner[0]}\t{upper_corner[1]}\t{upper_corner[2]}")
            print(f"Rotation: {self.R[i].X}")
            # Print variable B - indicate relative position of item 'i' wrt any other
            # along Y direction (1)
            # for j in items:
            #     print(f"B[{i},{j}, 1] = {B[i, j, 1].X}")
            # print("")
        for d in [2]:  # range(3):
            for i in self.items:
                for j in self.items:
                    if self.B[i, j, d].X > 0.5:
                        print(f"B[{i}, {j}, {d}]")
        print("----")
        for i in self.items:
            for j in self.items:
                if self.V[i, j].X > 0.5:
                    print(f"V[{i}, {j}]")
        print("----")
        for i in self.items:
            if self.R[i].X > 0.5:
                print(f"R[{i}]")
        print(
            "------------------------------------------------------------------------"
        )

        # Plotting the solution
        sizes = np.zeros((self.n_items, 3))
        x_o = np.zeros(self.n_items)
        y_o = np.zeros(self.n_items)
        x_e = np.zeros(self.n_items)
        y_e = np.zeros(self.n_items)
        coordinates = np.zeros((self.n_items, 3))
        for i in range(self.n_items):
            for d in range(2):
                sizes[i, d] = (
                    self.df_items.iloc[i][self.dimensions_names[d]] * (1 - self.R[i].X)
                    + self.df_items.iloc[i][self.dimensions_names[1 - d]] * self.R[i].X
                )

                coordinates[i, d] = self.X[i, d].X

            coordinates[i, 2] = self.X[i, 2].X
            sizes[i, 2] = self.df_items.iloc[i][self.dimensions_names[2]]

            x_o[i] = self.X[i, 0].X
            y_o[i] = self.X[i, 1].X

            x_e[i] = x_o[i] + sizes[i, 0]
            y_e[i] = y_o[i] + sizes[i, 1]

        # Save the information needed for plotting the solution using pickle
        if self.path_sol is not None:
            exact_solution = {
                "coordinates": coordinates,
                "sizes": sizes,
                "n_items": self.n_items,
                "df_vehicles": self.df_vehicles,
            }
            pickle.dump(exact_solution, self.path_sol)

        check_3D(
            coordinates=coordinates,
            sizes=sizes,
            n_items=self.n_items,
            df_vehicles=self.df_vehicles,
        )


if __name__ == "__main__":
    dataset = "test_8"

    df_items = pd.read_csv(
        os.path.join("paper-tests", "test_", dataset, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join("paper-tests", "test_", dataset, "vehicles.csv"),
    )

    solver = ExactSolver(df_items, df_vehicles)
    solver.solve()
