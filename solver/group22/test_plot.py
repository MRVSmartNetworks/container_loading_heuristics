from solution_representation import myStack3D
import os
import pandas as pd
import sys

"""
Display solution for specific truck, given ID.

This script accepts the truck ID as a command line 
argument when launching the program.

It is suggested to launch this program from the project root
"""

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "datasetA")

    df_items = pd.read_csv(os.path.join(data_path, "items.csv"))

    df_trucks = pd.read_csv(os.path.join(data_path, "vehicles.csv"))

    try:
        df_sol = pd.read_csv(os.path.join(".", "results", "solver22_sol.csv"))
    except:
        raise ValueError("Solution file not found!")

    if len(sys.argv) > 1:
        truck_id = int(sys.argv[1])
    else:
        truck_id = 100

    myStack3D(df_items, df_trucks, df_sol, truck_id)
