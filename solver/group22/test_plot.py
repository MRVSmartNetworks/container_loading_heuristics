from solution_representation import myStack3D
import os
import pandas as pd
import sys

### NOTE: to be ran from

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    df_items = pd.read_csv(os.path.join(data_path, "items.csv"))

    df_trucks = pd.read_csv(os.path.join(data_path, "vehicles.csv"))

    try:
        df_sol = pd.read_csv(os.path.join(".", "results", "solver22_sol.csv"))
    except:
        raise ValueError("Solution file not found!")

    if len(sys.argv) > 1:
        truck_id = str(sys.argv[1])
    else:
        truck_id = "V1_001"

    myStack3D(df_items, df_trucks, df_sol, truck_id)
