import pandas as pd


def create_comparison(
    df_sol: pd.DataFrame,
    df_vehicles: pd.DataFrame,
    dataset_name: str,
    solver_name: str,
    t: float,
) -> dict:
    n_vehicles = df_sol["idx_vehicle"].iloc[-1] + 1
    vehicles_type = pd.unique(df_sol["type_vehicle"])
    cost = 0.0
    for _, ele in (
        df_sol.filter(items=["idx_vehicle", "type_vehicle"])
        .drop_duplicates()
        .iterrows()
    ):
        cost += df_vehicles[df_vehicles.id_truck == ele.type_vehicle].iloc[0].cost

    new_comp = {
        "solver": [solver_name],
        "dataset": [dataset_name],
        "n_vehicles": [n_vehicles],
        "vehicles": [vehicles_type],
        "cost": round(cost, 2),
        "time": t,
    }
    return new_comp
