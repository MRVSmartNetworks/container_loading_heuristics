import os
import pandas as pd

DATASET = "datasetJ"
N_ITEMS = 50
HEIGHT = 100
RESIZE_WIDTH = 0.5
RESIZE_WEIGHT = 0.2


def create_instance(data_folder, df_items, df_vehicles):
    if not os.path.exists(os.path.join(".", "data", data_folder)):
        os.mkdir(os.path.join(".", "data", data_folder))
    df_items.to_csv(os.path.join("data", data_folder, "items.csv"), index=False)
    df_vehicles.to_csv(os.path.join("data", data_folder, "vehicles.csv"), index=False)


def check_instance(df_items: pd.DataFrame, df_vehicles: pd.DataFrame) -> pd.DataFrame:
    for i_item, item_row in df_items.iterrows():
        for i_vehicle, vehicle_row in df_vehicles.iterrows():
            # Limit item dimension to vehicle dimension if larger
            if item_row["length"] > vehicle_row["length"]:
                item_row["length"] = vehicle_row["length"]
                df_items.at[i_item, "length"] = vehicle_row["length"]
            if item_row["length"] > vehicle_row["width"]:
                item_row["length"] = vehicle_row["width"]
                df_items.at[i_item, "length"] = vehicle_row["width"]
            if item_row["width"] > vehicle_row["length"]:
                item_row["width"] = vehicle_row["length"]
                df_items.at[i_item, "width"] = vehicle_row["length"]
            if item_row["width"] > vehicle_row["width"]:
                item_row["width"] = vehicle_row["width"]
                df_items.at[i_item, "width"] = vehicle_row["width"]
    return df_items


if __name__ == "__main__":
    # Read dataset

    df_items = pd.read_csv(
        os.path.join(".", "data", DATASET, "items.csv"),
    )
    df_vehicles = pd.read_csv(
        os.path.join(".", "data", DATASET, "vehicles.csv"),
    )

    # Take N random items
    df_items = df_items.sample(n=N_ITEMS)

    df_items["nesting_height"] = 0

    # Constant height
    df_items["height"] = HEIGHT
    df_vehicles["height"] = HEIGHT

    # Resize dimension and weight capacity of vehicles
    df_vehicles.loc[:, ["length"]] = df_vehicles.loc[:, ["length"]] * RESIZE_WIDTH
    df_vehicles.loc[:, ["max_weight"]] = (
        df_vehicles.loc[:, ["max_weight"]] * RESIZE_WEIGHT
    )
    df_vehicles.loc[:, ["length", "width", "max_weight"]] = df_vehicles.loc[
        :, ["length", "width", "max_weight"]
    ].astype(int)

    df_items = check_instance(df_items, df_vehicles)

    create_instance(f"MOD{DATASET}", df_items, df_vehicles)
