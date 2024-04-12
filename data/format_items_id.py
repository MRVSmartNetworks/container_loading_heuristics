import sys
import pandas as pd

folder_name = sys.argv[1]
df_items = pd.read_csv(f"{folder_name}/items.csv")

for i in range(len(df_items)):
    df_items.at[i, "id_item"] = f"I" + str(i)

df_items.to_csv(f"{folder_name}/items.csv", index=False)
