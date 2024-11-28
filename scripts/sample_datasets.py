import pandas as pd
import os
import random

rootDir = "data"
for subdir, dirs, files in os.walk(rootDir):
    if subdir.startswith(rootDir + "/dataset"):
        newDatasetsDir = [subdir + str(i + 1) for i in range(5)]
        items = pd.read_csv(subdir + "/items.csv")
        vehicles = pd.read_csv(subdir + "/vehicles.csv")
        for newDir in newDatasetsDir:
            if not os.path.exists(newDir):
                os.makedirs(newDir)
                if len(items) >= 2000:
                    if len(items) > 4000:
                        n = 3000
                    else:
                        n = 2000
                    sampledItems = items.sample(n=n)
                    sampledItems.to_csv(newDir + "/items.csv")
                    vehicles.to_csv(newDir + "/vehicles.csv")
                    print("New dataset in", newDir)



