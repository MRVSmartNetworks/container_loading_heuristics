import numpy as np
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Extract passed file - the argument is the file name
        fname = sys.argv[1]
        ds = fname.split(".")[-2].split("_")[-1]
        try:
            with open(fname, "r") as f:
                # Store all numbers in an NDarray
                val = [float(line.rstrip()) for line in f]
                values = np.array(val)
                f.close()
        except:
            # May specify the path as relative to the script
            fname1 = os.path.join(os.path.dirname(__file__), fname)
            with open(fname1, "r") as f:
                # Store all numbers in an NDarray
                val = [float(line.rstrip()) for line in f]
                values = np.array(val)
                f.close()

    else:
        raise ValueError("Missing file name as command line argument!")

    # Mean
    m = np.mean(values)

    # Stdev
    std = np.std(values)

    print(f"Dataset: {ds}\n> Mean time: {m} s\n> Standard Deviation: {std} s")
