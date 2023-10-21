import os

"""
This is the configuration file for solver 22 including all variables controlling 
different properties of the code.
"""
# Random seed
SEED = 315054

# VERB and MORE_VERB are used to print more information during runs
VERB = False
MORE_VERB = False
# STATS prints scores of 2D solutions
STATS = False

# Print graphs (in Solver22)
PLOTS = False

# GUROBI is used to choose whether to use Gurobi to create stacks or not
GUROBI = True

# Flag for activating solution improvement
IMPROVE = False

# Fraction (percentage) of deleted trucks when improving solution
FRAC_DELETED_TRUCKS = 35  # %

# N_DEBUG is used to decide whether to execute assertions (much faster solution without)
N_DEBUG = False

MAX_ITER = 10000  # Maximum number of iterations for filling each truck
MAX_TRIES = 2  # Maximum number of full solutions evaluated

# Output file with solution#
# OUT_SOL = os.path.join("results", "solver22_sol_I.csv")

# File for storing runtimes
STORE_TIMES = False
OUT_TIME = os.path.join(
    ".", "solver", "group22", "my_stuff", "runtimes", "runtimes.txt"
)
