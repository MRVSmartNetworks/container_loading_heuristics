"""
This is the configuration file for solver 22 including all variables controlling 
different properties of the code.
"""

# VERB and MORE_VERB are used to print more information during runs
VERB = False
MORE_VERB = False
# STATS prints scores of 2D solutions
STATS = False

# GUROBI is used to choose whether to use Gurobi to create stacks or not
GUROBI = True

# Flag for activating solution improvement
IMPROVE = False

# Fraction (percentage) of deleted trucks when improving solution
FRAC_DELETED_TRUCKS = 35  # %

# N_DEBUG is used to decide whether to execute assertions (much faster solution without)
N_DEBUG = False

MAX_ITER = 10000  # Maximum number of iterations for filling each truck
MAX_TRIES = 3  # Maximum number of full solutions evaluated
