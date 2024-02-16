# Boolean to print every 2D vehicle plot
SINGLE_PLOT = False

# Number of iteration for the all solution
# set it to 1 for a single solution
N_SOL_ITER = 1

# Parameter used to obtain bets alpha and beta
TEST_PAR = False
PARAM_ITER = 1
PARAM_FILE = "solver\group23\parameters_file\datasetF"

# To print information related to area and weight ratio of trucks
PRINT = False

# If true select the ACO version with slices
SLICES = True

# Set number of weight clusters (set 1 for avoid clustering)
N_WEIGHT_CLUSTERS = 1
# Ant colony optimization parameters
ALPHA = 1
BETA = 1
N_ANTS = 90
N_ITER = 15
EVAPOR_COEFF = 0.2

# If true the solution created consist of one layer only (no stack of item N_item in a stack = 1)
ONE_LAYER = False
# Performance parameters
# AREA_RATIO is the parameters that choose when a solution is considered a good one.
# when the best area_ratio of the ants exceed the one of AREA_RATIO then the solution is cosidered
# good enough and the truck is completed.
# AREA_RATIO = 0.9 is the usual value setted for a good trade-off between performance and time
# AREA_RATIO = 1 is to obtain the maximum performance obtainable from the program (advice: 0.93 obtain likely the maximum performance with much faster solutions)
# (pay attention that also N_ANTS and N_ITER influences the performance of the program, both in time and objective value)
AREA_RATIO = 0.93
WEIGHT_RATIO = 0.98

VERB_LOCAL = True
DEBUG_LOCAL = False
ACO_ITERATIONS = False
SUMMARY = True
