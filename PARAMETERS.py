import numpy as np

# Linear Time Invariant
A = np.array([[0.0, 1.0],
              [-0.9, 1.8]])
C = np.array([[0.5, 1.0]])          # 1x2
Q = np.array([[0.0, 0.0],
              [0.0, 1.0]])
R = np.array([
    [0.1]
])

c = np.array([
    [1.0],
    [0.0]
])

dt = 1.0

Delta = 5.0
epsilon = 0.0
T = 1000

MU_W = np.zeros((A.shape[0], 1))   # process-noise mean in Assumption 1
ALPHA_FN = 0.1                   # temporary, change later
ALPHA_FP = 0.1                   # temporary, change later
LOOKAHEAD_ELL = 10                  # temporary, change later

XI_MODE = "optimal"   # "optimal", "delta", or "manual"
XI_VALUE = None       # only used if XI_MODE == "manual"

XI_WEIGHT_FP = 1.0
XI_WEIGHT_FN = 1.0

# Block length information

BLOCKLENGTH_N = 128
INFO_BITS_L = 64
CHANNEL_NOISE_VAR = 1.0

PT_MIN = 0.05
PT_MAX = 50
RHO = 50.0

P_R = 0.01
EPSILON_L = 0.05
TH_RECOVERY_LAMBDA = 4.0
TH_RECOVERY_KAPPA = 2.0

THETA0_CANDIDATES = list(range(1, 18))
THETA1_CANDIDATES = list(range(1, 6))

NUM_I_MONTE_CARLO = 3000
I_MONTE_CARLO_SEED = 42
I_MONTE_CARLO_BURN_IN = 1000
I_MONTE_CARLO_MAX_STEPS = 500000

AVERAGE_EPSILON_METHOD = "closed_form"
#AVERAGE_EPSILON_METHOD = "monte_carlo"
#AVERAGE_EPSILON_MC_SAMPLES = 20000

DEBUG_BURN_IN = 100
DEBUG_NUM_TRANSITIONS_PER_STATE = 500
DEBUG_MAX_STEPS = 1000
DEBUG_SEED = 42
DEBUG_VERBOSE_STEPS = 50
DEBUG_OUT_PATH = "sensor_predictive_horizon_debug.png"