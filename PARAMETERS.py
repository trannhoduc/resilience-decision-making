import numpy as np

# =============================================================================
# LINEAR TIME INVARIANT (LTI) SYSTEM
# =============================================================================

A = np.array([[0.0, 1.0],
              [-0.9, 1.8]])             # State transition matrix (2x2)
C = np.array([[0.5, 1.0]])             # Observation matrix (1x2)
Q = np.array([[0.0, 0.0],
              [0.0, 1.0]])             # Process noise covariance
R = np.array([[0.1]])                  # Measurement noise covariance

c = np.array([[1.0],
              [0.0]])                  # Input vector

dt = 1.0                               # Sampling time

# =============================================================================
# SIMULATION
# =============================================================================

T       = 1000                         # Total simulation steps
Delta   = 5.0                          # Disruption magnitude
# epsilon = 0.01                          # Perturbation parameter

MU_W = np.zeros((A.shape[0], 1))       # Process-noise mean (Assumption 1)

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

ALPHA_FN       = 0.1                   # False-negative rate (temporary)
ALPHA_FP       = 0.1                 # False-positive rate (temporary)
LOOKAHEAD_ELL  = 10                    # Lookahead horizon (temporary)

# =============================================================================
# XI (DECISION THRESHOLD) CONFIGURATION
# =============================================================================

XI_MODE      = "optimal"               # "optimal", "delta", or "manual"
XI_VALUE     = None                    # Used only when XI_MODE == "manual"

XI_WEIGHT_FP = 1.0                     # Penalty weight for false positives
XI_WEIGHT_FN = 1.0                     # Penalty weight for false negatives

# =============================================================================
# CHANNEL / BLOCK CODE
# =============================================================================

BLOCKLENGTH_N      = 128               # Block length n
INFO_BITS_L        = 256                # Information bits L
CHANNEL_NOISE_VAR  = 1.0               # Channel noise variance

PT_MIN = 0.05                          # Minimum transmission power
PT_MAX = 2000                            # Maximum transmission power
RHO    = 50.0                          # Power budget / SNR scaling

BW = 1e6                               # Bandwitch
T_SYM = 1/BW                           # Symbol duration
P_T = 20

# =============================================================================
# RECOVERY MODEL
# =============================================================================

P_R               = 0.001              # Recovery probability per step
EPSILON_L         = 0.1             # Loss probability threshold
TH_RECOVERY_LAMBDA = 4.0              # Recovery threshold lambda
TH_RECOVERY_KAPPA  = 2.0              # Recovery threshold kappa

# =============================================================================
# THETA CANDIDATE SETS
# =============================================================================

THETA0_CANDIDATES = list(range(1, 20)) # Candidate values for theta_0
THETA1_CANDIDATES = list(range(1, 10))  # Candidate values for theta_1

# =============================================================================
# MONTE CARLO (INVARIANT DISTRIBUTION)
# =============================================================================

NUM_I_MONTE_CARLO       = 100          # Number of Monte Carlo runs
I_MONTE_CARLO_SEED      = 42           # Random seed
I_MONTE_CARLO_BURN_IN   = 100          # Burn-in steps
I_MONTE_CARLO_MAX_STEPS = 10000        # Max steps per run

AVERAGE_EPSILON_METHOD  = "closed_form"
# AVERAGE_EPSILON_METHOD  = "monte_carlo"
# AVERAGE_EPSILON_MC_SAMPLES = 20000

# =============================================================================
# DEBUG
# =============================================================================

DEBUG_BURN_IN                  = 100
DEBUG_NUM_TRANSITIONS_PER_STATE = 100
DEBUG_MAX_STEPS                = 100
DEBUG_SEED                     = 42
DEBUG_VERBOSE_STEPS            = 50
DEBUG_OUT_PATH                 = "sensor_predictive_horizon_debug.png"
