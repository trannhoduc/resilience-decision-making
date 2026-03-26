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
T = 500

MU_W = np.zeros((A.shape[0], 1))   # process-noise mean in Assumption 1
ALPHA_FN = 0.0005                    # temporary, change later
ALPHA_FP = 0.0005                    # temporary, change later
LOOKAHEAD_ELL = 5                  # temporary, change later