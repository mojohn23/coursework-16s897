import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
import orbital_dynamics_with_gyro as ods
import attitude_estimation as att

q_true = ods.q_vals.T 

# SAVE FOR JULIA
np.save("PsycheScripts/xtraj.npy", ods.state.T)   # (16, n) full state
np.save("PsycheScripts/q_true.npy", q_true)

print("Saved simulation data for Julia.")