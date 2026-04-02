import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
import orbital_dynamics_with_gyro as ods
import attitude_estimation as att

q_true = ods.q_vals.T 


# k = 0
# Q_true = adcs.Q(q_true[:,k])

# for i in range(m):
#     pred = Q_true.T @ r_N[:,i]
#     meas = ytraj[3*i:3*i+3, k]

#     print("dot:", np.dot(pred, meas))

# SAVE FOR JULIA
np.save("PsycheScripts/xtraj.npy", ods.state.T)   # (16, n) full state
np.save("PsycheScripts/q_true.npy", q_true)
#np.save("PsycheScripts/gyro.npy", gyro)
#np.save("PsycheScripts/ytraj.npy", ytraj)
#np.save("PsycheScripts/r_N.npy", r_N)
#np.save("PsycheScripts/S_list.npy", np.array(S_list))

print("Saved simulation data for Julia.")