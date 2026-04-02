import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
import attitude_estimation as att

# -----------------------------
# LOAD JULIA OUTPUT
# -----------------------------
xfilt = np.load("PsycheScripts/xfilt.npy")   # (4,n)
P = np.load("PsycheScripts/P.npy")           # (3,3,n)
xtraj = np.load("PsycheScripts/xtraj.npy")
ytraj = np.load("PsycheScripts/ytraj.npy")
r_N = np.load("PsycheScripts/r_N.npy")
q_true = np.load("PsycheScripts/q_true.npy")

dt = 0.1
n = q_true.shape[1]
m = r_N.shape[1]   # ✅ FIX: no hardcoding

for k in range(n):
    if np.dot(xfilt[:,k], q_true[:,k]) < 0:
        xfilt[:,k] *= -1

print("xfilt shape:", xfilt.shape)
print("q_true shape:", q_true.shape)
print("P shape:", P.shape)
print("ytraj shape:", ytraj.shape)
print("r_N shape:", r_N.shape)

# -----------------------------
# ERROR METRIC
# -----------------------------
def attitude_error(q1, q2):
    Q1 = adcs.Q(q1)
    Q2 = adcs.Q(q2)

    dR = Q1.T @ Q2
    angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1, 1))

    return np.degrees(angle)

error = np.array([
    attitude_error(xfilt[:,k], q_true[:,k])
    for k in range(n)
])

# -----------------------------
# CONSISTENCY (better version)
# -----------------------------
# Small-angle approximation: P ≈ covariance of angle error
sigma = np.sqrt([np.trace(P[:,:,k]) for k in range(n)])
sigma_deg = np.degrees(sigma)

# -----------------------------
# STATIC ESTIMATOR (Wahba)
# -----------------------------
static_error = []

for k in range(n):
    # ✅ FIX: reshape using correct m
    yk = ytraj[:,k].reshape(3, m, order='F')
    for i in range(m):
        print("norm:", np.linalg.norm(yk[:,i]))

    Q_est, q_est = att.wahba_svd(yk, r_N)

    static_error.append(
        attitude_error(q_est, q_true[:,k])   # ✅ FIX: consistent metric
    )

static_error = np.array(static_error)
print(static_error)

# -----------------------------
# TIME
# -----------------------------
t = np.arange(n)*dt

# -----------------------------
# PLOTS
# -----------------------------

# 1. Quaternion comparison (cleaner)
plt.figure()
labels = ['q0','q1','q2','q3']
for i in range(4):
    plt.plot(t, q_true[i,:], label=f'{labels[i]} true')
    plt.plot(t, xfilt[i,:], '--', label=f'{labels[i]} est')
plt.legend()
plt.title('Quaternion Comparison')
plt.xlabel("Time [s]")
plt.grid()

# 2. MEKF error
plt.figure()
plt.plot(t, error)
plt.xlabel("Time [s]")
plt.ylabel("Error [deg]")
plt.title("MEKF Attitude Error")
plt.grid()

# 3. Static vs MEKF
plt.figure()
plt.plot(t, error, label="MEKF")
plt.plot(t, static_error, label="Static (Wahba)")
plt.xlabel("Time [s]")
plt.ylabel("Error [deg]")
plt.ylim([0,10])
plt.legend()
plt.title("MEKF vs Static Estimator")
plt.grid()

# 4. Consistency
plt.figure()
plt.plot(t, error, label="Error")
plt.plot(t, sigma_deg, label="sqrt(trace(P))")
plt.xlabel("Time [s]")
plt.ylabel("Degrees")
plt.legend()
plt.title("Consistency Check")
plt.grid()

plt.show()