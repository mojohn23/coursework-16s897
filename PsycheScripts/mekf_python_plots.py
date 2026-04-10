import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
import attitude_estimation as att

# load Julia output
xfilt = np.load("PsycheScripts/xfilt.npy")  
P = np.load("PsycheScripts/P.npy")          
xtraj = np.load("PsycheScripts/xtraj.npy")
ytraj = np.load("PsycheScripts/ytraj.npy")
r_N = np.load("PsycheScripts/r_N.npy")
q_true = np.load("PsycheScripts/q_true.npy")

dt = 0.1
n = q_true.shape[1]
m = r_N.shape[1] 

for k in range(n):
    if np.dot(xfilt[:,k], q_true[:,k]) < 0:
        xfilt[:,k] *= -1

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

# consistency metric
sigma = np.sqrt([np.trace(P[:,:,k]) for k in range(n)])
sigma_deg = np.degrees(sigma)

# static error
static_error_svd = []
static_error_davenport = []
static_error_gn = []

for k in range(n):
    yk = ytraj[:,k].reshape(3, m, order='F')

    Q_est_svd, q_est_svd = att.wahba_svd(yk, r_N)
    Q_est_davenport, q_est_davenport = att.wahba_davenport(yk, r_N)
    Q_est_gn, q_est_gn = att.wahba_gn()

    static_error_svd.append(
        attitude_error(q_est_svd, q_true[:,k])   # ✅ FIX: consistent metric
    )
    static_error_davenport.append(
        attitude_error(q_est_davenport, q_true[:,k])   # ✅ FIX: consistent metric
    )
    static_error_gn.append(
        attitude_error(q_est_gn, q_true[:,k])   # ✅ FIX: consistent metric
    )

static_error_svd = np.array(static_error_svd)
static_error_davenport = np.array(static_error_davenport)
static_error_gn = np.array(static_error_gn)

t = np.arange(n)*dt

#1. Quaternion comparison (cleaner)
plt.figure()
labels = ['q0','q1','q2','q3']
for i in range(4):
    plt.plot(t, q_true[i,:], label=f'{labels[i]} true')
    plt.plot(t, xfilt[i,:], '--', label=f'{labels[i]} est')
plt.legend()
plt.title('Quaternion Comparison')
plt.xlabel("Time [s]")
plt.grid()

# plt.figure()
# labels = ['q0','q1','q2','q3']
# plt.plot(t, q_true[1,:], label=f'{labels[1]} true')
# plt.plot(t, xfilt[1,:], '--', label=f'{labels[1]} est')
# plt.legend()
# plt.title('Quaternion Comparison')
# plt.xlabel("Time [s]")
# plt.grid()

# plt.figure()
# labels = ['q0','q1','q2','q3']
# plt.plot(t, q_true[2,:], label=f'{labels[2]} true')
# plt.plot(t, xfilt[2,:], '--', label=f'{labels[2]} est')
# plt.legend()
# plt.title('Quaternion Comparison')
# plt.xlabel("Time [s]")
# plt.grid()

# plt.figure()
# labels = ['q0','q1','q2','q3']
# plt.plot(t, q_true[3,:], label=f'{labels[3]} true')
# plt.plot(t, xfilt[3,:], '--', label=f'{labels[3]} est')
# plt.legend()
# plt.title('Quaternion Comparison')
# plt.xlabel("Time [s]")
# plt.grid()

# 2. MEKF error
plt.figure()
plt.plot(t, error)
plt.ylim([0,5])
plt.xlim([0,1.5])
plt.xlabel("Time [s]")
plt.ylabel("Error [deg]")
plt.title("MEKF Attitude Error")
plt.grid()

# # 3. Static vs MEKF
# plt.figure()
# #plt.plot(t, error, label="MEKF")
# plt.plot(t, static_error_svd, label="Static (Wahba - SVD)")
# #plt.plot(t, static_error_davenport, label="Static (Wahba - Davenport)")
# #plt.plot(t, static_error_gn, label="Static (GN)")
# plt.ylim([0,5])
# plt.xlabel("Time [s]")
# plt.ylabel("Error [deg]")
# plt.legend()
# plt.title("MEKF vs Static Estimator")
# plt.grid()

# 4. Consistency
plt.figure()
plt.plot(t, error, label="Error")
plt.plot(t, sigma_deg, label="sqrt(trace(P))")
plt.ylim([0,5])
plt.xlabel("Time [s]")
plt.ylabel("Degrees")
plt.legend()
plt.title("Consistency Check")
plt.grid()

plt.show()