import math
import numpy as np
import adcs_toolbox as adcs
from psyche_model import tot_moicom as J
import attitude_estimation as att_est
import matplotlib.pyplot as plt
import scipy

### CHAT I DON'T THINK WE EVEN NEED POSITION AND VELOCITY LOWKEY
# Psyche and orbit parameters for orbital dynamics model
G = 6.67430e-11
M = 2.287e19
mu = G * M
R_psyche = 111e3        # meters
altitude = 303e3        # meters
r0 = R_psyche + altitude
v_circ = np.sqrt(mu / r0)   # m/s
sun_vec = adcs.unit_vec(np.random.randn(3)) # Sun-pointing in inertial frame

### Initial conditions ###
# Assume 0.1 deg/s body rate, uncontrolled tumble (no gyro?)
q0 = adcs.unit_vec(np.random.randn(4)) # Random orientation, unit quaternion
omega0 = np.random.random(3) # [rad/s]
# r0 = np.array([r0, 0, 0]) # Starting at periapsis
# v0 = np.array([0, 0, v_circ]) # Assuming circular orbit
state0 = np.concatenate([q0, omega0])

def orbit_dynamics(t, state):
    q = state[0:4]
    omega = state[4:7]
    # r = state[7:10]
    # v = state[10:13]

    # Rotational motion (zero-torque)
    q_dot = 0.5 * adcs.G(q) @ omega
    omega_dot = -np.linalg.solve(J, np.cross(omega, J@omega))

    # Orbital motion
    # r_norm = np.linalg.norm(r)
    # a = -mu * r / r_norm**3

    return np.concatenate([q_dot, omega_dot])

def rk4(f, state0, t0, tf, dt):
    t_values = np.arange(t0, tf, dt)
    state_values = np.zeros((len(t_values), len(state0)))
    state = state0.copy()
    
    for i, t in enumerate(t_values):
        state_values[i] = state
        
        k1 = f(t, state)
        k2 = f(t + dt/2, state + dt*k1/2)
        k3 = f(t + dt/2, state + dt*k2/2)
        k4 = f(t + dt, state + dt*k3)
        
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return t_values, state_values

def Qdiff(Q_computed, Q_real):
    # Find the difference in degrees between two rotation matrices
    return (180/math.pi)*np.linalg.norm(adcs.unhat(scipy.linalg.logm(Q_computed.T@Q_real)))

dt = 0.1 # Time step [s]
n = 600 # Number of time steps
tf = n*dt

t, state = rk4(orbit_dynamics, state0, 0, tf, dt) # The state comes out horizontally as (n) x (q omega r v)
state = state.T # Rotate it back to be (q omega r v) x (n)

##### NOISE MODELS #######################################
# ytraj is (4*n_SRU + 3*n_CSS)x(n) representing two quaternion SRU measurements and 2 3-parameter CSS measurements
n_SRU = 2 # Two star trackers
n_CSS = 2 # Two Sun sensors
ytraj = np.zeros((4*n_SRU + 3*n_CSS, n)) # 14xn
r_B = np.zeros((3, n_CSS + n_SRU, n)) # 3x4xn
r_N_real = np.zeros((3, n_CSS + n_SRU, n)) # 3x4xn

# Noisy star trackers
S_SRU = att_est.S[0] # From attitude_estimation.py
sru_ref = np.zeros((3, n_SRU))
for i in range(n_SRU):
    sru_ref[:, i] = adcs.unit_vec(np.random.randn(3)) # Generate random true vector in the body frame
for k in range(n):
    qk = state[:4, k]
    for j in range(n_SRU):
        noise_vec = np.random.multivariate_normal(np.zeros(3), S_SRU) # This is a 3-parameter rotation
        delta_q = adcs.qexp(noise_vec) # Turn the noise vector to a noise delta quaternion
        q_noisy = adcs.qmult(qk, delta_q) # Quaternion multiply on the error
        ytraj[j*4:(j + 1)*4, k] = q_noisy # Output in ytraj

        r_N_real[:3, j, k] = sru_ref.T[j, :] # Real vector in inertial frame
        r_B[:3, j, k] = adcs.unit_vec(adcs.Q(q_noisy).T@r_N_real[:3, j, k]) # Noisy vector in body frame

# Noisy sun sensors
S_CSS = att_est.S[3] # From attitude_estimation.py
M_CSS = np.diag(1 + 0.06*np.random.randn(3))
b_CSS = 0.015*np.random.randn(3)
for k in range(n):
    qk = state[:4, k]
    y_ideal = adcs.Q(qk).T@sun_vec # Ideal in body frame
    for j in range(n_CSS):
        w_CSS = 0.015*np.random.randn(3) # Get new sample every time
        ytraj[j*3+4*n_SRU:(j+1)*3+4*n_SRU, k] = M_CSS@y_ideal + b_CSS + w_CSS

        r_N_real[:3, j + n_SRU, k] = sun_vec # Real vector in inertial frame
        r_B[:3, j + n_SRU, k] = M_CSS@y_ideal + b_CSS + w_CSS

# Noisy gyro
n_gyro = 1
M_gyro = np.diag(1 + 0.01*np.random.randn(3))
b_gyro = np.radians(0.0035)/3600*np.random.randn(3) # converted to [rad/s]
gyro = np.zeros((3*n_gyro, n))
for k in range(n):
    w_gyro = np.radians(0.0035)/60/np.sqrt(dt)*np.random.randn(3) # converted to [rad/sqrt(s)] and divided by sqrt(dt)
    gyro[:, k] = M_gyro@state[4:7, k] + b_gyro + w_gyro

##### KALMAN FILTER #######################################
def state_prediction(x, u, h):
    q = x[:4] # q_k+1|k = q_k|k * deltaq_k
    beta = x[4:7] # beta_k+1|k = beta_k
    x_k1k = np.concatenate([adcs.L(q)@adcs.qexp(1/2*h*(u - beta)), beta]) # x_k+1|k = [q_k+1|k; beta_k+1|k]
    return x_k1k

def state_prediction_deriv(x, u, h):
    q = x[:4]
    beta = x[4:7]
    qk1 = adcs.L(q)@adcs.qexp(1/2*h*(u - beta))
    dphidphi = adcs.G(qk1).T@adcs.R(adcs.qexp(1/2*h*(u - beta)))@adcs.G(q)
    dphidbeta = -1/2*h*adcs.G(qk1).T@adcs.G(q)
    A = np.block([[dphidphi, dphidbeta], [np.zeros([3, 3]), np.eye(3)]]) # A = [dphidphi, dphidbeta; 0, I]
    return A

def innovation(x, y):
    q = x[:4]
    zk = np.zeros(3*(n_SRU + n_CSS))
    for k in range(n_SRU):
        y_SRU = y[k*4:(k + 1)*4]
        zk[k*3:(k + 1)*3] = adcs.qlog(adcs.qmult(adcs.qinv(q), y_SRU))
    y_CSSpred = adcs.Q(q).T@sun_vec
    for k in range(n_CSS):
        y_CSS = y[4*n_SRU + k*3:4*n_SRU + (k + 1)*3]
        zk[3*n_SRU + k*3:3*n_SRU + (k + 1)*3] = y_CSS - M_CSS@y_CSSpred
    return zk


def innovation_deriv(x, y):
    q = x[:4]
    C = np.zeros((3*(n_SRU + n_CSS), 6))
    for k in range(n_SRU):
        y_SRU = y[k*4:(k + 1)*4]
        C[3*k:3*(k + 1), :3] = adcs.H.T@adcs.R(y_SRU)@adcs.Tmat@adcs.G(q)
    y_CSSpred = adcs.Q(q).T@sun_vec
    for k in range(n_CSS):
        # y_CSS = y[4*n_SRU + k*3:4*n_SRU + (k + 1)*3]
        C[3*n_SRU + 3*k:3*n_SRU + 3*(k + 1), :3] = adcs.hat(y_CSSpred)
    return C

# Noise models as matrices
# Process noise V = diag([w_gryo, b_gyro]) 6x6
V = np.block([[(np.radians(0.0035)/60/np.sqrt(dt))**2*np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), (np.radians(0.0035)/3600)**2*np.eye(3)]])
# Measurement noise W = diag([S_SRU, S_SRU, S_CSS, S_CSS]) 12x12
W = np.zeros((3*(n_SRU + n_CSS), 3*(n_SRU + n_CSS)))
for i in range(n_SRU):
    W[3*i:3*i + 3, 3*i:3*i + 3] = S_SRU
for i in range(n_CSS):
    W[3*i + 3*n_SRU:3*i + 3*n_SRU + 3, 3*i + 3*n_SRU:3*i + 3*n_SRU + 3] = S_CSS

# Initialize filter
# The state vector is x = [q beta]
x_kk = np.zeros((7, n))
x_kk[:4, 0] = adcs.unit_vec(q0)

P_kk = np.zeros((6, 6, n))
P_kk[:, :, 0] = 10*np.eye(6) # Arbitrary value, start small

# Begin loop
for k in range(n - 1):
    # Prediction
    x_k1k = state_prediction(x_kk[:, k], gyro[:, k], dt)
    A = state_prediction_deriv(x_kk[:, k], gyro[:, k], dt)
    P_k1k = A@P_kk[:, :, k]@A.T + V

    # Innovation
    z = innovation(x_k1k, ytraj[:, k + 1]) # come back to dis
    C = innovation_deriv(x_k1k, ytraj[:, k + 1])
    S = C@P_k1k@C.T + W

    # Kalman gain
    K = P_k1k@C.T@np.linalg.inv(S)

    # Update
    phik1 = -(K@z)[:3]
    betak1 = x_k1k[4:7] - (K@z)[3:6]
    qk1 = adcs.L(x_k1k[:4])@np.concatenate([[np.sqrt(1 - phik1.T@phik1)], phik1])
    x_kk[:, k + 1] = np.concatenate([adcs.unit_vec(qk1), betak1])
    P_kk[:, :, k + 1] = (np.eye(6) - K@C)@P_k1k@(np.eye(6) - K@C).T + K@W@K.T

##### Make plots ##################################################
if True:
    for i in range(4):
        plt.figure()
        plt.plot(t, x_kk[i, :], label = 'filter')
        plt.plot(t, state[i, :], linestyle = '--', label = 'truth')
        plt.title(f'Quaternion component q[{i + 1}]')
        plt.legend()
        plt.xlabel('Time (s)')
    for i in range(3):
        plt.figure()
        plt.plot(t, x_kk[i + 4, :], label = 'filter')
        plt.axhline(b_gyro[i], linestyle = '--', label = 'truth', color = 'orange')
        plt.title(f'Beta component b[{i + 1}]')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('rad/s')

# Compare filter to static estimate
if False:
    error_kalman = np.zeros(n)
    error_svd = np.zeros(n)
    error_dav = np.zeros(n)

    for i in range(n):
        # "Real" attitude
        q_real = state[:4, i]
        Q_real = adcs.Q(q_real)

        # Compare the filter attitude to the state ("real") attitude
        q_filt = x_kk[:4, i]
        Q_filt = adcs.Q(q_filt)
        error_kalman[i] = Qdiff(Q_filt, Q_real)

        # Compare the Wahba attitude to the state ("real") attitude
        Q_dav, _ = att_est.wahba_davenport(r_B[:, :, i], r_N_real[:, :, i])
        error_dav[i] = Qdiff(Q_dav, Q_real)

        Q_svd, _ = att_est.wahba_svd(r_B[:, :, i], r_N_real[:, :, i])
        error_svd[i] = Qdiff(Q_svd, Q_real)


    plt.figure()
    plt.plot(t, error_kalman, color = 'blue', label = 'Kalman error')
    plt.plot(t, error_svd, color = 'red', label = 'Wahba SVD error')
    plt.plot(t, error_dav, color = 'orange', label = 'Wahba q-Method error')
    plt.title('Error comparison')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Error (deg)')

# Compare filter covariance
if True:
    sigma = np.zeros((6, n))
    phi_error = np.zeros((3, n))
    for i in range(n):
        sigma[:, i] = np.sqrt(np.array([np.diag(P_kk[:, :, i])])) # 1-sigma is sqrt(P_kk)

        q_filt = x_kk[:4, i]
        q_real = state[:4, i]
        q_error  = adcs.qmult(adcs.qinv(q_filt), q_real)  # Error quaternion
        phi_error[:, i] = adcs.qlog(q_error) # Get the error as a 3-parameter vector
    
    plt.figure()
    plt.plot(t[1:], sigma[0, 1:], linestyle = '--', color = 'blue', label = '1-sigma')
    plt.plot(t[1:], -sigma[0, 1:], linestyle = '--', color = 'blue')
    plt.plot(t[1:], phi_error[0, 1:], color = 'blue', label = 'Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('phi_x')

    plt.figure()
    plt.plot(t[1:], sigma[1, 1:], linestyle = '--', color = 'blue', label = '1-sigma')
    plt.plot(t[1:], -sigma[1, 1:], linestyle = '--', color = 'blue')
    plt.plot(t[1:], phi_error[1, 1:], color = 'blue', label = 'Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('phi_y')

    plt.figure()
    plt.plot(t[1:], sigma[2, 1:], linestyle = '--', color = 'blue', label = '1-sigma')
    plt.plot(t[1:], -sigma[2, 1:], linestyle = '--', color = 'blue')
    plt.plot(t[1:], phi_error[2, 1:], color = 'blue', label = 'Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('phi_z')