import math
import numpy as np
import adcs_toolbox as adcs
from psyche_model import tot_moicom as J
from attitude_estimation import S

### CHAT I DON'T THINK WE EVEN NEED POSITION AND VELOCITY LOWKEY
# Psyche and orbit parameters for orbital dynamics model
G = 6.67430e-11
M = 2.287e19
mu = G * M
R_psyche = 111e3        # meters
altitude = 303e3        # meters
r0 = R_psyche + altitude
v_circ = np.sqrt(mu / r0)   # m/s
n_body = np.array([1,0,0]) # Sun-pointing in inertial x-direction

### Initial conditions ###
# Assume 0.1 deg/s body rate, uncontrolled tumble (no gyro?)
q0 = adcs.unit_vec(np.random.randn(4)) # Random orientation, unit quaternion
omega0 = np.array([math.radians(0.1), 0, 0]) # [rad/s]
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

dt = 0.1 # Time step [s]
n = 3 # Number of time steps
tf = n*dt # One min

t, state = rk4(orbit_dynamics, state0, 0, tf, dt) # The state comes out horizontally as (n) x (q omega r v)
state = state.T # Rotate it back to (q omega r v) x (n)

##### NOISE MODELS #######################################
# ytraj is (4*n_SRU + 3*n_CSS)x(n) representing two quaternion SRU measurements and 2 3-parameter CSS measurements
n_SRU = 2 # Two star trackers
n_CSS = 2 # Two Sun sensors
ytraj = np.zeros((4*n_SRU + 3*n_CSS, n)) # 14xn

# Noisy star trackers
S_SRU = S[0] # From attitude_estimation.py
for k in range(n):
    qk = state[:4, k]
    for j in range(n_SRU):
        noise_vec = np.random.multivariate_normal(np.zeros(3), S_SRU) # This is a 3-parameter rotation
        delta_q = adcs.qexp(noise_vec) # Turn the noise vector to a noise delta quaternion
        ytraj[j*4:(j + 1)*4, k] = adcs.qmult(qk, delta_q) # Quaternion multiply on the error

# Noisy sun sensors
S_CSS = S[3] # From attitude_estimation.py
M_CSS = np.diag(1 + 0.06*np.random.randn(3))
b_CSS = 0.015*np.random.randn(3)
for k in range(n):
    qk = state[:4, k]
    phik = adcs.qlog(qk)
    for j in range(n_CSS):
        w_CSS = 0.015*np.random.randn(3) # Get new sample every time
        ytraj[j*3+8:(j+1)*3+8, k] = M_CSS@phik + b_CSS + w_CSS

# Noisy gyro
n_gyro = 1
M_gyro = np.diag(1 + 0.01*np.random.randn(3))
b_gyro = np.radians(0.0035)/3600*np.random.randn(3) # converted to [rad/s]
gyro = np.zeros((3*n_gyro, n))
for k in range(n):
    w_gyro = np.radians(0.0035)/60/np.sqrt(dt)*np.random.randn(3) # converted to [rad/sqrt(s)] and divided by sqrt(dt)
    gyro[:, k] = state[4:7, k] + b_gyro + w_gyro

##### KALMAN FILTER #######################################
def state_prediction(x, u, h):
    q = x[:4] # q_k+1|k = q_k|k * deltaq_k
    beta = x[4:7] # beta_k+t|k = beta_k
    return np.concatenate([adcs.L(q)@adcs.qexp(1/2*h*(u - beta)), beta]) # x_k+1|k = [q_k+1|k; beta_k+1|k]

def state_prediction_deriv(x, u, h):
    q = x[:4]
    beta = x[4:7]
    qk1 = adcs.L(q)@adcs.qexp(1/2*h*(u - beta))
    dphidphi = adcs.G(qk1).T@adcs.R(adcs.qexp(1/2*h*(u - beta)))@adcs.G(q)
    dphidbeta = -1/2*h*adcs.G(qk1).T@adcs.G(q)
    return np.block([[dphidphi, dphidbeta], [np.zeros([3, 3]), np.eye(3)]]) # A = [dphidphi, dphidbeta; 0, I]

def innovation(x, y):
    q = x[:4]
    zk = np.zeros(3*(n_SRU + n_CSS))
    for k in range(n_SRU):
        y_SRU = y[k*4:(k + 1)*4]
        zk[k*3:(k + 1)*3] = adcs.qlog(adcs.qmult(adcs.qinv(q), y_SRU))
    for k in range(n_CSS):
        y_CSS = y[8 + k*3:8 + (k + 1)*3]
        y_CSSpred = adcs.R(q).T@y_CSS
        zk[3*n_SRU + k*3:3*n_SRU + (k + 1)*3] = y_CSS - M_CSS@y_CSSpred
        return zk


def innovation_deriv(x, y):
    q = x[:4]
    C = np.zeros((3*(n_SRU + n_CSS), 6))
    for k in range(n_SRU):
        y_SRU = y[k*4:(k + 1)*4]
        C[3*k:3*(k + 1), :3] = adcs.H.T@adcs.R(y_SRU)@adcs.Tmat@adcs.G(q)
    for k in range(n_CSS):
        y_CSS = y[8 + k*3:8 + (k + 1)*3]
        C[3*n_SRU + 3*k:3*n_SRU + 3*(k + 1), :3] = adcs.H.T@adcs.R(y_CSS[:, k])@adcs.Tmat@adcs.G(q)
    return C

# Noise models as matrices
# Process noise V = [w_gryo, b_gyro]
V = np.block([[(np.radians(0.0035)/60/np.sqrt(dt))**2*np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), (np.radians(0.0035)/3600)**2*np.eye(3)]])
# Measurement noise W = [S_SRU, S_SRU, S_CSS, S_CSS] 12x12 diagonal
W = np.zeros((3*(n_SRU + n_CSS), 3*(n_SRU + n_CSS)))
for i in range(n_SRU):
    W[3*i:3*i + 3, 3*i:3*i + 3] = S_SRU
for i in range(n_CSS):
    W[3*i + 3*n_SRU:3*i + 3*n_SRU + 3, 3*i + 3*n_SRU:3*i + 3*n_SRU + 3] = S_CSS

# The state vector is x = [q beta]
x_kk = np.zeros((7, n))
x_kk[:4, 0] = adcs.unit_vec(q0 + 0.01*np.random.randn(4))

P_kk = np.zeros((6, 6, n))
P_kk[:, :, 0] = 0.5*np.eye(6) # not sure why we assume everything is 1/2 here

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
    phik1 = (-K@z)[:3]
    betak1 = x_k1k[4:7] - (K@z)[3:6]
    qk1 = adcs.L(x_k1k[:4])@np.concatenate([[np.sqrt(1 - phik1.T@phik1)], phik1])
    x_kk[:, k + 1] = np.concatenate([qk1, betak1])
    P_kk[:, :, k + 1] = (np.eye(6) - K@C)@P_k1k@(np.eye(6) - K@C).T + K@W@K.T

##### Make plots ##################################################
