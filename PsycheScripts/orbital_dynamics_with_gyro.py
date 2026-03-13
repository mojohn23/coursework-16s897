import numpy as np
import matplotlib.pyplot as plt
import adcs_toolbox as adcs
import safe_mode as sm
import orbital_dynamics_sim as ods

G = 6.67430e-11
M = 2.287e19
mu = G * M

R_psyche = 111e3        # meters
altitude = 303e3        # meters
r0 = R_psyche + altitude

v_circ = np.sqrt(mu / r0)   # m/s

# importing inertia matrix calculated in problem 1
J = sm.J 

# assuming panel normal is in positive x direction
n_body = np.array([1,0,0])

# 10 RPM to rad/s
omega_mag = 10 * 2*np.pi / 60
omega_desired = omega_mag * n_body

# same dynamic balance calculation from PSET 1
omega_s = omega_desired
Js = (omega_s/np.linalg.norm(omega_s)) @ J @ (omega_s/np.linalg.norm(omega_s))
rho_s = omega_s[0] * (1.2*J[0,0] - Js)

A = np.vstack([
    omega_s.reshape(1,3),
    adcs.hat(omega_s)
])

B = np.hstack([
    rho_s * omega_s[0],
    -adcs.hat(omega_s) @ (J @ omega_s)
])

rho0, _, _, _ = np.linalg.lstsq(A, B, rcond=None)


q0 = np.array([1,0,0,0]) 
omega0 = omega_desired + 0.01*np.random.randn(3)
r0_vec = np.array([r0, 0, 0])
v0_vec = np.array([0, 0, v_circ])

# full state: [q, omega, rho, r, v]
state0 = np.concatenate([q0, omega0, rho0, r0_vec, v0_vec])

def gyrostat_orbit(t, state):
    q = state[0:4]
    omega = state[4:7]
    rho = state[7:10]
    r = state[10:13]
    v = state[13:16]

    q_dot = 0.5 * adcs.G(q) @ omega
    omega_dot = -np.linalg.solve(J, np.cross(omega, J @ omega + rho))
    rho_dot = np.zeros(3)

    # translational motion
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3

    return np.concatenate([q_dot, omega_dot, rho_dot, v, a])

# simulation parameters
t0 = 0
tf = 100
dt = 0.1  

t, state = ods.rk4(gyrostat_orbit, state0, t0, tf, dt)
q_vals = state[:,0:4]
r_vals = state[:,10:13]

# pointing error
sun_vector = np.array([1,0,0])  # sun along inertial x
pointing_error_deg = []

for q in q_vals:
    n_eci = adcs.Q(q) @ n_body
    theta = np.degrees(np.arccos(np.clip(np.dot(n_eci, sun_vector), -1, 1)))
    pointing_error_deg.append(theta)

pointing_error_deg = np.array(pointing_error_deg)

# plot quaternion components
plt.figure(figsize=(10,6))
plt.plot(t, q_vals[:,0], label='q0')
plt.plot(t, q_vals[:,1], label='q1')
plt.plot(t, q_vals[:,2], label='q2')
plt.plot(t, q_vals[:,3], label='q3')
plt.xlabel('Time [s]')
plt.ylabel('Quaternion components')
plt.legend()
plt.title('Attitude Quaternion Components Over Time')
plt.grid()
plt.show()

# plot pointing error
plt.figure(figsize=(10,5))
plt.plot(t, pointing_error_deg)
plt.xlabel('Time [s]')
plt.ylabel('Pointing Error [deg]')
plt.title('Solar Panel Pointing Error Over Time')
plt.grid()
plt.show()