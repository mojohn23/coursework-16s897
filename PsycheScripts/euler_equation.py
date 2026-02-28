import math as m
import numpy as np
import matplotlib.pyplot as plt
import psyche_model as psy
import adcs_toolbox as adcs

J = psy.tot_moicom

def euler_simp(t, state, J = psy.tot_moicom):
    # t is not actually used here
    # State vector must be 1x7 horizontal vec
    q = state[:4]
    w = state[4:]

    qdot = 1/2*(adcs.G(q) @ w)
    wdot = -np.linalg.solve(J, adcs.hat(w) @ (J @ w))
    state = np.hstack((np.array(qdot), np.array(wdot)))
    return state

def rk4(f, state0, t0, tf, dt):
    t_val = np.arange(t0, tf, dt)
    state_val = np.zeros((len(t_val), len(state0))) # Initialize array
    state = state0.copy()

    for i, t in enumerate(t_val):
        state_val[i] = state

        k1 = f(t, state)
        k2 = f(t + dt/2, state + dt*k1/2)
        k3 = f(t + dt/2, state + dt*k2/2)
        k4 = f(t + dt, state + dt*k3)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t_val, state_val

def plot_omega(t_val, state_val, title):
    fig, ax = plt.subplots()
    ax.plot(t_val, state_val[:, 4], color = 'blue', label = '$\omega_x$')
    ax.plot(t_val, state_val[:, 5], color = 'red', label = '$\omega_y$')
    ax.plot(t_val, state_val[:, 6], color = 'green', label = '$\omega_z$')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title(title)

# Assume initial attitude is perfect pointing
q0 = np.array([1, 0, 0, 0]) # [scalar, vector]
t0 = 0
tf = 60 # [sec]
dt = 0.1 # [sec]

# Set perturbation here
perturbation = np.array([0, 0, 0])
# perturbation = 0.1*np.random.randint(1, 5, size = (1, 3))

# First case: only w about minor axis
w = np.array([10*2*m.pi/60, 0, 0]) # [rad/s], equivalent to 10 RPM
w = w + perturbation[0]
state0 = np.hstack([q0, w])
t_val, state_val = rk4(euler_simp, state0, t0, tf, dt)
if perturbation.all == 0:
    plot_omega(t_val, state_val, 'Case 1: Pure Rotation about Minor Axis')
else:
    plot_omega(t_val, state_val, 'Case 1: Rotation with Perturbation about Minor Axis')

# Second case: only w about intermediate axis
w = np.array([0, 10*2*m.pi/60, 0]) # [rad/s], equivalent to 10 RPM
w = w + perturbation[0]
state0 = np.hstack([q0, w])
t_val, state_val = rk4(euler_simp, state0, t0, tf, dt)
if perturbation.all == 0:
    plot_omega(t_val, state_val, 'Case 2: Pure Rotation about Intermediate Axis')
else:
    plot_omega(t_val, state_val, 'Case 2: Rotation with Perturbation about Intermediate Axis')

# Third case: only w about major axis
w = np.array([0, 0, 10*2*m.pi/60]) # [rad/s], equivalent to 10 RPM
w = w + perturbation[0]
state0 = np.hstack([q0, w])
t_val, state_val = rk4(euler_simp, state0, t0, tf, dt)
if perturbation.all == 0:
    plot_omega(t_val, state_val, 'Case 3: Pure Rotation about Major Axis')
else:
    plot_omega(t_val, state_val, 'Case 3: Rotation with Perturbation about Major Axis')