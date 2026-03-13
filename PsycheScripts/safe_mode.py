import math as m
import numpy as np
import scipy.linalg as la
from scipy.linalg import expm
import matplotlib.pyplot as plt
import psyche_model as psy
import adcs_toolbox as adcs

##### Problem 1.1 #####

J = psy.tot_moicom

def perturb_inertia(J):
    eigenvalues, V = np.linalg.eig(J)
    D = np.diag(eigenvalues)

    d = 0.03 * np.random.randn(3)
    D_perturb = D @ (np.eye(3) + np.diag(d))

    v = np.deg2rad(3) * np.random.randn(3)
    V_perturb = V @ expm(adcs.hat(v))

    J_perturb = V_perturb @ D_perturb @ V_perturb.T

    return J_perturb

J = perturb_inertia(J)

##### Problem 1.2 #####

# assuming panel normal is in positive z direction in body frame
panel_normal = np.array([0,0,1])

# convert 10 RPM to rad/s
omega_mag = 10 * 2*np.pi / 60
print(omega_mag)

# desired angular velocity
omega_desired = omega_mag * panel_normal

##### Problem 1.3 #####

# dynamic balance calculation
omega_s = omega_desired
Js = (omega_s/np.linalg.norm(omega_s)) @ J @ (omega_s/np.linalg.norm(omega_s))

# enforcing the 1.2 inertia ratio constraint
rho_s = omega_s[2] * (1.2*J[2,2] - Js)

# intermediate functions to make the math easier to read
A = np.vstack([
    omega_s.reshape(1,3),
    adcs.hat(omega_s)
])

B = np.hstack([
    rho_s * omega_s[2],
    -adcs.hat(omega_s) @ (J @ omega_s)
])

rho0, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

##### Problem 1.4 #####

def dynamics(x, u):

    q = x[0:4]
    q = q / np.linalg.norm(q)

    omega = x[4:7]
    rho = x[7:10]

    rho_dot = u

    q_dot = 0.5 * adcs.G(q) @ omega

    # gyrostat dynamics
    omega_dot = -np.linalg.solve(
        J,
        rho_dot + adcs.hat(omega) @ (J @ omega + rho)
    )

    x_dot = np.concatenate([q_dot, omega_dot, rho_dot])

    return x_dot


def rk4_step(x, u, h):

    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*h*k1, u)
    k3 = dynamics(x + 0.5*h*k2, u)
    k4 = dynamics(x + h*k3, u)

    xn = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    xn[0:4] = xn[0:4] / np.linalg.norm(xn[0:4])

    return xn

##### Problem 1.5 #####

# initial conditions, small perturbation in the initial angular velocity
q0 = np.array([1,0,0,0])
omega0 = omega_desired + 0.01*np.random.randn(3)
x0 = np.concatenate([q0, omega0, rho0])

h = 0.1
n = 600

xhist = np.zeros((10, n))
xhist[:,0] = x0

u = np.zeros(3)

for k in range(n-1):
    xhist[:,k+1] = rk4_step(xhist[:,k], u, h)

t = np.arange(n) * h

omega_hist = xhist[4:7,:]

##### Plot Results #####

# plot angular velocity components over time 
plt.figure()
plt.plot(t, omega_hist[0], label='ωx')
plt.plot(t, omega_hist[1], label='ωy')
plt.plot(t, omega_hist[2], label='ωz')

plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity Components")
plt.legend()
plt.grid()
plt.show()

# plot total angular momentum magnitude over time
h_total = np.zeros(n)

for k in range(n):
    omega = xhist[4:7,k]
    rho = xhist[7:10,k]
    h_vec = J @ omega + rho
    h_total[k] = np.linalg.norm(h_vec)

plt.figure()
plt.plot(t, h_total)
plt.xlabel("Time (s)")
plt.ylabel("|h|")
plt.ylim(0, 1.2*np.max(h_total))
plt.title("Total Angular Momentum Magnitude")
plt.grid()
plt.show()

# plot rotor momentum components over time
rho_hist = xhist[7:10,:]

plt.figure()
plt.plot(t, rho_hist[0], label='ρx')
plt.plot(t, rho_hist[1], label='ρy')
plt.plot(t, rho_hist[2], label='ρz')

plt.xlabel("Time (s)")
plt.ylabel("Rotor Momentum")
plt.title("Rotor Momentum Components")
plt.legend()
plt.grid()
plt.show()