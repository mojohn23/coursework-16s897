import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Based on demo code from Lecture 4

# helper functions

def hat(v):
    return np.array([
        [0,      -v[2],  v[1]],
        [v[2],    0,    -v[0]],
        [-v[1],   v[0],  0]
    ])

H = np.vstack([np.zeros((1,3)), np.eye(3)])

def L(q):
    return np.block([
        [np.array([[q[0]]]),        -q[1:4]],
        [q[1:4].reshape(3,1), q[0]*np.eye(3) + hat(q[1:4])]
    ])

def R(q):
    return np.block([
        [np.array([[q[0]]]),        -q[1:4]],
        [q[1:4].reshape(3,1), q[0]*np.eye(3) - hat(q[1:4])]
    ])

def G(q):
    return L(q) @ H

def Q(q):
    return H.T @ L(q) @ R(q).T @ H

J = np.array([
    [56950.27703515, 0.0, 0.0],
    [0.0, 5039.18217077, 0.0],
    [0.0, 0.0, 58455.44159671]
])

def dynamics(x):
    q = x[0:4]
    omega = x[4:7]

    q_dot = 0.5 * G(q) @ omega
    omega_dot = -np.linalg.solve(J, hat(omega) @ J @ omega)

    return np.hstack([q_dot, omega_dot])

def rkstep(x):
    f1 = dynamics(x)
    f2 = dynamics(x + 0.5*h*f1)
    f3 = dynamics(x + 0.5*h*f2)
    f4 = dynamics(x + h*f3)

    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xn[0:4] = xn[0:4] / np.linalg.norm(xn[0:4])

    return xn


# simulation
h = 0.1
n = 1000
tf = n * h

q0 = np.array([1, 0, 0, 0])
omega0 = np.array([1, 1, 0]) + 0.1*np.random.randn(3)
x0 = np.hstack([q0, omega0])

xhist = np.zeros((7, n))
xhist[:,0] = x0

for k in range(n-1):
    xhist[:,k+1] = rkstep(xhist[:,k])

hn = np.zeros((3,n))
T = np.zeros(n)

for k in range(n):
    T[k] = 0.5 * xhist[4:7,k].T @ J @ xhist[4:7,k]
    hn[:,k] = Q(xhist[0:4,k]) @ (J @ xhist[4:7,k])

# plot momentum sphere

u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, alpha=0.3)

hhist = np.zeros((3,n))
for k in range(n):
    hvec = J @ xhist[4:7,k]
    hhist[:,k] = 1.03 * hvec / np.linalg.norm(hvec)

ax.plot3D(hhist[0,:], hhist[1,:], hhist[2,:])

principal_axes = np.eye(3)

# plot equilibrium points 
for i in range(3):
    axis = principal_axes[:,i]
    ax.scatter(axis[0], axis[1], axis[2], s=80)
    ax.scatter(-axis[0], -axis[1], -axis[2], s=80)

ax.set_xlabel("Hx")
ax.set_ylabel("Hy")
ax.set_zlabel("Hz")
ax.set_title("Momentum Sphere with Trajectory")
ax.set_box_aspect([1, 1, 1]) 

plt.show()