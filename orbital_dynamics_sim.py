import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11
M = 2.287e19
mu = G * M

R_psyche = 111e3        # meters
altitude = 303e3        # meters
r0 = R_psyche + altitude

v_circ = np.sqrt(mu / r0)   # m/s
print(v_circ)

# Initial state (polar circular orbit)
state0 = np.array([
    r0, 0, 0,   # x, y, z
    0, 0, v_circ    # vx, vy, vz
])

def two_body(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    
    return np.array([vx, vy, vz, ax, ay, az])

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


T_orbit = 2 * np.pi * np.sqrt(r0**3 / mu)
t0 = 0
tf = 2 * T_orbit
dt = 10.0  

t, state = rk4(two_body, state0, t0, tf, dt)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

# Plot orbit
ax.plot(state[:,0], state[:,1], state[:,2], linewidth=1.5)

# Draw Psyche as a sphere
u = np.linspace(0, 2*np.pi, 50)
v_sphere = np.linspace(0, np.pi, 50)

x_s = R_psyche * np.outer(np.cos(u), np.sin(v_sphere))
y_s = R_psyche * np.outer(np.sin(u), np.sin(v_sphere))
z_s = R_psyche * np.outer(np.ones(np.size(u)), np.cos(v_sphere))

ax.plot_surface(x_s, y_s, z_s, alpha=0.3)
axis_length = 1.2 * r0

ax.quiver(0,0,0, axis_length,0,0)
ax.quiver(0,0,0, 0,axis_length,0)
ax.quiver(0,0,0, 0,0,axis_length)

ax.text(axis_length,0,0,'X', fontsize=12)
ax.text(0,axis_length,0,'Y', fontsize=12)
ax.text(0,0,axis_length,'Z', fontsize=12)

max_range = r0
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Polar Orbit Around Psyche")

plt.show()