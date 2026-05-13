# psyche_actuators
import numpy as np
import adcs_toolbox as adcs
import psyche_model as psyche
import matplotlib.pyplot as plt

class rwa:
    def __init__(self, position):
        # Position is relative to the SPACECRAFT BUS COM, then corrected for the total spacecraft COM
        self.position = position - psyche.tot_com.reshape(3,) # Position relative to total spacecraft COM
        self.axis = adcs.unit_vec(position) # Spin axis relative to the total spacecraft COM
        self.mass = 12 # [kg]
        self.MOI = 1 # [moment of inertia?]

rwa1 = rwa(np.array([0.5/2, 0.5/2, 0.5]))
rwa2 = rwa(np.array([0.5/2, -0.5/2, 0.5]))
rwa3 = rwa(np.array([-0.5/2, -0.5/2, 0.5]))
rwa4 = rwa(np.array([-0.5/2, 0.5/2, 0.5]))

B_wheel = np.array([rwa1.axis, rwa2.axis, rwa3.axis, rwa4.axis]).T # wheel Jacobian

### SOLAR RADIATION PRESSURE #################################################
P_0 = 1367 # [W/m^2]
c = 299792458
R_0 = 1 # AU
R_psyche = 2.92 # AU
P_srp = P_0/c*(R_0/R_psyche)**2

rho_a = 0.7 # absorption coefficient
rho_s = 0.2 # specular reflectivity coefficient
rho_d = 0.1 # diffusion reflectivity coefficient

tot_com = psyche.tot_com.reshape(3,)

# Do a little monte carlo to estimate the max torque for a particular orientation relative to the Sun
N = 2
torque_store = np.zeros((N, 3))
sun_store = np.zeros((N, 3))
for i in range(N):
    r_sun = adcs.unit_vec(np.array([np.sqrt(2)/2, np.sqrt(2)/2, np.sqrt(2)/2]))
    # r_sun = adcs.unit_vec(np.random.randn(3)) # nominally, in +Z direction np.array([0, 0, 1])
    # if r_sun[2] > 0:
    #     r_sun[2] = -r_sun[2] # force the sun to be under the spacecraft

    # Every component has .nvecx/y/z/1/2 in psyche_model.py, run for each surface
    # Every component also has a .com which is relative to the origin but the distance to the total COM is self.com - tot_com
    # All normal vectors are stored as array([[], [], []]) so convert back to ([])
    components = [psyche.bus, psyche.antenna1, psyche.antenna2, psyche.rod1L, psyche.rod2L, psyche.rod3L, psyche.rod4L, psyche.rod5L, psyche.panelmidL, psyche.panelouterL, psyche.panelinnerL, psyche.panelupL, psyche.paneldownL, psyche.rod1R, psyche.rod2R, psyche.rod3R, psyche.rod4R, psyche.rod5R, psyche.panelmidR, psyche.panelouterR, psyche.panelinnerR, psyche.panelupR, psyche.paneldownR]
    # components = [psyche.bus, psyche.panelmidL, psyche.panelmidR]
    F_srp = np.zeros(3)
    torque_tot = np.zeros(3)
    for i in range(len(components)):
        m = components[i].mass
        
        F_x1 = np.zeros(3)
        F_x2 = np.zeros(3)
        F_y1 = np.zeros(3)
        F_y2 = np.zeros(3)
        F_z1 = np.zeros(3)
        F_z2 = np.zeros(3)

        # X-faces first ###
        area = components[i].sax
        if components[i].nvecx1 is not None:
            normal_vec1 = components[i].nvecx1.reshape(3,)
            costheta1 = np.dot(normal_vec1, r_sun)
            if costheta1 > 0:
                F_x1 = -P_srp*area*costheta1*((1 - rho_s)*r_sun + 2*(rho_s*costheta1 + rho_d/3)*normal_vec1)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_x1)
                # print(torque_tot)
                # print(f"X1, {components[i]}")


        if components[i].nvecx2 is not None:
            normal_vec2 = components[i].nvecx2.reshape(3,)
            costheta2 = np.dot(normal_vec2, r_sun)
            if costheta2 > 0:
                F_x2 = -P_srp*area*costheta2*((1 - rho_s)*r_sun + 2*(rho_s*costheta2 + rho_d/3)*normal_vec2)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_x2)
                # print(torque_tot)
                # print(f'X2, {components[i]}')
        
        # Y-faces ###
        area = components[i].say
        if components[i].nvecy1 is not None:
            normal_vec1 = components[i].nvecy1.reshape(3,)
            costheta1 = np.dot(normal_vec1, r_sun)
            if costheta1 > 0:
                F_y1 = -P_srp*area*costheta1*((1 - rho_s)*r_sun + 2*(rho_s*costheta1 + rho_d/3)*normal_vec1)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_y1)
                # print(torque_tot)
                # print(f'Y1, {components[i]}')

        if components[i].nvecy2 is not None:
            normal_vec2 = components[i].nvecy2.reshape(3,)
            costheta2 = np.dot(normal_vec2, r_sun)
            if costheta2 > 0:
                F_y2 = -P_srp*area*costheta2*((1 - rho_s)*r_sun + 2*(rho_s*costheta2 + rho_d/3)*normal_vec2)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_y2)
                # print(torque_tot)
                # print(f'Y2, {components[i]}')

        # Z-faces ###
        if str(type(components[i])) == "<class 'psyche_model.FrustumObj'>": # Frustum has two different Z faces
            area = components[i].saz1
        else:
            area = components[i].saz
        if components[i].nvecz1 is not None:
            normal_vec1 = components[i].nvecz1.reshape(3,)
            costheta1 = np.dot(normal_vec1, r_sun)
            if costheta1 > 0:
                F_z1 = -P_srp*area*costheta1*((1 - rho_s)*r_sun + 2*(rho_s*costheta1 + rho_d/3)*normal_vec1)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_z1)
                # print(torque_tot)
                # print(f'Z1, {components[i]}')

        if str(type(components[i])) == "<class 'psyche_model.FrustumObj'>": # Frustum has two different Z faces
            area = components[i].saz2
        if components[i].nvecz2 is not None:
            normal_vec2 = components[i].nvecz2.reshape(3,)
            costheta2 = np.dot(normal_vec2, r_sun)
            if costheta2 > 0:
                F_z2 = -P_srp*area*costheta2*((1 - rho_s)*r_sun + 2*(rho_s*costheta2 + rho_d/3)*normal_vec2)
                torque_tot += np.cross(components[i].com.reshape(3,) - tot_com, F_z2)
                # print(torque_tot)
                # print(f'Z2, {components[i]}')

        # Total solar radiation pressure force is the sum of forces from all sides
        F_srp = F_x1 + F_x2 + F_y1 + F_y2 + F_z1 + F_z2 + F_srp

    # F = ma -> a = F/m
    a_srp = F_srp/psyche.tot_mass
    # torque_store[i, :] = torque_tot
    # sun_store[i, :] = r_sun

plt.figure()
plt.plot(range(N), torque_store[:, 0], label = '$\tau_x$')
plt.plot(range(N), torque_store[:, 1], label = '$\tau_y$')
plt.plot(range(N), torque_store[:, 2], label = '$\tau_z$')
plt.xlabel('Trial #')
plt.ylabel('Torque [N-m]')
plt.title('Monte Carlo for max torque induced by solar radiation')
# Find the spike
np.max(torque_store[:, 1])
ind = np.argmax(torque_store[:, 1])
max_sun = sun_store[ind] # is the unit vector
print(max_sun)

### GRAVITY GRADIENT TORQUE ##########################################
G = 6.67430e-11 # [m^3/kg/s^2]
mu_psyche = G*2.287e19 # Gravitational parameter [m^3/s^2]
J = psyche.tot_moicom
orbit_b = 303*1000
r0 = np.array([0, 0, orbit_b]) # km to [m]
v0 = np.array([0, np.sqrt(mu_psyche/orbit_b), 0])
state0 = np.concatenate([r0, v0])

def orbit(t, state):
    r = state[0:3]
    v = state[3:6]

    # translational motion
    r_norm = np.linalg.norm(r)
    a = -mu_psyche * r / r_norm**3

    return np.concatenate([v, a])

def orbit_srp(t, state):
    r = state[0:3]
    v = state[3:6]

    # translational motion
    r_norm = np.linalg.norm(r)
    a = -mu_psyche * r / r_norm**3

    if r[2] > -218000:
        a = a + a_srp

    return np.concatenate([v, a])

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

# simulation parameters
t0 = 0
tf = 2*np.pi*orbit_b**(3/2)/np.sqrt(mu_psyche) # [sec]
dt = 100

t, state = rk4(orbit, state0, t0, tf, dt)
r_vals = state[:,0:3]

tau_g = np.zeros((len(r_vals), 3))
tau_norm = np.zeros(len(r_vals))
for i in range(len(r_vals)):
    tau_g[i, :3] = np.cross(3*mu_psyche/(r_vals[i, :3].T@r_vals[i, :3])**(5/2)*r_vals[i, :3], J@r_vals[i, :3])
    tau_norm[i] = np.linalg.norm(tau_g)

np.max(tau_norm)
np.argmax(tau_norm)

plt.figure()
plt.plot(t/tf, tau_g[:, 0])
plt.xlabel('Time [fraction of orbit]')
plt.ylabel('Gravity gradient torque [Nm]')

t, state_srp = rk4(orbit_srp, state0, t0, tf, dt)
r_srp_vals = state_srp[:, 0:3]
r_norm = np.linalg.norm(r_srp_vals[:, :3], axis=1)

plt.figure()
plt.plot(t/(2*np.pi*orbit_b**(3/2)/np.sqrt(mu_psyche)), r_norm-orbit_b)
plt.xlabel('Time as Percentage of Orbit')
plt.ylabel('Change in Distance from Psyche Center [m]')

# Just to visually show the shadow-- plot not needed for computation
if False:
    theta = range(0, 360)
    y = np.zeros(360)
    x = np.zeros(360)
    for i in theta:
        y[i] = 111*1000*np.sin(np.deg2rad(theta[i]))
        x[i] = 111*1000*np.cos(np.deg2rad(theta[i]))

    plt.figure()
    plt.plot(r_vals[:, 1], r_vals[:, 2], color = 'green', linestyle = '--')
    plt.plot(y, x, color = 'grey')
    plt.axhline(-281*1000, linestyle = '--', color = 'red')
    plt.axvline(111*1000, 0, 0.5, color = 'black')
    plt.axvline(-111*1000, 0, 0.5, color = 'black')
    plt.axis('square')

