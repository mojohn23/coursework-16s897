import math
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import adcs_toolbox as adcs

H = np.vstack([np.zeros((1, 3)), np.eye(3)])

##### Part 4: Wahba's Problem: Wahba's Solver Functions ##################################
S_SRU = np.array([[4.85e-6**2, 0, 0], [0, 4.85e-6**2, 0], [0, 0, 3.88e-5**2]])
S_CSS = np.array([[0.017**2, 0, 0], [0, 0.017**2, 0], [0, 0, 0.017**2]])
sensors = 4 # Assume 2 SRU measurements and 2 CSS measurements
w = np.array([4.85e-6**-2, 4.85e-6**-2, 0.017**-2, 0.017**-2]) # Initialize weights
S = [S_SRU, S_SRU, S_CSS, S_CSS]

def random_sensor(n_SRU, n_CSS):
    # Generate random "ground truth" attitude
    q_real = adcs.qexp(np.random.randn(3)) # Use randn (normal distribution) not rand!!
    Q_real = adcs.Q(q_real)

    r_B = np.zeros((3, n_CSS + n_SRU))
    r_N_real = np.zeros((3, n_CSS + n_SRU))

    # Generate random SRU measurements: multiplicative error
    for i in range(n_SRU):
        r_B[:, i] = adcs.unit_vec(np.random.randn(3)) # Generate random unit vector in the body frame
        r_N_real[:, i] = adcs.unit_vec(Q_real.T@r_B[:, i]) # Transform real vector to inertial frame, real inertial frame
        noise_vec = np.random.multivariate_normal(np.zeros(3), S_SRU) # Sample noise vector as 3-param
        delta_q = adcs.qexp(noise_vec) # Turn into noise quaternion
        q_noisy = adcs.qmult(q_real, delta_q) # Multiply the error to get the noisy quaternion
        r_B[:, i] = adcs.unit_vec(adcs.Q(q_noisy).T@r_N_real[:, i]) # With noisy quaternion, produce noisy measurement in the body frame

# Generate random CSS measurements: additive error
    for i in range (n_CSS):
        r_B[:, n_SRU + i] = adcs.unit_vec(np.random.randn(3)) # Generate random unit vector in the body frame
        r_N_real[:, n_SRU + i] = adcs.unit_vec(Q_real.T@r_B[:, n_SRU + i]) # Transform real vector to inertial frame, real inertial frame
        r_B[:, n_SRU + i] = adcs.unit_vec(r_B[:, n_SRU + i] + np.random.multivariate_normal(np.zeros(3), S_CSS)) # r_B = r_B_real + noise, overwrite r_B and normalize

    return r_B, r_N_real, Q_real, q_real

# #Convex Relaxation Method
# # Abandoned because this requires extra optimization packages that idk how to use
# def wahba_convex(w, r_B, r_N_real)
    # B_cr = np.zeros((3, 3)) # Initialize attitude profile matrix
    # for i in range(sensors):
    #     B_cr = B_cr + w[i]*r_B[:, i].reshape(3, 1)*r_N_real[:, i]

# SVD Method
def wahba_svd(r_B, r_N_real):
    B_svd = np.zeros((3, 3)) # Initialize attitude profile matrix
    for i in range(sensors):
        B_svd = B_svd + w[i]*r_B[:, i].reshape(3, 1)*r_N_real[:, i]
    U, S, Vh = np.linalg.svd(B_svd) # Look at numpy documentation- Vh is Hermitian V so V = Vh.conj().T
    if np.linalg.det(B_svd) < 0:
        Q_svd = Vh.conj().T@np.diag([1, 1, np.linalg.det(Vh.conj().T@U.T)])@U.T
    else:
        Q_svd = Vh.conj().T@U.T
    q_svd = adcs.q(Q_svd)
    return Q_svd, q_svd

# Davenport q Method
def wahba_davenport(r_B, r_N_real):
    D = np.zeros((4, 4)) # Initialize Davenport matrix
    for i in range(sensors):
        D = D + w[i]*adcs.L(adcs.H@r_N_real[:, i]).T@adcs.R(adcs.H@r_B[:, i])
    val, vec = np.linalg.eigh(D)
    q_dav = vec[:, -1]
    Q_dav = adcs.Q(q_dav)
    return Q_dav, q_dav

# Gauss-Newton Method
def wahba_gn(r_B, r_N_real):
    tol = 1e-6
    count = 0
    q_gn = adcs.qexp(np.random.randn(3)) # Initialize first guess
    phi = np.ones(3) # Initialize first phi to start loop

    while np.max(np.abs(phi)) > tol and count < 20:
        s = residual(q_gn) # (3n, )
        ds = dsdq(residual, q_gn, tol) # (3n, n)
        grads = ds@adcs.G(q_gn) # (3n, 3)
        phi = -np.linalg.solve(grads.T@grads, grads.T@s) # (3, )
        q_new = np.concatenate([[np.sqrt(1 - phi.T@phi)], phi]) # (4, )
        q_gn = adcs.qmult(q_gn, q_new) # (4, )
        count = count + 1
    Q_gn = adcs.Q(q_gn)
    return Q_gn, q_gn

def residual(q_gn):
    return (w*(r_N_real - adcs.Q(q_gn)@r_B)).ravel()

def dsdq(residual, q_gn, tol):
    m = len(q_gn)
    f0 = residual(q_gn)
    jacob = np.zeros((len(f0), m))
    for i in range(m):
        dx = np.zeros(m)
        dx[i] = tol
        jacob[:, i] = (residual(q_gn + dx) - residual(q_gn - dx)) / (2 * tol)
    return jacob
    
def Qdiff(Q_computed, Q_real):
    # Find the difference in degrees between two rotation matrices
    return (180/math.pi)*np.linalg.norm(adcs.unhat(scipy.linalg.logm(Q_computed.T@Q_real)))
 
##### Part 3: Checking Attitude Sensor Covariance Matrix Statistics ################
if True:
    trials = 2000
    error_SRU = np.zeros((trials, 3))
    error_CSS = np.zeros((trials, 3))

    # Generate a random vector, add noise and store the error difference
    for i in range(trials):
    ### CSS measurements ###
        r_B_real = np.random.randn(3) # Generate random 3-D vector
        r_B_real = adcs.unit_vec(r_B_real) # Normalize the true body frame vector

        r_B_CSS = r_B_real + np.random.multivariate_normal(np.zeros(3), S_CSS) # Add sampled noise
        r_B_CSS = adcs.unit_vec(r_B_CSS) # Normalize again
        error_CSS[i] = r_B_CSS - r_B_real

    ### SRU measurements ###
        # q_real = np.random.randn(4) # Generate random quaternion
        # q_real = q_real/np.linalg.norm(q_real) # Normalize to unit quaternion
        q_real = adcs.qexp(r_B_real) # Use the same random vector as before, but cast to quaternion. Can also just generate a new random quaternion in the lines above but idk

        noise_vec = np.random.multivariate_normal(np.zeros(3), S_SRU) # This is a 3-parameter rotation
        delta_q = adcs.qexp(noise_vec) # Turn the noise vector to a noise delta quaternion
        q_SRU = adcs.qmult(q_real, delta_q) # Quaternion multiply on the error
        q_error = adcs.qmult(q_SRU, adcs.qinv(q_real))
        error_SRU[i] = 2*q_error[1:]
    
    covar_SRU = np.cov(error_SRU.T) # should be close to S_SRU
    covar_CSS = np.cov(error_CSS.T) # should be close to S_CSS

    print(covar_SRU - S_SRU)
    print(covar_CSS - S_CSS)

##### Part 4: Wahba's Problem: Monte Carlo Methods ##################################
if True:
    trials = 100 # Number of runs
    t_svd = np.zeros(trials)
    t_dav = np.zeros(trials)
    t_gn = np.zeros(trials)

    error_svd = np.zeros(trials)
    error_dav = np.zeros(trials)
    error_gn = np.zeros(trials)

    for i in range(trials):
        r_B, r_N_real, Q_real, q_real = random_sensor(2, 2)

        start_svd = time.perf_counter()
        Q_svd, q_svd = wahba_svd(r_B, r_N_real)
        t_svd[i] = time.perf_counter() - start_svd

        start_dav = time.perf_counter()
        Q_dav, q_dav = wahba_davenport(r_B, r_N_real)
        t_dav[i] = time.perf_counter() - start_dav

        start_gn = time.perf_counter()
        Q_gn, q_gn = wahba_gn(r_B, r_N_real)
        t_gn[i] = time.perf_counter() - start_gn

        error_svd[i] = Qdiff(Q_svd, Q_real)
        error_dav[i] = Qdiff(Q_dav, Q_real)
        error_gn[i] = Qdiff(Q_gn, Q_real)

    # Make a graph of runtime vs. error with the optimal point at min(time), min(error)
    fig, ax = plt.subplots()
    ax.plot(t_svd, error_svd, 'o', color = 'blue', label = 'SVD')
    ax.plot(t_dav, error_dav, 'o', color = 'red', label = 'Davenport q Method')
    ax.plot(t_gn, error_gn, 'o', color = 'green', label = 'Gauss-Newton Method')
    ax.plot(0, 0, '*', color = 'black', label = 'Optimal Point')
    plt.legend()
    plt.xlabel('Runtime (s)')
    plt.ylabel('Error (deg)')
    plt.title('Comparison of Monte Carlo Runs for Wahba Solvers')