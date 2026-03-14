import math
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import adcs_toolbox as adcs

##### Part 4: Wahba's Problem: Wahba's Solver Functions ##################################
S_SRU = np.array([[4.85e-6**2, 0, 0], [0, 4.85e-6**2, 0], [0, 0, 3.88e-5**2]])
S_CSS = np.array([[0.017**2, 0, 0], [0, 0.017**2, 0], [0, 0, 0.017**2]])
sensors = 4 # Assume 2 SRU measurements and 2 CSS measurements
w = np.array([4.85e-6**-2, 4.85e-6**-2, 0.017**-2, 0.017**-2]) # Initialize weights
S = [S_SRU, S_SRU, S_CSS, S_CSS]

def random_sensor(sensors):
    # Generate random "ground truth" attitude
    q_real = adcs.qexp(np.random.randn(3)) # Use randn (normal distribution) not rand!!
    Q_real = adcs.Q(q_real)

    # Generate random "sensor readings" in the body frame and normalize as unit vec
    r_B = np.random.randn(3, sensors)
    for i in range(sensors):
        r_B[:, i] = adcs.unit_vec(r_B[:, i])
    r_N_real = Q_real @ r_B # Perfect sensor readings in the inertial frame

    # Add noise to the sensor readings in the body frame and renormalize as unit vec
    for i in range(sensors):
        r_B[:, i] = r_B[:, i] + np.random.multivariate_normal(np.zeros(3), S[i])
        r_B[:, i] = adcs.unit_vec(r_B[:, i])
    # r_B becomes the noisy sensor readings in the body frame
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
    val, vec = np.linalg.eig(D)
    q_dav = vec[:, np.argmax(val)]
    Q_dav = adcs.Q(q_dav)
    return Q_dav, q_dav

# Gauss-Newton Method
def wahba_gn():
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

def qdiff(q_computed, q_real):
    # Find the difference between two quaternions
    if np.sign(q_real[0]) == np.sign(q_computed[0]):
    # Same sign, subtract
        return q_real - q_computed
    else:
    # Opposite signs, add
        return q_real + q_computed
    
def Qdiff(Q_computed, Q_real):
    # Find the difference in degrees between two rotation matrices
    return (180/math.pi)*np.linalg.norm(adcs.unhat(scipy.linalg.logm(Q_computed.T@Q_real)))
 

##### Part 3: Checking Attitude Sensor Covariance Matrix Statistics ################
if False:
    trials = 100
    error_SRU = np.zeros((trials, 3))
    error_CSS = np.zeros((trials, 3))

    # Generate a random vector, add noise store the error difference
    for i in range(trials):
        r_B_true = np.random.randn(3)

        r_B_SRU = r_B_true + np.random.multivariate_normal(np.zeros(3), S_SRU)
        r_B_CSS = r_B_true + np.random.multivariate_normal(np.zeros(3), S_CSS)

        error_SRU[i] = r_B_SRU - r_B_true
        error_CSS[i] = r_B_CSS - r_B_true
    
    covar_SRU = np.cov(error_SRU.T) # should be close to S_SRU
    covar_CSS = np.cov(error_CSS.T) # should be close to S_CSS

##### Part 4: Wahba's Problem: Monte Carlo Methods ##################################
if True:
    trials = 30 # Number of runs
    t_svd = np.zeros(trials)
    t_dav = np.zeros(trials)
    t_gn = np.zeros(trials)

    error_svd = np.zeros(trials)
    error_dav = np.zeros(trials)
    error_gn = np.zeros(trials)

    for i in range(trials):
        r_B, r_N_real, Q_real, q_real = random_sensor(sensors)

        start_svd = time.perf_counter()
        Q_svd, q_svd = wahba_svd(r_B, r_N_real)
        t_svd[i] = time.perf_counter() - start_svd

        start_dav = time.perf_counter()
        Q_dav, q_dav = wahba_svd(r_B, r_N_real)
        t_dav[i] = time.perf_counter() - start_dav

        start_gn = time.perf_counter()
        Q_gn, q_gn = wahba_gn()
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