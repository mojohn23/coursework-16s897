import math
import numpy as np

# A general repository of functions useful for ADCS in the Hamiltonian convention [s, v]

H = np.vstack([np.zeros((1, 3)), np.eye(3)])
Tmat = np.block([[np.array([[1]]), np.zeros((1, 3))], [np.zeros((3, 1)), -np.eye(3)]])

def rot_simple(angle: float, axis: str, unit:str = 'deg'):
    # Input by default assumes degrees, specify otherwise
    axis = axis.upper()
    unit = unit.upper()

    # Need to convert from degrees to rad for math trig functions
    if unit in ('DEG', 'DEGREE', 'DEGREES'):
        theta = math.radians(angle)
    elif unit in ('RAD', 'RADIAN', 'RADIANS'):
        theta = angle
    else:
        raise ValueError('Error: Unit must be deg or rad, deg by default')

    c = math.cos(theta)
    s = math.sin(theta)

    if axis == 'X':
        rot_mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'Y':
        rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'Z':
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError('Error: Axis must be X, Y, or Z')
    return rot_mat

def quat(r, theta: float):
    # r is a 1x3 array, theta must be in rad
    s = np.array([math.cos(theta)])
    v = math.sin(theta)*r
    q = np.hstack((s, v))
    return q

def qinv(q):
    # q is a 1x4 horizontal quaternion with [s, v]
    return Tmat @ q

def qconj(q):
    # q is a 1x4 horizontal quaternion with [s, v]
    return np.diag([1, -1, -1, -1])@q

def qmult(q1, q2):
    # q1 and q2 must be 1x4 arrays
    return L(q1) @ q2

def qexp(phi):
    # phi is a 1x3 array
    theta = np.linalg.norm(phi)
    s = math.cos(theta)
    v = phi*np.sinc(theta/math.pi) # sinc is in the np library
    # Return a 1x4 quaternion
    return np.hstack([[s], v])

def qlog(q):
    # q is a 1x4 horizontal quaternion [s, v]
    s = q[0]
    v = q[1:4]
    theta = np.arccos(np.clip(s, -1, 1))
    phi = theta*unit_vec(v)
    return phi

def hat(x):
    # x is a 1x3 horizontal vector, as standard np.array
    if x.ndim > 1 or x.size != 3:
        raise ValueError('In hat(x), x should be a 1x3 horizontal vector')
    x = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    # x returns as a 3x3 matrix
    return x

def unhat(x):
    # x is a 3x3 skew-symmetric matrix
    return 1/2 * np.array([x[2, 1] - x[1, 2], x[0, 2] - x[2, 0], x[1, 0] - x[0, 1]])

def L(q):
    # q is a 1x4 horizontal quaternion with [s, v]
    if q.ndim > 1 or q.size != 4:
        raise ValueError('In L(q), q should be a 1x4 horizontal scalar-first quaternion')
    s = q[0]
    v = q[1:4]
    q = np.block([[np.array([[s]]), -v.reshape(1, 3)], [v.reshape(3, 1), s*np.eye(3) + hat(v)]])
    # q returns as a 4x4 matrix
    return q

def R(q):
    # q is a 1x4 horizontal quaternion with [s, v]
    if q.ndim > 1 or q.size != 4:
        raise ValueError('In R(q), q should be a 1x4 horizontal scalar-first quaternion')
    s = q[0]
    v = q[1:4]
    q = np.block([[np.array([[s]]), -v.reshape(1, 3)], [v.reshape(3, 1), s*np.eye(3) - hat(v)]])
    # q returns as a 4x4 matrix
    return q

def G(q):
    # q is a 1x4 horizontal quaternion with [s, v]
    # Returns a 4x3
    return L(q)@H

def Q(q):
    # From quaternion to rotation matrix
    # q is a 1x4 horizontal quaternion with [s, v]
    # Returns a 4x3
    return H.T @ L(q) @ R(q).T @ H

def T(r1, r2):
    # r1 and r2 should be 1x3 arrays
    t1 = r1
    t2 = np.cross(r1, r2)/np.linalg.norm(np.cross(r1, r2))
    t3 = np.cross(t1, t2)/np.linalg.norm(np.cross(t1, t2))
    T = np.array([t1, t2, t3])
    # Returns a 3x3
    return T.T

def unit_vec(x):
    # x is a 1x3 array
    return x/np.linalg.norm(x)

def q(Q):
    # Get quaternion from rotation matrix
    r11, r12, r13 = Q[0, 0], Q[0, 1], Q[0, 2]
    r21, r22, r23 = Q[1, 0], Q[1, 1], Q[1, 2]
    r31, r32, r33 = Q[2, 0], Q[2, 1], Q[2, 2]
    
    # Calculate quaternion components
    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)
    q1 = 0.5 * np.sqrt(1 + r11 - r22 - r33) * np.sign(r32 - r23)
    q2 = 0.5 * np.sqrt(1 - r11 + r22 - r33) * np.sign(r13 - r31)
    q3 = 0.5 * np.sqrt(1 - r11 - r22 + r33) * np.sign(r21 - r12)
    
    return np.array([q0, q1, q2, q3])