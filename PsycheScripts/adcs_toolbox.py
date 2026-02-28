import math
import numpy as np

# A general repository of functions useful for ADCS

H = np.vstack([np.zeros((1, 3)), np.eye(3)])
T = np.block([[np.array([[1]]), np.zeros((1, 3))], [np.zeros((3, 1)), -np.eye(3)]])

def hat(x):
    # x is a 1x3 horizontal vector, as standard np.array
    if x.ndim > 1 or x.size != 3:
        raise ValueError('In hat(x), x should be a 1x3 horizontal vector')
    x = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    # x returns as a 3x3 matrix
    return x

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
    return L(q)@H

def Q(q):
    return H.T @ L(q) @ R(q).T @ H