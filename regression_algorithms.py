import numpy as np
from cvxopt import matrix, solvers


def least_squares(phi, y):
    """
    Least-squares (LS).
    """
    phi = np.transpose(phi)
    theta_left = np.linalg.inv(np.matmul(phi, np.transpose(phi)))
    theta_right = np.matmul(phi, y)
    theta = np.matmul(theta_left, theta_right)
    return np.ndarray.flatten(np.array(theta))


def regularized_ls(phi, y, lamda):
    """
    Regularized LS (RLS).
    """
    phi = np.transpose(phi)
    theta_left = np.linalg.inv(np.matmul(phi, np.transpose(phi)) 
        + lamda * np.matlib.identity(len(phi)))
    theta_right = np.matmul(phi, y)
    theta = np.matmul(theta_left, theta_right)
    return np.ndarray.flatten(np.array(theta))


def l1_regularized_ls(phi, y, lamda):
    """
    L1-regularized LS (LASSO).
    """
    phi = np.transpose(phi)
    phi_phi = np.matmul(phi, np.transpose(phi))
    phi_y = np.matmul(phi, y)
    P = matrix(np.concatenate((
        np.concatenate((phi_phi, - phi_phi)), 
        np.concatenate((- phi_phi, phi_phi))
        ), axis = 1))
    q = matrix(lamda * np.ones([1, 2 * len(phi)]) 
        - np.concatenate((phi_y, - phi_y)))
    G = matrix(- np.matlib.identity(2 * len(phi)))
    h = matrix(np.zeros([1, 2 * len(phi)]))
    sol = solvers.qp(P, q.T, G, h.T)
    theta_plus = sol["x"][: len(phi)]
    theta_minus = sol["x"][len(phi) :]
    theta = theta_plus - theta_minus
    return np.ndarray.flatten(np.array(theta))


def robust_regression(phi, y):
    """
    Robust regression (RR).
    """
    phi = np.transpose(phi)
    c = matrix(np.concatenate((np.zeros([1, len(phi)]), 
        np.ones([1, len(y)])), axis = 1))
    id_mat = - np.matlib.identity(len(y))
    G = matrix(np.concatenate((
        np.concatenate((- np.transpose(phi), np.transpose(phi))), 
        np.concatenate((id_mat, id_mat))
        ), axis = 1))
    h = matrix(np.concatenate((- y, y)))
    sol = solvers.lp(c.T, G, h)
    theta = sol["x"][: len(phi)]
    return np.ndarray.flatten(np.array(theta))
