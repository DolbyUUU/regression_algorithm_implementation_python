import numpy as np
from cvxopt import matrix, solvers


def least_squares(phi, y):
    """
    Solves the least-squares (LS) problem using the normal equations.
    
    Parameters:
        phi (ndarray): The design matrix with dimensions (m, n).
        y (ndarray): The target values with dimensions (m,).
    
    Returns:
        ndarray: The estimated parameter vector with dimensions (n,).
    """
    # Transpose the design matrix.
    phi = np.transpose(phi)
    
    # Compute the left and right hand sides of the normal equations.
    theta_left = np.linalg.inv(np.matmul(phi, np.transpose(phi)))
    theta_right = np.matmul(phi, y)
    
    # Compute the estimated parameter vector.
    theta = np.matmul(theta_left, theta_right)
    
    # Flatten and return the parameter vector.
    return np.ndarray.flatten(np.array(theta))


def regularized_ls(phi, y, lamda):
    """
    Solves the regularized least-squares (RLS) problem using ridge regression.
    
    Parameters:
        phi (ndarray): The design matrix with dimensions (m, n).
        y (ndarray): The target values with dimensions (m,).
        lamda (float): The regularization parameter.
    
    Returns:
        ndarray: The estimated parameter vector with dimensions (n,).
    """
    # Transpose the design matrix.
    phi = np.transpose(phi)
    
    # Compute the left and right hand sides of the ridge regression problem.
    theta_left = np.linalg.inv(np.matmul(phi, np.transpose(phi)) 
        + lamda * np.matlib.identity(len(phi)))
    theta_right = np.matmul(phi, y)
    
    # Compute the estimated parameter vector.
    theta = np.matmul(theta_left, theta_right)
    
    # Flatten and return the parameter vector.
    return np.ndarray.flatten(np.array(theta))


def l1_regularized_ls(phi, y, lamda):
    """
    Solves the L1-regularized least-squares (LASSO) problem using linear programming.
    
    Parameters:
        phi (ndarray): The design matrix with dimensions (m, n).
        y (ndarray): The target values with dimensions (m,).
        lamda (float): The regularization parameter.
    
    Returns:
        ndarray: The estimated parameter vector with dimensions (n,).
    """
    # Transpose the design matrix.
    phi = np.transpose(phi)
    
    # Compute the left and right hand sides of the LASSO problem.
    phi_phi = np.matmul(phi, np.transpose(phi))
    phi_y = np.matmul(phi, y)
    
    # Construct the optimization problem using the CVXOPT library.
    P = matrix(np.concatenate((
        np.concatenate((phi_phi, - phi_phi)), 
        np.concatenate((- phi_phi, phi_phi))
        ), axis = 1))
    q = matrix(lamda * np.ones([1, 2 * len(phi)]) 
        - np.concatenate((phi_y, - phi_y)))
    G = matrix(- np.matlib.identity(2 * len(phi)))
    h = matrix(np.zeros([1, 2 * len(phi)]))
    
    # Solve the optimization problem.
    sol = solvers.qp(P, q.T, G, h.T)
    
    # Extract the estimated parameter vector.
    theta_plus = sol["x"][: len(phi)]
    theta_minus = sol["x"][len(phi) :]
    theta = theta_plus - theta_minus
    
    # Flatten and return the parameter vector.
    return np.ndarray.flatten(np.array(theta))


def robust_regression(phi, y):
    """
    Robust regression (RR).

    Solves the following optimization problem:
    min ||x||_1
    subject to y = phi' x

    Parameters:
    phi (np.array): Design matrix of shape (N, M)
    y (np.array): Target values of shape (N,)

    Returns:
    x (np.array): Estimated coefficients of shape (M,)
    """
    # Transpose the design matrix
    phi = np.transpose(phi)

    # Construct the optimization problem using cvxopt library
    c = matrix(np.concatenate((np.zeros([1, len(phi)]), 
        np.ones([1, len(y)])), axis=1))
    id_mat = - np.matlib.identity(len(y))
    G = matrix(np.concatenate((
        np.concatenate((- np.transpose(phi), np.transpose(phi))), 
        np.concatenate((id_mat, id_mat))
        ), axis=1))
    h = matrix(np.concatenate((- y, y)))
    sol = solvers.lp(c.T, G, h)

    # Extract the estimated coefficients from the solution
    x = sol["x"][: len(phi)]

    # Flatten the array and return
    return np.ndarray.flatten(np.array(x))
