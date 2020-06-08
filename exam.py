from collections import Counter
from math import exp
import numpy as np

def find_market_portfolio(mu, C, risk_free_rate):
    assert(C.shape[0] == C.shape[1])
    n = C.shape[0]
    C_inv = np.linalg.inv(C)
    u = np.ones((1, n))
    return (mu - risk_free_rate * u).dot(C_inv) / ((mu - risk_free_rate * u).dot(C_inv).dot(u.T))

def mean_and_variance_of_a_portfolio(weight, mean_vector, cov_matrix):
    """
    Input are all numpy arrays
    """
    # Reshape as row vectors
    weight = weight.reshape(1, -1)
    mean_vector = mean_vector.reshape(1, -1)

    assert(cov_matrix.shape[0] == cov_matrix.shape[1])

    mu = mean_vector.dot(weight.T)
    var = weight.dot(cov_matrix).dot(weight.T)
    return mu, var

mu = np.array([0.18, 0.22])
C = np.array([[0.0225, -7.5e-3], [-7.5e-3, 0.01]])
MP=find_market_portfolio(mu, C, 0.1)
print(MP)
print(mean_and_variance_of_a_portfolio(MP, mu, C))

def find_min_var_portfolio(C):
    """
    Input is an nxn covariance matrix
    """
    assert(C.shape[0] == C.shape[1])
    n = C.shape[0]
    C_inv = np.linalg.inv(C)
    u = np.ones((1, n))

    return u.dot(C_inv) / (u.dot(C_inv).dot(u.T))

print(find_min_var_portfolio(C))