#!/bin/python3
import numpy as np
from collections import defaultdict as dd


def expectation(M):
    mu = sum([k*p for k, p in M.items()])
    return mu


def stddev(M):
    mu = expectation(M)
    sigma = sum([(k-mu) ** 2 * p for k, p in M.items()]) ** 0.5
    return sigma


def return_and_risk(K):
    """
    K: a map of return -> probability
    """
    mu = expectation(K)
    sigma = stddev(K)

    return mu, sigma


def covariance(M):
    """
    M: a map from a tuple (X1, X2) -> probability
    """

    # Sanity check: probability sums to 1
    assert(sum([p for _, p in M.items()]) == 1.0)
    X1_map = dd(float)
    X2_map = dd(float)

    for (x1, x2), p in M.items():
        X1_map[x1] += p
        X2_map[x2] += p

    E_X1 = expectation(X1_map)
    E_X2 = expectation(X2_map)

    cov = sum([(x1 - E_X1) * (x2 - E_X2) * p for (x1, x2), p in M.items()])
    return cov


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

def find_min_var_portfolio(C):
    """
    Input is an nxn covariance matrix
    """
    assert(C.shape[0] == C.shape[1])
    n = C.shape[0]
    C_inv = np.linalg.inv(C)
    u = np.ones((1, n))

    return u.dot(C_inv) / (u.dot(C_inv).dot(u.T))

def find_market_portfolio(mu, C, risk_free_rate):
    assert(C.shape[0] == C.shape[1])
    n = C.shape[0]
    C_inv = np.linalg.inv(C)
    u = np.ones((1, n))
    return (mu - risk_free_rate * u).dot(C_inv) / ((mu - risk_free_rate * u).dot(C_inv).dot(u.T))

def cov_matrix_from_std_and_corr(std, corr):
    """
    std : map from stock index -> risk of stock
    corr : map from (stock 1 index, stock 2 index) -> stock
    """
    n = len(std)
    cov_mat = np.zeros((n,n))
    for (i,j), _corr in list(corr.items()):
        corr[j,i] = _corr
    for i in range(n):
        cov_mat[i,i] = std[i] * std[i]
        for j in range(i+1,n):
            cov_mat[i,j] = corr[i,j] * std[i] * std[j]
            cov_mat[j,i] = cov_mat[i,j]
    
    return cov_mat

def find_efficient_frontier(mu, C):\
    # Symmetric computations of A & B
    C_inv = np.linalg.inv(C)
    u = np.ones((C.shape[0]))
    M = np.array([
        [u.dot(C_inv).dot(u.T), mu.dot(C_inv).dot(u.T)],
        [u.dot(C_inv).dot(mu.T), mu.dot(C_inv).dot(mu.T)]
    ])
    def subroutine(l, r):
        L = np.dot(l.dot(C_inv).dot(l.T),r).dot(C_inv)
        R = np.dot(l.dot(C_inv).dot(r.T),l).dot(C_inv)
        return 1 / np.linalg.det(M) * (L-R)
    
    A, B = subroutine(u, mu), subroutine(mu, u)
    return A, B



""" Problem 1(a) """
K_M = {
    0.15: 0.5,
    -0.15: 0.25,
    0: 0.25
}
print(return_and_risk(K_M))
K_T = {
    0.15: 0.25,
    -0.15: 0.25,
    -0.05: 0.25,
    0.1: 0.25
}
print(return_and_risk(K_T))
K = {
    (0.15, 0.15): 0.25,
    (-0.15, -0.15): 0.25,
    (0.15, -0.05): 0.25,
    (0, 0.1): 0.25,
}
print(covariance(K))

""" Problem 1(b) """
w = np.array([80*112.7/17000, 3*1550 / 17000, (17000-80*112.7-3*1550)/17000])
mu = np.array([0.0375, 0.0125, 0])
omega = np.array([
    [0.12437342963832748**2, 0.00890625, 0],
    [0.00890625, 0.11924240017711821**2, 0],
    [0,0,0]
])

print(mean_and_variance_of_a_portfolio(w, mu, omega))

""" Problem 1(c) """
C = np.array([
    [0.12437342963832748**2, 0.00890625],
    [0.00890625, 0.11924240017711821**2]
])
print(find_min_var_portfolio(C))

""" Problem 4(a) """
sigma_NAC = 0.05
sigma_NOI = 0.11
corr = 0.1
mu = np.array([0.09, 0.15])
C = np.array([
    [sigma_NAC ** 2, sigma_NAC * sigma_NOI * corr],
    [sigma_NAC * sigma_NOI * corr, sigma_NOI ** 2]
])
risk_free_rate = 0.02

print(find_market_portfolio(mu, C, risk_free_rate))

""" Problem 5(a) """
std = {0:0.08, 1: 0.1, 2:0.13, 3:0.1, 4:0.1, 5:0.08}
corr = {(0,1):0.5, (0,2):0.3, (0,3):0, (0,4):0, (0,5):0,
        (1,2):0.4, (1,3):-0.1, (1,4):-0.2, (1,5):-0.2,
        (2,3):0, (2,4):0, (2,5):0.1,
        (3,4):0.2, (3,5):0.8,
        (4,5):0.1}
cov_mat = cov_matrix_from_std_and_corr(std, corr)
print(cov_mat)

""" Problem 5(b) """
MVP = find_min_var_portfolio(cov_mat)
print(MVP)
mu = np.array([0.08, 0.12, 0.15, 0.06, 0.20, 0.10])
print(mean_and_variance_of_a_portfolio(MVP, mu, cov_mat))

""" Problem 5(c) """
MP = find_market_portfolio(mu, cov_mat, 0.03)
print(MP)
print(mean_and_variance_of_a_portfolio(MP, mu, cov_mat))

""" Problem 5(d) """
print(find_efficient_frontier(mu, cov_mat))