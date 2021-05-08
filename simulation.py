import numpy as np


def distance(item1, item2, N):
    """"
    returns the (shortest) distance between item1 and item2 if there is a total of N items in the simulation set
    """
    return min(abs(item2-item1), N - abs(item2-item1))


def createCovarianceMatrix(sigma, rho, N):
    """
    Creates the covariance matrix
    """
    cov_matrix = np.zeros((N, N))
    for i in range(N):  #TODO if problems arise, other sim counts 1->N instead of 0->N-1
        for j in range(N):
            dist = distance(i, j, N)
            cov_matrix[i][j] = sigma * (rho ** dist)    # = sigma * rho^dist

    return cov_matrix

