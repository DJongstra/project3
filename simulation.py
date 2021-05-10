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

# TODO vertainty equivalent

def welfareScore(CiT, Xi, T):
    """
    welfare: average of the realized values
    """
    welfare_score = 0.0
    for i in range(len(CiT)):
        welfare_score += Xi[CiT[i]]
    return welfare_score / T  # = 1/T * sum_(n in Cit) (Xi,n)


def initSigma(CiT, NiT, sigma_Xi):
    """"
    partition covariance matrix
    """
    sigma11 = np.zeros((len(CiT), len(CiT)))
    sigma12 = np.zeros((len(CiT), len(NiT)))
    sigma21 = np.zeros((len(NiT), len(CiT)))
    sigma22 = np.zeros((len(NiT), len(NiT)))

    for i in range(len(CiT)):
        for j in range(len(CiT)):
            sigma11[i][j] = sigma_Xi[CiT[i]][CiT[j]]

        for j in range(len(NiT)):
            sigma21[j][i] = sigma_Xi[CiT[i]][NiT[j]]


    for i in range(len(NiT)):
        for j in range(len(CiT)):
            sigma12[j][i] = sigma_Xi[NiT[i]][CiT[j]]

        for j in range(len(NiT)):
            sigma22[i][j] = sigma_Xi[NiT[i]][NiT[j]]    # known part

    return sigma11, sigma12, sigma21, sigma22




