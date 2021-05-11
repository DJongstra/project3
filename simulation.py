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

def welfareScore(CiT, Ui, T):
    """
    welfare: average of the realized values
    """
    welfare_score = 0.0
    for i in range(len(CiT)):
        welfare_score += Ui[CiT[i]]
    return welfare_score / T  # = 1/T * sum_(n in Cit) (Ui,n)


def initSigma(CiT, NiT, sigma_Ui):
    """"
    partition covariance matrix
    """
    sigma11 = np.zeros((len(CiT), len(CiT)))
    sigma12 = np.zeros((len(CiT), len(NiT)))
    sigma21 = np.zeros((len(NiT), len(CiT)))
    sigma22 = np.zeros((len(NiT), len(NiT)))

    for i in range(len(CiT)):
        for j in range(len(CiT)):
            sigma11[i][j] = sigma_Ui[CiT[i]][CiT[j]]

        for j in range(len(NiT)):
            sigma21[j][i] = sigma_Ui[CiT[i]][NiT[j]]


    for i in range(len(NiT)):
        for j in range(len(CiT)):
            sigma12[j][i] = sigma_Ui[NiT[i]][CiT[j]]

        for j in range(len(NiT)):
            sigma22[i][j] = sigma_Ui[NiT[i]][NiT[j]]    # known part

    return sigma11, sigma12, sigma21, sigma22

#get_mubar_sigmamu is split up into different functions

def muBar(sigma_Ui, Ui, x1, sigma11, sigma12, sigma21, sigma22, mu_t, mu_Nt):
    """
    :param Ui: utility
    :param x1: consumed items? #TODO double check
    :param sigma11: split up covariance matrix
    :param sigma21: split up covariance matrix
    :param mu_t: initial mean beliefs user has over items in Ct
    :param mu_Nt: initial mean beliefs user has over items not in Ct
    :return: muBar, used for the resulting beliefs over the remaining items after item consumption
    """
    #TODO delete all params that are not used?
    Ct = np.array([Ui[n] for n in x1])  # get consumed items
    inverse = np.linalg.inv(sigma11)
    innerMulti = sigma21 * inverse
    muBar = mu_Nt + innerMulti * (Ct - mu_t)   # mu_N-t + sigma(N-t,t)*sigma^-1(t,t) * (c_t - mu_t)

    return muBar


def sigmaBar(sigma_Ui, Ui, x1, sigma11, sigma12, sigma21, sigma22, mu_t, mu_Nt):
    """
    uses the (split up) covariance matrix to calculate sigmaBar, used for the resulting beliefs over the remaining items after item consumption
    """
    # TODO delete all params that are not used?
    inverse = np.linalg.inv(sigma11)
    innerMulti = sigma21 * inverse
    sigmaBar = sigma22 - (innerMulti + sigma12)     # = sigma(N-t,N-t) - sigma(N-t,t)*sigma^-1(t,t) * sigma(t,N-t)

    return sigmaBar







