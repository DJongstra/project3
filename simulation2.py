import numpy
import numpy as np
import json

from datetime import datetime

from itertools import product

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    for i in range(N):
        for j in range(N):
            dist = distance(i, j, N)
            cov_matrix[i][j] = sigma * (rho ** dist)    # = sigma * rho^dist

    return cov_matrix


def certainty_equivalent(gamma, mu, sigma):
    """
    :param gamma: risk aversion parameter,
                higher gamma implies higher risk-aversion,
                0 is neutral case
    :param mu: expected value (mu_n for item n)
    :param sigma: expected variance (sigma_nn for item n)
    :return: certainty equivalents ( = the sure value that a user is indifferent consuming a certain item)
    """
    new_mu = mu - (0.5 * gamma * numpy.diag(sigma))
    return new_mu

def welfareScore(CiT, Ui, T):
    """
    welfare: average of the realized values
    """
    welfare_score = 0.0
    for i in range(len(CiT)):
        welfare_score = welfare_score + Ui[CiT[i]]
    return welfare_score*(T**(-1))

def init_sigma(Cit, Nit,Sigma_Ui):
    """"
    partition covariance matrix
    """
    sigma_11 = np.ones((len(Cit), len(Cit)), dtype=float)
    sigma_12 = np.ones((len(Cit), len(Nit)), dtype=float)
    sigma_21 = np.ones((len(Nit), len(Cit)), dtype=float)
    sigma_22 = np.ones((len(Nit), len(Nit)), dtype=float)

    for i in range(len(Cit)):
        for j in range(len(Cit)):
            sigma_11[i, j] = Sigma_Ui[Cit[i], Cit[j]]

        for j in range(len(Nit)):
            sigma_21[j, i] = Sigma_Ui[Cit[i], Nit[j]]

    for i in range(len(Nit)):
        for j in range(len(Cit)):
            sigma_12[j, i] = Sigma_Ui[Nit[i], Cit[j]]

        for j in range(len(Nit)):
            sigma_22[i, j] = Sigma_Ui[Nit[i], Nit[j]]       # known part

    return sigma_11, sigma_12, sigma_21, sigma_22


def get_mubar_sigmamu(Ui, x1, Sigma11, Sigma12, Sigma21, Sigma22, mu_t, mu_Nt):
    """
    :param Ui: utility
    :param x1: consumed items
    :param Sigma11: split up covariance matrix
    :param Sigma12: split up covariance matrix
    :param Sigma21: split up covariance matrix
    :param Sigma22: split up covariance matrix
    :param mu_t: initial mean beliefs user has over items in Ct
    :param mu_Nt: initial mean beliefs user has over items not in Ct
    :return: parameters used for the resulting beliefs over the remaining items after item consumption
    """

    a = np.array([Ui[n] for n in x1])   # get consumed items

    inverse_matrix = np.linalg.inv(Sigma11)

    inner= np.dot(Sigma21,inverse_matrix)

    mubar = mu_Nt + np.dot(inner, a - mu_t)     # mu_N-t + sigma(N-t,t)*sigma^-1(t,t) * (c_t - mu_t)

    sigmabar = Sigma22 - (np.dot(inner, Sigma12))   # = sigma(N-t,N-t) - sigma(N-t,t)*sigma^-1(t,t) * sigma(t,N-t)
    return mubar, sigmabar


def update_Ui(Cit, Ui, mu_Ui, Sigma_Ui, Nset):
    """
    :param Cit: consumed items up to time t for user i
    :param Ui: expected utility for user i
    :param mu_Ui: all initial mean beliefs
    :param Sigma_Ui: covariance matrix
    :param Nset: all items
    """

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # ??_bar = ??_1 + ??12 ??22^-1 ( a - ??_2 )

    x1 = Cit    # consumed items
    x2 = [n for n in Nset if n not in Cit]  # items not yet consumed

    mu1 = [mu_Ui[n] for n in x1]    # initial mean beliefs user has over items already consumed
    mu2 = [mu_Ui[n] for n in x2]    # initial mean beliefs user has over items not yet consumed
    Sigma11, Sigma12, Sigma21, Sigma22 = init_sigma(x1, x2, Sigma_Ui)

    mubar, sigmabar = get_mubar_sigmamu(Ui, x1, Sigma11, Sigma12, Sigma21, Sigma22, mu1, mu2)
    return mubar, sigmabar

def choice_helper(ce, choice_set):
    cit = choice_set[np.argmax(ce)]     # choose item with highest expected utility
    return cit


def choice_part(V_i, mu_V_i, Sigma_V_i, V, T, Nset, gamma, beta):
    """
    :param V_i: idiosyncratic component = personal preferences user
    :param mu_V_i: initial mean beliefs
    :param Sigma_V_i:   covariance matrix
    :param V: common value
    :param T: amount of items to consume, small fraction of total items
    :param Nset: total itemset
    :param gamma: risk-aversion parameter
    :param beta: degree to which valuations are individualized or generalized across users
    """
    C_iT = np.empty(dtype=int, shape=0)     # consumption history
    R_iT = np.empty(dtype=int, shape=0)

    cur_V = np.copy(V)

    for t in range(T):
        choice_set = [n for n in Nset if n not in C_iT]
        mu_Vit = np.copy(mu_V_i)
        Sigma_Vit = np.copy(Sigma_V_i)
        if len(C_iT) > 0:
            # update beliefs
            mu_Vit, Sigma_Vit = update_Ui(C_iT, np.copy(V_i), np.copy(mu_Vit), np.copy(Sigma_Vit), Nset)
            cur_V = [V[i] for i in Nset if i not in C_iT]

        #mu_Uit = mu_Vit + beta * cur_V
        mu_Uit = mu_Vit + np.dot(beta, cur_V)

        # make choice
        ce_Uit = certainty_equivalent(gamma, mu_Uit, Sigma_Vit) # ??: uncertainty aversion

        c_it = choice_helper(ce_Uit, choice_set)
        r_it = choice_set[np.argmax([V[i] for i in choice_set])]
        R_iT = np.append(R_iT, r_it)
        C_iT = np.append(C_iT, c_it)        # update consumption history

    return C_iT, R_iT

def choice_omni(U_i, T, Nset):
    """
    :param U_i: utility
    :param T: amount of items to consume, small fraction of total items
    :param Nset: total itemset
    """
    C_iT = np.empty(dtype=int, shape=0)
    for t in range(T):
        choice_set = [n for n in Nset if n not in C_iT]
        sub_U_i = [U_i[n] for n in choice_set]
        c_it = choice_helper(sub_U_i, choice_set)
        C_iT = np.append(C_iT, c_it)

    return C_iT


def choice_ind(U_i, mu_U_i, Sigma_U_i, T, Nset, gamma):
    """
    :param U_i: utility
    :param mu_U_i: initial mean beliefs
    :param Sigma_U_i:   covariance matrix
    :param T: amount of items to consume, small fraction of total items
    :param Nset: total itemset
    :param gamma: risk-aversion parameter
    """
    C_iT = np.empty(dtype=int, shape=0)
    for t in range(T):
        if len(C_iT) > 0:
            mu_Uit, Sigma_Uit = update_Ui(C_iT, np.copy(U_i), np.copy(mu_U_i), np.copy(Sigma_U_i), Nset)
        else:
            mu_Uit = np.copy(mu_U_i)
            Sigma_Uit = np.copy(Sigma_U_i)

        # make choice
        ce_Uit = certainty_equivalent(gamma, mu_Uit, Sigma_Uit)
        choice_set = [n for n in Nset if n not in C_iT]

        c_it = choice_helper(ce_Uit, choice_set)
        C_iT = np.append(C_iT, c_it)

    return C_iT


def simulate(N, T, beta, nr_ind, Sigma_V_i, Sigma_V, Sigma_V_ibar, gamma, seed):

    print("iteration: ", seed+1)
    np.random.seed(seed)

    Nset = [n for n in range(N)]  # set of N items I = {1, ..., N}



    #C_pop = {"no_rec" : np.zeros((nr_ind, T),dtype=int), "omni" : np.zeros((nr_ind, T),dtype=int), "partial" : np.zeros((nr_ind, T),dtype=int)}
    #W_pop = {"no_rec" : np.zeros((nr_ind, T)), "omni" : np.zeros((nr_ind, T)), "partial" : np.zeros((nr_ind, T))}
    #R_pop = {"no_rec" : np.zeros((nr_ind, T)), "omni" : np.zeros((nr_ind, T)), "partial" : np.zeros((nr_ind, T))}

    C_pop = {"partial" : np.zeros((nr_ind, T),dtype=int)}
    W_pop = { "partial" : np.zeros((nr_ind, T))}
    R_pop = { "partial" : np.zeros((nr_ind, T))}

    # V = (v_n) n in I aka: common value component v_n in vector form

    # MvNormal(mu, sig)
    # Construct a multivariate normal distribution with mean mu and covariance represented by sig.
    # https://juliastats.github.io/Distributions.jl/stable/multivariate/#Distributions.MvNormal
    mu_V = np.zeros(N,dtype=float)
    V = np.random.multivariate_normal(mu_V, Sigma_V)

    for it_ind in range(nr_ind):
        # V_i = (v_in) n in I aka: consumer i???s idiosyncratic taste for good n in vector form

        mu_V_i = np.random.multivariate_normal(np.zeros(N,dtype=float), Sigma_V_ibar)
        mu_V_ibar = mu_V_i
        V_i = np.random.multivariate_normal(mu_V_i, Sigma_V_i)

        # Utility in vector form
        U_i = V_i + (beta * V)
        mu_U_i = mu_V_i + beta * mu_V
        mu_U_i = mu_U_i.astype(float)
        mu_V_i = mu_V_i.astype(float)


        ## NO RECOMMENDATION CASE
        #Sigma_U_i = Sigma_V_i + beta ** 2 * (Sigma_V)
        #C_iT_no_rec = choice_ind(np.copy(U_i), np.copy(mu_U_i), np.copy(Sigma_U_i), T, Nset, gamma)

        #C_pop["no_rec"][it_ind, :] = C_iT_no_rec
        #W_pop["no_rec"][it_ind, :] = U_i[C_iT_no_rec]

        ## OMNISCIENT CASE
        #C_iT = choice_omni(np.copy(U_i), T, Nset)
        #C_pop["omni"][it_ind, :] = C_iT
        #W_pop["omni"][it_ind, :] = U_i[C_iT]

        ## PARTIAL REC Case
        C_iT_partial, R_iT = choice_part(np.copy(V_i), np.copy(mu_V_i), np.copy(Sigma_V_i), np.copy(V), T, Nset, gamma, beta)
        C_pop["partial"][it_ind, :] = C_iT_partial
        W_pop["partial"][it_ind, :] = np.copy(U_i[C_iT_partial])
        R_pop["partial"][it_ind, :] = R_iT

    return {"Consumption" : C_pop, "Welfare" : W_pop, "Rec" : R_pop }


if __name__ == '__main__':

    nr_pop = 100
    #
    nr_ind = 100
    #
    sigma_ibar = .1
    #
    rho_ibar = 0.0

    N_vals = [200]

    T_vals = [20]

    # Covariance structure
    rho_vals = [0.0]#, 0.1 , 0.3, 0.5, 0.7, 0.9]
    # utility idiosyncratic degree
    beta_vals = [0.0]#, 0.4   , 0.8, 1., 2., 5.]
    # absolute risk aversion
    alpha_vals = [0]#, 0.3   , 0.6, 1., 5.]

    sigma_vals = [0.25]#, 0.5   , 1.0, 2.0, 4.0]

    params = list(product(N_vals, T_vals, rho_vals, beta_vals, sigma_vals, alpha_vals))

    WORKING_DIR = './'

    NUM_SIMS_TO_WRITE = 25
    file_idx = 1
    sim_results = {}
    total_num = 0
    print(params)
    for (N, T, rho, beta, sigma, alpha) in params:
        print("STARTING")
        print(f"N: {N}, T: {T}, ??: {rho} ??: {beta} ??: {sigma} ??: {alpha}")
        print(datetime.now())
        #flush(stdout) # so that nohup shows progress
        sigma_i = sigma

        Sigma_V_i = createCovarianceMatrix(sigma, rho, N)
        Sigma_V = createCovarianceMatrix(sigma, rho, N)

        Sigma_V_ibar = createCovarianceMatrix(sigma_ibar,rho_ibar,N)
        dict_key_str = f"({N},{T},{rho},{beta},{sigma},{alpha},{nr_pop},{nr_ind})"
        sim_results[dict_key_str] = []
        for i in range(nr_pop):

            sim_results[dict_key_str].append(simulate(N, T, beta, nr_ind, Sigma_V_i,  Sigma_V,  Sigma_V_ibar,  alpha, i))

        total_num = total_num
        if total_num > NUM_SIMS_TO_WRITE:
            file_name = "new_sim_"+str(file_idx)+".json"
            with open(WORKING_DIR + file_name,"w") as f:
                json.dump(sim_results,f)

            file_idx = file_idx + 1
            total_num = 0
            sim_results = {}
        else:
            #print(NUM_SIMS_TO_WRITE)
            #print(total_num)
            total_num  = total_num + 1

    file_name = "new_sim_" + str(file_idx) + ".json"

    sim_results_dumped = json.dumps(sim_results, cls=NumpyEncoder)

    with open(WORKING_DIR + file_name, "w") as f:
        f.write(sim_results_dumped)

    print(total_num)
    print(NUM_SIMS_TO_WRITE)
