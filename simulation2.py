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
    for i in range(N):  #TODO if problems arise, other sim counts 1->N instead of 0->N-1
        for j in range(N):
            dist = distance(i, j, N)
            cov_matrix[i][j] = sigma * (rho ** dist)    # = sigma * rho^dist

    return cov_matrix

def iota(n, N):
    return [np.cos(n/N)*np.pi,np.sin(n/N)*np.pi]

def certainty_equivalent(alpha, mu, sigma):
    new_mu = mu - (0.5*alpha*numpy.diag(sigma))
    return new_mu

def w_fun(CiT, Ui, T):
    w_score = 0.0
    for i in range(len(CiT)):
        w_score = w_score + Ui[CiT[i]]
    return w_score*(T**(-1))

def init_sigma(Cit, Nit,Sigma_Ui):
    sigma_11 = np.ones((len(Cit),len(Cit)), dtype=float)
    sigma_12 = np.ones((len(Cit),len(Nit)), dtype=float)
    sigma_21 = np.ones((len(Nit),len(Cit)), dtype=float)
    sigma_22 = np.ones((len(Nit),len(Nit)), dtype=float)

    for i in range(len(Cit)):
        for j in range(len(Cit)):
            sigma_11[i,j] = Sigma_Ui[Cit[i],Cit[j]]

        for j in range(len(Nit)):
            sigma_21[j,i] = Sigma_Ui[Cit[i], Nit[j]]

    for i in range(len(Nit)):
        for j in range(len(Cit)):
            sigma_12[j,i] = Sigma_Ui[Nit[i],Cit[j]]

        for j in range(len(Nit)):
            sigma_22[i,j] = Sigma_Ui[Nit[i], Nit[j]]

    return sigma_11,sigma_12,sigma_21,sigma_22



def get_mubar_sigmamu(Sigma_Ui,Ui,x1,Sigma11,Sigma12,Sigma21,Sigma22,mu1,mu2):


    a = [Ui[n] for n in x1]
    a = np.array(a)

    inv_mat = np.linalg.inv(Sigma11)

    #inner = Sigma21 * inv_mat
    inner= np.dot(Sigma21,inv_mat)

    #mubar = mu2 + inner * (a - mu1)
    mubar = mu2 + np.dot(inner,a - mu1)

    #sigmabar = Sigma22 - (inner * Sigma12)
    sigmabar = Sigma22 - (np.dot(inner, Sigma12))
    return mubar, sigmabar


def update_Ui(Cit,Ui,mu_Ui,Sigma_Ui,Nset):

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # μ_bar = μ_1 + Ε12 Ε22^-1 ( a - μ_2 )

    x1 = Cit
    x2 = [n for n in Nset if n not in Cit]

    mu1 = [mu_Ui[n] for n in x1]
    mu2 = [mu_Ui[n] for n in x2]

    Sigma11, Sigma12, Sigma21, Sigma22 = init_sigma(x1, x2, Sigma_Ui)

    mubar, sigmabar = get_mubar_sigmamu(Sigma_Ui, Ui, x1, Sigma11, Sigma12, Sigma21, Sigma22, mu1, mu2)
    return mubar, sigmabar

def choice_helper(ce, choice_set):
    cit = choice_set[np.argmax(ce)]
    return cit


def thompson_sampling(mu_V,Sigma_V):

    draws = [np.random.normal(mu_V[ii],Sigma_V[ii, ii]) for ii in range(len(mu_V))]
    c_it = np.argmax(draws)

    return c_it

def choice_part(
    V_i,
    mu_V_i,
    Sigma_V_i,
    V,
    T,
    N,
    Nset,
    alpha,
    beta
):
    C_iT = np.empty(dtype=int,shape=0)
    R_iT = np.empty(dtype=int,shape=0)

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
        ce_Uit = certainty_equivalent(alpha, mu_Uit, Sigma_Vit) # γ: uncertainty aversion

        c_it = choice_helper(ce_Uit, choice_set)
        r_it = choice_set[np.argmax([V[i] for i in choice_set])]
        R_iT = np.append(R_iT,r_it)
        C_iT = np.append(C_iT,c_it)


    return C_iT, R_iT

def choice_omni(U_i,T,N, Nset):
    C_iT = np.empty(dtype=int,shape=0)
    for t in range(T):
        choice_set = [n for n in Nset if n not in C_iT]
        sub_U_i = [U_i[n] for n in choice_set]
        c_it = choice_helper(sub_U_i, choice_set)
        C_iT = np.append(C_iT, c_it)

    return C_iT


def choice_ind(U_i,
mu_U_i,
Sigma_U_i,
T,
N,
Nset,
alpha
):


    C_iT = np.empty(dtype=int,shape=0)
    for t in range(T):
        if len(C_iT) > 0:
            mu_Uit, Sigma_Uit = update_Ui(C_iT, np.copy(U_i), np.copy(mu_U_i), np.copy(Sigma_U_i), Nset)
        else:
            mu_Uit = np.copy(mu_U_i)
            Sigma_Uit = np.copy(Sigma_U_i)


        # make choice
        ce_Uit = certainty_equivalent(alpha, mu_Uit, Sigma_Uit)
        choice_set = [n for n in Nset if n not in C_iT]

        c_it = choice_helper(ce_Uit, choice_set)
        C_iT = np.append(C_iT, c_it)

    return C_iT


def simulate(N,
T,
sigma,
sigma_i,
sigma_ibar,
beta,
nr_ind,
Sigma_V_i,
Sigma_V,
Sigma_V_ibar,
alpha,
seed
):


    print("iteration: $seed ")
    np.random.seed(seed)

    Nset = [n for n in range(N)]  # set of N items I = {1, ..., N}



    C_pop = {"no_rec" : np.zeros((nr_ind, T),dtype=int), "omni" : np.zeros((nr_ind, T),dtype=int), "partial" : np.zeros((nr_ind, T),dtype=int)}
    W_pop = {"no_rec" : np.zeros((nr_ind, T),dtype=int), "omni" : np.zeros((nr_ind, T),dtype=int), "partial" : np.zeros((nr_ind, T),dtype=int)}
    R_pop = {"no_rec" : np.zeros((nr_ind, T),dtype=int), "omni" : np.zeros((nr_ind, T),dtype=int), "partial" : np.zeros((nr_ind, T),dtype=int)}

    # V = (v_n) n in I aka: common value component v_n in vector form

    # MvNormal(mu, sig)
    # Construct a multivariate normal distribution with mean mu and covariance represented by sig.
    # https://juliastats.github.io/Distributions.jl/stable/multivariate/#Distributions.MvNormal
    mu_V = np.zeros(N,dtype=float)
    V = np.random.multivariate_normal(mu_V, Sigma_V)

    for it_ind in range(nr_ind):
        # V_i = (v_in) n in I aka: consumer i’s idiosyncratic taste for good n in vector form

        mu_V_i = np.random.multivariate_normal(np.zeros(N,dtype=float), Sigma_V_ibar)
        mu_V_ibar = mu_V_i
        V_i = np.random.multivariate_normal(mu_V_i, Sigma_V_i)

        # Utility in vector form
        U_i = V_i + (beta * V)
        mu_U_i = mu_V_i + beta * mu_V
        mu_U_i = mu_U_i.astype(float)
        mu_V_i = mu_V_i.astype(float)


        ## NO RECOMMENDATION CASE
        Sigma_U_i = Sigma_V_i + beta ** 2 * (Sigma_V)
        C_iT_no_rec = choice_ind(np.copy(U_i), np.copy(mu_U_i), np.copy(Sigma_U_i), T, N, Nset, alpha)

        #print(C_pop["no_rec"])
        #print(C_pop["no_rec"][it_ind, :])
        #print(C_iT_no_rec)
        C_pop["no_rec"][it_ind, :] = C_iT_no_rec
        W_pop["no_rec"][it_ind, :] = U_i[C_iT_no_rec]

        ## OMNISCIENT CASE
        C_iT = choice_omni(np.copy(U_i), T, N, Nset)
        C_pop["omni"][it_ind, :] = C_iT
        W_pop["omni"][it_ind, :] = U_i[C_iT]

        ## PARTIAL REC Case
        C_iT_partial, R_iT = choice_part(np.copy(V_i), np.copy(mu_V_i), np.copy(Sigma_V_i), np.copy(V), T, N, Nset, alpha, beta)
        C_pop["partial"][it_ind, :] = C_iT_partial
        W_pop["partial"][it_ind, :] = np.copy(U_i[C_iT_partial])
        R_pop["partial"][it_ind, :] = R_iT


    return {"Consumption" : C_pop, "Welfare" : W_pop, "Rec" : R_pop }

nr_pop = 10
#
nr_ind = 10
#
sigma_ibar = .1
#
rho_ibar = 0.0

N_vals = [20]

T_vals = [20]

# Covariance structure
rho_vals = [0.0]#[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
# utility idiosyncratic degree
beta_vals = [0.0]#, 0.4, 0.8, 1., 2., 5.]
# absolute risk aversion
alpha_vals = [0.0]#, 0.3, 0.6, 1., 5.]

sigma_vals = [0.25]#[0.25, 0.5, 1.0, 2.0, 4.0]

params = list(product(N_vals, T_vals, rho_vals, beta_vals, sigma_vals, alpha_vals))

WORKING_DIR = './'

NUM_SIMS_TO_WRITE = 25
file_idx = 1
sim_results = {}
total_num = 0
print(params)
for (N, T, rho, beta, sigma, alpha) in params:
    print("STARTING")
    print(f"N: {N}, T: {T}, ρ: {rho} β: {beta} σ: {sigma} α: {alpha}")
    print(datetime.now())
    #flush(stdout) # so that nohup shows progress
    sigma_i = sigma

    Sigma_V_i = createCovarianceMatrix(sigma, rho, N)
    Sigma_V = createCovarianceMatrix(sigma,rho,N)

    Sigma_V_ibar = createCovarianceMatrix(sigma_ibar,rho_ibar,N)
    dict_key_str = f"N: {N} T {T} rho {rho} beta {beta} sigma {sigma} alpha {alpha} nr_pp {nr_pop} nr_ind {nr_ind}"
    sim_results[dict_key_str] = []
    for i in range(nr_pop):

        sim_results[dict_key_str].append(simulate(N, T,sigma, sigma_i, sigma_ibar, beta, nr_ind, Sigma_V_i,  Sigma_V,  Sigma_V_ibar,  alpha, i))

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

with open(WORKING_DIR + file_name,"w") as f:
    f.write(sim_results_dumped)

print(total_num)
print(NUM_SIMS_TO_WRITE)
