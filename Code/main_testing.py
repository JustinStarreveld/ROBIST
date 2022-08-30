# Import packages
import numpy as np
import pandas as pd
import cvxpy as cp
import mosek
import time
import scipy
from sklearn.model_selection import train_test_split

import phi_divergence as phi
import robust_sampling as rs
import dataio
import util

# Problem specific functions:
def generate_data(random_seed, k, N):
    np.random.seed(random_seed)
    data = np.random.uniform(-1,1,size = (N,k)) # generates N random scenarios    
    return data 

def generate_data_with_nominal(random_seed, k, N):
    data_nominal = np.array([[0] * k])
    np.random.seed(random_seed)
    data = np.random.uniform(-1,1,size = (N-1,k)) # generate N-1 scenarios
    data = np.concatenate((data_nominal,data)) # add nominal case to training data
    return data

def solve_SCP(k, S, time_limit):
    #k = S.shape[1]
    x = cp.Variable(k, nonneg = True)
    if len(S) > 0:
        constraints = [(S @ x) - 1 <= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    else:
        constraints = [cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def uncertain_constraint(S, x):
    return (np.dot(S, x)) - 1

def get_true_prob(x, k):
    return(1/2+1/(2*x[k-1]))
    
def solve_toyproblem_true_prob(beta, k):
    x = cp.Variable(k, nonneg = True)
    constraints = [(1-2*beta)*x[k-1] + 1 >= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)

# Set parameter values
random_seed = 1
alpha = 0.01
beta = 0.90
k = 2
N_train = 250
N_test = 250
N_total = 500
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

data = generate_data(random_seed, k, N_total)
data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)

# RS-related parameters
time_limit_search = 10*60
time_limit_solve = 5*60 # in seconds
max_nr_solutions = 10000 # for easy problems with long time limits, we may want extra restriction
add_strategy = 'random_vio'
remove_strategy = 'random_any'
clean_strategy = (30000, 'random_inactive')
add_remove_threshold = 0.00 #0.10 controls the ambiguity around adding/removing
use_tabu = False

# (runtime, num_iter, solutions, 
#  best_sol, pareto_solutions) = rs.gen_and_eval_alg(data_train, data_test, beta, alpha, time_limit_search, time_limit_solve, 
#                                                            max_nr_solutions, add_strategy, remove_strategy, clean_strategy, 
#                                                            add_remove_threshold, use_tabu,  
#                                                            par, phi_div, phi_dot, numeric_precision,
#                                                            solve_SCP, uncertain_constraint, random_seed)

    
(x, obj, j, s_j, size_S, time_determine_set_sizes,
 time_main_solves, time_determine_supp) = util.solve_with_Garatti2022(10, 0.90, 10e-6, solve_SCP, uncertain_constraint, 
                                                                       generate_data, 1, time_limit_solve,
                                                                       numeric_precision)









