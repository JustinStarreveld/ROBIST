# Import packages
import numpy as np
import cvxpy as cp
import mosek
import time
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
        constraints = [-(1-cp.exp(-(S @ x))) <= 0, x <= 1]
    else:
        constraints = []
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def uncertain_constraint(S, x):
    return -(1-np.exp(-(np.dot(S,x))))

def check_robust(bound, numeric_precision, beta=0):
    return (bound <= beta + numeric_precision)

def get_true_exp(x, k):
    true_exp = 1 / (2**k)
    for i in range(1,k+1):
        true_exp = true_exp * (1/x[i]) * (np.exp(x[i]) - np.exp(-x[i]))
    return true_exp - 1
    
def solve_toyproblem_true_exp(beta, k):
    x = cp.Variable(k, nonneg = True)
    s = cp.Variable(k)
    constraints = [cp.sum(-cp.log(x)) + x + s <= k*np.log(2), cp.exp(-s) - cp.exp(-2*x - s) <= 1]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)

# Set parameter values
random_seed = 1
risk_measure = 'exp_constraint_leq' # options: 'chance_constraint', 'exp_constraint_leq', 'exp_constraint_geq'
num_obs_per_bin = 5
alpha = 0.10
beta = 0
k = 10
N_train = 2500
N_test = 2500
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

# Get generated data
N_total = N_train + N_test

data = generate_data(random_seed, k, N_total)
data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)

N_eval = 1000000
data_eval = generate_data(random_seed + 99, k, N_eval)

time_limit_search = 10*60 # in seconds (time provided to search algorithm)
time_limit_mosek = 10*60 # in seconds (for larger MIP / LP solves)
time_limit_solve = 5*60 # in seconds (for individuals solves of SCP)
max_nr_solutions = 1 # for easy problems with long time limits, we may want extra restriction
add_remove_threshold = 0.00 # This determines when randomness is introduced in add/removal decision
use_tabu = False # Determines whether the tabu list are used in the search

add_strategy = 'random_vio'
remove_strategy = 'random_active'
clean_strategy = (1000, 'all_inactive')


(runtime, num_iter, solutions, 
 best_sol, pareto_solutions) = rs.gen_and_eval_alg(data_train, data_test, beta, alpha, time_limit_search, time_limit_solve, 
                                                    max_nr_solutions, add_strategy, remove_strategy, clean_strategy, 
                                                    add_remove_threshold, use_tabu,
                                                    phi_div, phi_dot, numeric_precision,
                                                    solve_SCP, uncertain_constraint, check_robust,
                                                    risk_measure, random_seed, num_obs_per_bin, data_eval)



