# Import packages
import numpy as np
import cvxpy as cp
import mosek
import time
import math
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

def solve_P_SCP(k, S, settings):
    time_limit = settings.get('time_limit')
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

def unc_func(data, x, settings):
    return (np.dot(data,x)) - 1

def compute_prob_add(lhs_constr):
    method = 'deterministic_w_1%'
    if method == 'deterministic':
        if lhs_constr <= 0:
            return 0
        else:
            return 1
    elif method == 'deterministic_w_1%':
        if lhs_constr <= 0:
            return 0.01
        else:
            return 0.99
    elif method == 'sigmoid':
        return util.compute_prob_add_sigmoid(lhs_constr)
    else:
        print('Error: do not recognize method in "compute_prob_add" function')
        return 1
    
def stopping_cond(stop_info, **kwargs):
    if (kwargs.get('elapsed_time',0) >= stop_info.get('max_elapsed_time', 10e12) 
        or kwargs.get('num_solutions',0) >= stop_info.get('max_num_solutions', 10e12)
        or kwargs.get('num_iterations',0) >= stop_info.get('max_num_iterations', 10e12)):
        return True
    else:
        return False

def analytic_eval(x, problem_info):
    k = problem_info['k']
    return get_true_prob(x, k)
    
def get_true_prob(x, k):
    return(1/2+1/(2*x[k-1]))
    
def solve_toyproblem_true_prob(beta, k):
    x = cp.Variable(k, nonneg = True)
    constraints = [(1-2*beta)*x[k-1] + 1 >= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)

alpha=0.05
beta=0.75
max_num_iterations = 1000

# Set parameter values
k = 2
beta = beta
problem_info = {'k': k,
                'risk_measure': 'probability', # options: 'probability'/'expectation'
                'time_limit': 5*60,
                'desired_prob_guarantee_beta': beta}


# Set our own algorithm parameter values
conf_param_alpha = alpha
N_total = 200
N_train = int(N_total / 2)
N_test = N_total - N_train
bound_settings = {'min_num_obs_per_bin': 5,
                  'num_bins_range': [10,min(20, math.floor(N_test/5))]}


# Set other parameter values
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

# Get generated data
random_seed = 0
data = generate_data_with_nominal(random_seed, k, N_total)
data_train = data[:N_train]
data_test = data[N_train:]

stop_info = {'max_elapsed_time': 10*60, # in seconds (time provided to search algorithm)
             'max_num_solutions': 10000,
             'max_num_iterations': max_num_iterations}

use_tabu = False # Determines whether the tabu list are used in the search

add_strategy = 'random_vio'
remove_strategy = 'random_any'
clean_strategy = None #(100, 'all_inactive') 

# Sets the Problem to be solved at each iteration
solve_P = solve_P_SCP
# solve_P = solve_P_SAA


(runtime, 
 num_iter, 
 solutions, 
 best_sol, 
 pareto_solutions) = rs.gen_and_eval_alg_con(solve_P, unc_func, problem_info,
                                             data_train, data_test, conf_param_alpha, 
                                             bound_settings, phi_div, phi_dot,
                                             stopping_cond, stop_info, compute_prob_add,
                                             add_strategy, remove_strategy, clean_strategy, 
                                             use_tabu, numeric_precision, random_seed, 
                                             None, None, 
                                             analytic_eval,
                                             True, True)