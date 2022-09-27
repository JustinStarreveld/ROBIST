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
import scipy.stats

# Problem specific functions:
# Problem specific functions:
def generate_data(random_seed, k, N):
    np.random.seed(random_seed)
    
    # NOTE: not entirely clear in Esfahani & Kuhn paper whether they refer to stdev or var
    sys_risk_mean = 0
    sys_risk_stdev = math.sqrt(0.02)
    unsys_risk_mean = np.fromiter(((i * 0.03) for i in range(1,k+1)), float)
    unsys_risk_stdev = np.fromiter((math.sqrt(i * 0.025) for i in range(1,k+1)), float)
    
    data = np.empty([N,k])
    for n in range(0, N):
        sys_return = np.random.normal(sys_risk_mean, sys_risk_stdev)
        for i in range(0, k):
            unsys_return = np.random.normal(unsys_risk_mean[i], unsys_risk_stdev[i])
            data[n, i] = sys_return + unsys_return
            
    return data 

def solve_SCP(k, S, time_limit):
    x = cp.Variable(k, nonneg = True)
    theta = cp.Variable(1)
    rho = 10 #TODO: fix this hardcode
    alpha = 0.2
    tau = cp.Variable(1)
    
    constraints = [-theta - (S @ x) + rho*tau <= 0, 
                   -theta - (1+(rho/alpha))*(S @ x) + (rho*(1-(1/alpha))*tau) <= 0, 
                   cp.sum(x) == 1]#,
                   #-theta <= 10e3] # final contraint added to ensure that problem is bounded
    
    obj = cp.Maximize(-theta)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    x_value = np.concatenate((theta.value,tau.value,x.value)) # Combine x, tau and theta into 1 single solution vector
    return(x_value, prob.value)

def uncertain_constraint(S, x):
    rho = 10 #TODO: fix this hardcode
    alpha = 0.2
    # Assume that x[0] contains theta and x[1] contains tau
    # Contrary to other applications, we have 2 uncertain constraints (because of max), the max{} should be <= 0
    constr1 = -x[0] + np.dot(S,x[2:]) + rho*x[1]
    constr2 = -x[0] - (1+(rho/alpha))*np.dot(S,x[2:]) + (rho*(1-(1/alpha))*x[1])
    
    return np.maximum(constr1, constr2)

def check_robust(bound, numeric_precision, beta=0):
    return (bound <= beta + numeric_precision)

# Set parameter values (as in Kuhn paper)
k = 10
risk_measure = 'exp_constraint_leq' # options: 'chance_constraint', 'exp_constraint'
num_obs_per_bin = 10
alpha = 0.20
beta = 0
N_total = 3000 # 30, 300, 3000
N_train = int(N_total / 2)
N_test = N_total - N_train

# Set other parameter values
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

# Get generated data
random_seed = 1
data = generate_data(random_seed, k, N_total)               
data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)

data = generate_data(random_seed, k, N_total)
data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)

#OPTIONAL:
N_eval = 10000
data_eval = generate_data(random_seed + 99, k, N_eval)

time_limit_search = 1*60 # in seconds (time provided to search algorithm)
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



