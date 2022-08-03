# Import packages
import numpy as np
import cvxpy as cp
import mosek
import time

import phi_divergence as phi
import robust_sampling as rs
import dataio
import util

# Problem specific functions:
def generate_data(k, N):
    np.random.seed(1)
    data = np.random.uniform(-1,1,size = (N,k)) # generates N random scenarios    
    return data 

def generate_data_with_nominal(k, N):
    data_nominal = np.array([[0] * k])
    np.random.seed(1)
    data = np.random.uniform(-1,1,size = (N-1,k)) # generate N-1 scenarios
    data = np.concatenate((data_nominal,data)) # add nominal case to training data
    return data

def solve_SCP(S, time_limit):
    k = S.shape[1]
    x = cp.Variable(k, nonneg = True)
    constraints = [(S @ x) - 1 <= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def uncertain_constraint(S, x):
    return (S @ x) - 1

def get_true_prob(x, k):
    return(1/2+1/(2*x[k-1]))
    
def solve_toymodel_true_prob(beta, k):
    x = cp.Variable(k, nonneg = True)
    constraints = [(1-2*beta)*x[k-1] + 1 >= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)


headers = ['$k$', 'strategy', 
           'Obj.~Alg.', 'Obj.~(true prob.)', 'Gap (\%)', 'LB', 'True Prob.', 
           '\#Iterations (add)', '\#Iterations (remove)', '\#Iterations (improve)', 
           '$|\mathcal{S}|$', 'Time until best found']

output_data = {}

# Variables parameter values
k_settings = [100000, 1000000]
N_train = 1000
N_test = 10000
time_limit_search = 15*60
strategy_settings = ['add', 'add + improve', 'add + improve + remove']

# Fixed parameter values
alpha = 10**-6
beta = 0.95
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 1
time_limit_mosek = 1*60
time_limit_solve = 1*time_limit_search # in seconds 
max_nr_solutions = 1000000 # for easy problems with long time limits, we may want extra restriction
numeric_precision = 1e-6 # To correct for floating-point math operations

# Alg settings
random_seed = 1
add_strategy = 'random_vio'
#['smallest_vio', 'N*(beta-lb)_smallest_vio', 'random_vio', 'random_weighted_vio']
remove_strategy = 'all_inactive'
improve_strategy = 'random_active'

for k in k_settings:
    
    x_true, obj_true = solve_toymodel_true_prob(beta, k)
    
    data_train = generate_data_with_nominal(k, N_train)
    data_test = generate_data(k, N_test)
    
    for strategy in strategy_settings:
        if strategy != 'add':
            improve_strategy = 'random_active'
        else:
            improve_strategy = None
            
        if strategy == 'add + improve + remove':
            threshold_time_solve = 0.001 * time_limit_search
        else:
            threshold_time_solve = time_limit_search
    
        
        runtime_search, num_iter, solutions = rs.search_alg(data_train, beta, alpha, time_limit_search, time_limit_solve, 
                                                   threshold_time_solve, max_nr_solutions, add_strategy, remove_strategy,
                                                   improve_strategy, par, phi_div, phi_dot, numeric_precision,
                                                   solve_SCP, uncertain_constraint, random_seed)
        
        runtime_eval, best_sol, pareto_solutions = rs.evaluate_alg(solutions, data_test, beta, alpha, par, phi_div, phi_dot, 
                                                 uncertain_constraint, numeric_precision)
        obj_alg = best_sol['obj']
        time_best_found = best_sol['time']
        lb_alg = best_sol['lb_test']
        num_scen = len(best_sol['scenario_set'])
        
        
        x_true_prob = get_true_prob(best_sol['sol'], k)
        obj_gap_true =  100*(obj_true - obj_alg)/obj_true

        output_data[(k, strategy)] = [f'{round(obj_alg,3):.3f}',
                                       f'{round(obj_true,3):.3f}',
                                       f'{round(obj_gap_true,1):.1f}',
                                       f'{round(lb_alg,3):.3f}',
                                      f'{round(x_true_prob,3):.3f}',
                                       num_iter['add'], 
                                       num_iter['remove'],
                                       num_iter['improve'],
                                      num_scen,
                                       f'{round(time_best_found, 0):.0f}']

dataio.write_output_to_latex(2, headers, output_data)

output_file_name = 'new_output'
with open(r'output/'+output_file_name+'.txt','w+') as f:
    f.write(str(output_data))








