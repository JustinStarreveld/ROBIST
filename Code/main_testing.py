# Import packages
import numpy as np
import pandas as pd
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


headers = ['add_strategy', 'improve_strategy', 'seed', 
           'Obj.~Alg.', 'Obj.~(true prob.)', 'Gap (\%)', 'LB', 'True Prob.', 
           '\#Iter.~(add)', '\#Iter.~(remove)', '\#Iter.~(improve)', '$|\mathcal{X}|$',
           '$|\mathcal{S}_{best}|$', '$|\mathcal{S}_{max}|$', '$|\mathcal{S}_{avg}|$', 'Time until best found']

output_data = {}

# Variables parameter values
k = 10000 #[10, 100, 1000, 10000] #
N_train = 10000
N_test = 10000
time_limit_search = 1*60

random_seed_settings = [i for i in range(1, 3)]

# Fixed parameter values
alpha = 10**-6
beta = 0.95
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 1
time_limit_mosek = 1*60
time_limit_solve = 1*time_limit_search # in seconds 
solve_time_threshold = 0.05*time_limit_search
max_nr_solutions = 1000 # for easy problems with long time limits, we may want extra restriction
numeric_precision = 1e-6 # To correct for floating-point math operations

# Alg settings
add_strategy_settings = ['random_vio', 'random_weighted_vio'] 
improve_strategy_settings = ['random_any', 'random_active']
#add_strategy = 'random_vio' #['smallest_vio', 'N*(beta-lb)_smallest_vio', 'random_vio', 'random_weighted_vio']
remove_strategy = 'all_inactive'
#improve_strategy = 'random_active' #'random_any'


x_true, obj_true = solve_toyproblem_true_prob(beta, k)
data_train = generate_data_with_nominal(k, N_train)
data_test = generate_data(k, N_test)

count_runs = 0
for add_strategy in add_strategy_settings:
    
    for improve_strategy in improve_strategy_settings:
    
        for random_seed in random_seed_settings:

            runtime_search, num_iter, solutions = rs.search_alg(data_train, beta, alpha, time_limit_search, time_limit_solve, 
                                                       solve_time_threshold, max_nr_solutions, add_strategy, remove_strategy,
                                                       improve_strategy, par, phi_div, phi_dot, numeric_precision,
                                                       solve_SCP, uncertain_constraint, random_seed)

            runtime_eval, best_sol, pareto_solutions = rs.evaluate_alg(solutions, data_test, beta, alpha, par, phi_div, phi_dot, 
                                                     uncertain_constraint, numeric_precision)
            obj_alg = best_sol['obj']
            time_best_found = best_sol['time']
            lb_alg = best_sol['lb_test']
            num_scen_best = len(best_sol['scenario_set'])
            
            # get num scen avg and max
            num_scen_avg = 0
            num_scen_max = 0
            for sol in solutions:
                num_scen = len(sol['scenario_set'])
                num_scen_avg += num_scen
                if num_scen > num_scen_max:
                    num_scen_max = num_scen
            num_sol = len(solutions)
            num_scen_avg = num_scen_avg / num_sol

            
            x_true_prob = get_true_prob(best_sol['sol'], k)
            obj_gap_true =  100*(obj_true - obj_alg)/obj_true

            output_data[(add_strategy, improve_strategy, random_seed)] = [f'{round(obj_alg,3):.3f}',
                                                                          f'{round(obj_true,3):.3f}',
                                                                          f'{round(obj_gap_true,1):.1f}',
                                                                          f'{round(lb_alg,3):.3f}',
                                                                          f'{round(x_true_prob,3):.3f}',
                                                                          num_iter['add'], 
                                                                          num_iter['remove'],
                                                                          num_iter['improve'],
                                                                          num_sol,
                                                                          num_scen_best,
                                                                          num_scen_max,
                                                                          f'{round(num_scen_avg,1):.1f}',
                                                                          f'{round(time_best_found, 0):.0f}']
                                           
            count_runs += 1
            print("Completed run: " + str(count_runs))




# def plot_pareto_curve(pareto_solutions, beta, best_obj, save_plot, plot_type, show_legend):
#     # first we convert the list of tuples to a numpy array to get data in proper format
#     array = np.array([*pareto_solutions])
#     sorted_array = array[np.argsort(array[:, 0])]
#     x = sorted_array[:,0] # contains lb
#     y = sorted_array[:,1] # contains obj
#     x = 1 - x
    
#     plt.plot(x, y, "-o")
#     plt.vlines(1-beta, 0, np.max(y), linestyles ="dotted")
    
#     plt.xlabel("violation probability")
#     plt.ylabel("objective value");
    
#     plt.show()






