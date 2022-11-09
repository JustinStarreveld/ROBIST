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
from scipy.special import erfinv

# Problem specific functions:
def generate_data(random_seed, k, N):
    np.random.seed(random_seed)
    # NOTE: not entirely clear in Esfahani & Kuhn paper whether they refer to stdev or var
    sys_risk_mean = 0
    #sys_risk_stdev = math.sqrt(0.02)
    sys_risk_stdev = 0.02
    unsys_risk_mean = np.fromiter(((i * 0.03) for i in range(1,k+1)), float)
    #unsys_risk_stdev = np.fromiter(( math.sqrt(i * 0.025) for i in range(1,k+1)), float)
    unsys_risk_stdev = np.fromiter(( i * 0.025 for i in range(1,k+1)), float)
    
    data = np.empty([N,k])
    for n in range(0, N):
        sys_return = np.random.normal(sys_risk_mean, sys_risk_stdev)
        for i in range(0, k):
            unsys_return = np.random.normal(unsys_risk_mean[i], unsys_risk_stdev[i])
            data[n, i] = sys_return + unsys_return
            
    return data 

def emp_eval(x, data, settings):
    rho = settings.get('rho', 10)
    CVaR_alpha = settings.get('CVaR_alpha', 0.20)
    x_sol = x[2:]
    emp_returns = - (np.dot(data, x_sol))
    exp_loss = np.mean(emp_returns, dtype=np.float64)
    VaR = np.quantile(emp_returns, 1-CVaR_alpha, method='inverted_cdf') # gets threshold for top CVaR_alpha-% highest losses
    above_VaR = (emp_returns > VaR)
    cVaR = np.mean(emp_returns[above_VaR])
    return exp_loss + (rho*cVaR)

def solve_P_SCP(k, S, settings):
    time_limit = settings.get('time_limit')
    rho = settings.get('rho', 10)
    CVaR_alpha = settings.get('CVaR_alpha', 0.20)
    tau = cp.Variable(1)
    a_1 = -1
    b_1 = rho
    a_2 = -1 - (rho/CVaR_alpha)
    b_2 = rho*(1 - (1/CVaR_alpha))
    
    x = cp.Variable(k, nonneg = True)
    theta = cp.Variable(1)
    gamma = cp.Variable(1)
    constraints = [gamma - theta <= 0,
                   a_1*(S @ x) + b_1*tau - gamma <= 0, 
                   a_2*(S @ x) + b_2*tau - gamma <= 0, 
                   cp.sum(x) == 1]
    
    obj = cp.Maximize(-theta) #equivalent to min \theta
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, 
                   # verbose=True,
                   mosek_params = {mosek.dparam.optimizer_max_time: time_limit}
                   )
    except cp.error.SolverError:
        print("Note: error occured in solving SCP problem...")
        return(None, float('-inf'))
    # Combine theta, tau and x into 1 single solution vector
    x_value = np.concatenate((theta.value,tau.value,x.value))
    # Because we are actually minimizing, we add "-" to obj
    return(x_value, -prob.value)

def solve_P_SAA(k, S, settings):
    time_limit = settings.get('time_limit')
    rho = settings.get('rho', 10)
    CVaR_alpha = settings.get('CVaR_alpha', 0.20)
    a_1 = -1
    b_1 = rho
    a_2 = -1 - (rho/CVaR_alpha)
    b_2 = rho*(1 - (1/CVaR_alpha))
    N = S.shape[0]
    
    x = cp.Variable(k, nonneg = True)
    theta = cp.Variable(1)
    gamma = cp.Variable(N)
    tau = cp.Variable(1)
    
    constraints = [(1/N)*cp.sum(gamma) - theta <= 0,
                   a_1*(S @ x) + b_1*tau - gamma <= 0, 
                   a_2*(S @ x) + b_2*tau - gamma <= 0, 
                   cp.sum(x) == 1]
    
    obj = cp.Maximize(-theta) #equivalent to min \theta
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, 
                   # verbose=True,
                   mosek_params = {mosek.dparam.optimizer_max_time: time_limit}
                   )
    except cp.error.SolverError:
        print("Note: error occured in solving SAA problem...")
        return(None, float('-inf'))
    # Combine theta, tau and x into 1 single solution vector
    x_value = np.concatenate((theta.value,tau.value,x.value))
    # Because we are actually minimizing, we add "-" to obj
    return(x_value, -prob.value)

def unc_func(data, x, settings):
    rho = settings.get('rho', 10)
    CVaR_alpha = settings.get('CVaR_alpha', 0.20)
    a_1 = -1
    b_1 = rho
    a_2 = -1 - (rho/CVaR_alpha)
    b_2 = rho*(1 - (1/CVaR_alpha))
    # Assume that x[0] contains theta and x[1] contains tau
    constr1 = a_1*np.dot(data,x[2:]) + b_1*x[1] 
    constr2 = a_2*np.dot(data,x[2:]) + b_2*x[1] 
    return np.maximum(constr1, constr2)

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
    
def stopping_cond(stop_info, elapsed_time, num_solutions):
    if (elapsed_time > stop_info.get('max_elapsed_time', 10e12) 
        or num_solutions > stop_info.get('max_num_solutions', 10e12)):
        return True
    else:
        return False
    
    

# Set parameter values (as in Kuhn paper)
k = 10
rho = 10
CVaR_alpha = 0.20
problem_info = {'rho': rho, 
                    'CVaR_alpha': CVaR_alpha,
                    'risk_measure': 'expectation', # options: 'probability'/'expectation'
                    'time_limit': 10*60}

# Set our own algorithm parameter values
conf_param_alpha = 0.05

N_eval = 1000000
data_eval = generate_data(999, k, N_eval)

# Set other parameter values
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

time_limit_search = 10*60 # in seconds (time provided to search algorithm)
max_nr_solutions = 10000 # for easy problems with long time limits, we may want extra restriction
use_tabu = False # Determines whether the tabu list are used in the search

add_strategy = 'random_vio'
remove_strategy = 'random_any'
clean_strategy = (999999, 'all_inactive') # set arbitrarily high such that it never occurs

# Sets the Problem to be solved at each iteration
# solve_P = solve_P_SCP
solve_P = solve_P_SAA

vec_obj_Cert = []
vec_obj_OoS = []
# vec_obj_rel = []

N_settings = [30, 300, 3000]
random_seed_settings = [i for i in range(3)]

for N_total in N_settings:
    count_runs = 1
    for random_seed in random_seed_settings:
        N_train = int(N_total / 2)
        N_test = N_total - N_train
        bound_settings = {'min_num_obs_per_bin': 5,
                          'num_bins_range': [min(5, math.floor(min(N_train,N_test)/5)-1), 
                                             min(20, math.floor(min(N_train,N_test)/5))]}
        data = generate_data(random_seed, k, N_total)   
        data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)   

        (runtime, 
         num_iter, 
         solutions, 
         best_sol, 
         pareto_solutions) = rs.gen_and_eval_alg_obj(solve_P, unc_func, problem_info,
                                                     data_train, data_test, conf_param_alpha, 
                                                     bound_settings, phi_div, phi_dot,
                                                     time_limit_search, 
                                                     max_nr_solutions, add_strategy, 
                                                     remove_strategy, clean_strategy, 
                                                     util.compute_prob_add, use_tabu,
                                                     numeric_precision, random_seed, 
                                                     data_eval, emp_eval_obj, 
                                                     analytic_out_perf=None,
                                                     verbose=False)

        if best_sol['sol'] is not None:                                                   
            obj = best_sol['bound_test']
            vec_obj_Cert.append(obj)
            eval_true_obj = emp_eval_obj(best_sol['sol'], data_eval, problem_info)
            vec_obj_OoS.append(eval_true_obj)

    #         print("-----------------")
            print(count_runs)
        else:
            print(str(count_runs) + ": No feasible solution found")
        
        count_runs += 1
        
    print("-----------------")
    print(N_total)
    print("-----------------")
    print(f'{round(np.mean(vec_obj_Cert),3):.3f}'+"(" + f'{round(np.std(vec_obj_Cert),3):.3f}' + ")")
    print(f'{round(np.mean(vec_obj_OoS),3):.3f}'+"(" + f'{round(np.std(vec_obj_OoS),3):.3f}' + ")")
    print("-----------------")




















