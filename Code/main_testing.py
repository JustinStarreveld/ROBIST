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
def generate_data(random_seed, m, N):
    '''
    Taken from:
    https://nbviewer.org/github/MOSEK/Tutorials/blob/master/dist-robust-portfolio/Data-driven_distributionally_robust_portfolio.ipynb
    '''
    np.random.seed(random_seed)
    R = np.vstack([np.random.normal(
        i*0.03, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
    return (R.transpose())

def generate_data_2(random_seed, m, N):
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

def solve_SCP(k, S, time_limit):
    x = cp.Variable(k, nonneg = True)
    theta = cp.Variable(1)
    gamma = cp.Variable(1)
    rho = 10 #TODO: fix this hardcode
    CVaR_alpha = 0.20
    tau = cp.Variable(1)
    a_1 = -1
    b_1 = rho
    a_2 = -1 - (rho/CVaR_alpha)
    b_2 = rho*(1 - (1/CVaR_alpha))
    
    constraints = [gamma - theta <= 0,
                   a_1*(S @ x) + b_1*tau - gamma <= 0, 
                   a_2*(S @ x) + b_2*tau - gamma <= 0, 
                   cp.sum(x) == 1]
    
    obj = cp.Maximize(-theta) #equivalent to min \theta
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    x_value = np.concatenate((theta.value,gamma.value,tau.value,x.value)) # Combine x, tau and theta into 1 single solution vector
    return(x_value, prob.value)

def uncertain_constraint(S, x):
    rho = 10 #TODO: fix this hardcode
    CVaR_alpha = 0.20
    # Assume that x[1] contains gamma and x[2] contains tau
    # Contrary to other applications, we have 2 uncertain constraints (because of max), the max{} should be <= 0
    constr1 = -np.dot(S,x[3:]) + rho*x[2] - x[1]
    constr2 = -(1+(rho/CVaR_alpha))*np.dot(S,x[3:]) + (rho*(1-(1/CVaR_alpha))*x[2]) - x[1] 
    return np.maximum(constr1, constr2)

def check_robust(bound, numeric_precision, beta=0):
    return (bound <= beta + numeric_precision)

def emp_eval_obj(x, data, rho, CVaR_alpha):
    x_sol = x[3:]
    emp_returns = - (np.dot(data, x_sol))

    exp_loss = np.mean(emp_returns, dtype=np.float64)
    # print(exp_loss)
    
    VaR = np.quantile(emp_returns, 1-CVaR_alpha, method='inverted_cdf') # gets threshold for top CVaR_alpha-% highest losses
    above_VaR = (emp_returns > VaR)
    cVaR = np.mean(emp_returns[above_VaR])
    # print(cVaR)
    
    return exp_loss + (rho*cVaR)

def analytic_out_perf(x, rho, CVaR_alpha):
    '''
    https://nbviewer.org/github/MOSEK/Tutorials/blob/master/dist-robust-portfolio/Data-driven_distributionally_robust_portfolio.ipynb
    Method to calculate the analytical value for the out-of-sample performance.
    [see Rockafellar and Uryasev]
    '''
    x_sol = x[3:]
    m = len(x_sol)
    mu = np.arange(1, m+1)*0.03
    var = 0.02 + (np.arange(1, m+1)*0.025)

    # Constants for CVaR calculation.
    rho = 10
    beta = 1-CVaR_alpha
    c2_beta = 1/(np.sqrt(2*np.pi)*(np.exp(erfinv(2*beta - 1))**2)*(1-beta))
    
    mean_loss = -np.dot(x_sol, mu)
    # print(mean_loss)
    sd_loss = np.sqrt(np.dot(x_sol**2, var))
    cVaR = mean_loss + (sd_loss*c2_beta)
    # print(cVaR)
    return mean_loss + (rho*cVaR)

def solve_SCP_SAA(k, S, time_limit):
    # hardcoded for now...
    rho = 10 
    CVaR_alpha = 0.20
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
    # Combine x, tau and theta into 1 single solution vector
    x_value = np.concatenate((theta.value,np.array([None]),tau.value,x.value))
    return(x_value, prob.value)

def uncertain_constraint_SAA(S, x):
    # hardcoded for now...
    rho = 10
    CVaR_alpha = 0.20
    a_1 = -1
    b_1 = rho
    a_2 = -1 - (rho/CVaR_alpha)
    b_2 = rho*(1 - (1/CVaR_alpha))
    N = S.shape[0]
    
    # Assume that x[1] contains gamma (vec of length N) and x[2] contains tau
    # Contrary to other applications, we have 2 uncertain constraints (because of max)
    # the max should be <= 0
    constr1 = a_1*(S @ x[3:]) + b_1*x[2]
    constr2 = a_2*(S @ x[3:]) + b_2*x[2]
    return np.maximum(constr1, constr2) - x[0]

# Set parameter values (as in Kuhn paper)
k = 10
rho = 10
CVaR_alpha = 0.20
risk_measure = 'exp_constraint_leq' # options: 'chance_constraint', 'exp_constraint'
alpha = 0.50
beta = 0
N_total = 3000 # 30, 300, 3000
N_train = int(N_total / 2)
N_test = N_total - N_train
num_obs_per_bin = max(N_test / 50, 5)

# Set other parameter values
par = 1
phi_div = phi.mod_chi2_cut
phi_dot = 2
numeric_precision = 1e-6 # To correct for floating-point math operations

# Get generated data
random_seed = 0
data = generate_data_2(random_seed, k, N_total)   
data_train, data_test = train_test_split(data, train_size=(N_train/N_total),
                                         shuffle=True, random_state=random_seed)
# data_train = generate_data_2(random_seed, k, N_train)  
# data_test = generate_data_2(random_seed, k, N_test)  

time_limit_search = 1*60 # in seconds (time provided to search algorithm)
time_limit_mosek = 10*60 # in seconds (for larger MIP / LP solves)
time_limit_solve = 10*60 # in seconds (for individuals solves of SCP)
max_nr_solutions = 1000 # for easy problems with long time limits, we may want extra restriction
add_remove_threshold = 0.00 # This determines when randomness is introduced in add/removal decision
use_tabu = False # Determines whether the tabu list are used in the search

add_strategy = 'random_vio'
remove_strategy = 'random_any'
clean_strategy = (999999, 'all_inactive') # set arbitrarily high such that it never occurs

# Alters the Sampled Convex Problem
solve_SCP = solve_SCP_SAA
uncertain_constraint = uncertain_constraint_SAA

for alpha in [0.6, 0.4, 0.2, 0.01]:
    (runtime, num_iter, solutions, 
     best_sol, pareto_solutions) = rs.gen_and_eval_alg(data_train, data_test, beta, alpha, time_limit_search, time_limit_solve, 
                                                        max_nr_solutions, add_strategy, remove_strategy, clean_strategy, 
                                                        add_remove_threshold, use_tabu,
                                                        phi_div, phi_dot, numeric_precision,
                                                        solve_SCP, uncertain_constraint, check_robust,
                                                        risk_measure, random_seed, 
                                                       num_obs_per_bin, None, emp_eval_obj, analytic_out_perf)
    
    if best_sol is not None:                                                   
        obj = best_sol['obj']
        print("obj_S    : " + f'{round(-obj,3):.3f}')
        eval_true_obj = analytic_out_perf(best_sol['sol'], 10, 0.20)
        print("obj_true : " + f'{round(eval_true_obj,3):.3f}')                                                  


