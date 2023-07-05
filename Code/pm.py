"""
Basic functions for portfolio management problem
"""  
# external imports
import numpy as np
import cvxpy as cp
import time
import math

# Problem specific functions:
def generate_data_natarajan2008(random_seed, N, **kwargs):
    k = kwargs['dim_x']    
    np.random.seed(random_seed)
    gamma = np.fromiter((((1/2)*(1 + (i/(k+1)))) for i in range(1,k+1)), float)
    return_pos = np.fromiter(((math.sqrt((1-gamma[i])*gamma[i])/gamma[i]) for i in range(0,k)), float)
    return_neg = np.fromiter((-(math.sqrt((1-gamma[i])*gamma[i])/(1-gamma[i])) for i in range(0,k)), float)
    data = np.empty([N,k])
    for n in range(0, N):
        for i in range(0, k):
            prob = np.random.uniform()
            if prob <= gamma[i]:
                data[n, i] = return_pos[i]
            else:
                data[n, i] = return_neg[i]
    return data 

def generate_data_mohajerin2018(random_seed, N, **kwargs):
    k = kwargs['dim_x']
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

def solve_SCP(S, **kwargs):
    setup_time_start = time.time()
    dim_x = kwargs['dim_x']
    x = cp.Variable(dim_x, nonneg = True)
    theta = cp.Variable(1)
    constraints = []
    for s in range(len(S)):
        constraints.append(- cp.sum(cp.multiply(S[s], x)) <= theta)
    constraints.append(cp.sum(x) == 1)
    obj = cp.Minimize(theta) # must formulate as min problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 5*60) - (time.time() - setup_time_start)
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    primal_solution = [x.value, theta.value]
    obj = prob.value
    if kwargs.get('get_dual_sol', False):
        dual_solution = []
        for s in range(len(S)):
            dual_solution.append(constraints[s].dual_value)
        return primal_solution, obj, dual_solution
    return primal_solution, obj
    
def unc_function(solution, data, **kwargs):
  return - np.dot(data, solution[0]) 

def eval_OoS(solution, obj, data, eval_unc_obj, **kwargs):
    unc_obj_func = eval_unc_obj['function']
    desired_rhs = eval_unc_obj['info']['desired_rhs']
    evals = unc_obj_func(solution, data, **kwargs)  
    p_vio = sum(evals>(obj+(1e-6))) / len(data) 
    VaR = - np.quantile(evals, desired_rhs, method='inverted_cdf')
    test = -evals <= VaR
    CVaR = sum(-evals[test]) / sum(test)
    return p_vio, VaR, CVaR









