"""
Basic functions for weighted distribution problem
"""  
# external imports
import numpy as np
import cvxpy as cp
import time

# problem specific functions:
def generate_unc_param_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    scale_dim_problem = kwargs.get('scale_dim_problem', 1)
    m = 5*scale_dim_problem
    n = 10*scale_dim_problem
    
    # generate demand vector param
    d_care = np.array([25, 38, 18, 39, 60, 35, 41, 22, 74, 30])
    d_nom = ()
    for i in range(scale_dim_problem):
        d_i = d_care
        d_nom = d_nom + tuple(d_i.reshape(1, -1)[0])
    d = np.random.default_rng(seed=random_seed).dirichlet(d_nom, N) * sum(d_nom)
    
    # generate production efficiency param
    p_care = np.array([[5.0, 7.6, 3.6, 7.8, 12.0, 7.0, 8.2, 4.4, 14.8, 6.0],
                      [3.8, 5.8, 2.8, 6.0, 9.2, 5.4, 6.3, 3.4, 11.4, 4.6],
                      [2.3, 3.5, 1.6, 3.5, 5.5, 3.2, 3.7, 2.0, 6.7, 2.7],
                      [2.6, 4.0, 1.9, 4.1, 6.3, 3.7, 4.3, 2.3, 7.8, 3.2],
                      [2.4, 3.6, 1.7, 3.7, 5.7, 3.3, 3.9, 2.1, 7.0, 2.9]])
    if scale_dim_problem > 1:
        p_nom = np.block([[p_care for i in range(scale_dim_problem)] for j in range(scale_dim_problem)])
    else:
        p_nom = p_care
    p = np.random.random_sample(size = (N,m,n)) * (p_nom*1.05 - p_nom*0.95) + (p_nom*0.95)
    # comb = list(zip(d,p))
    # data = np.vstack(comb)
    data = list(zip(d,p))
    return data

def get_fixed_param_data(random_seed, **kwargs):
    np.random.seed(random_seed)
    scale_dim_problem = kwargs.get('scale_dim_problem', 1)
    
    # fixed parameter values from Care (2014)
    C_care = np.array([[1.8, 2.2, 1.5, 2.2, 2.6, 2.1, 2.2, 1.7, 2.8, 1.9],
                        [1.6, 1.9, 1.3, 1.9, 2.3, 1.9, 2.0, 1.5, 2.5, 1.7],
                        [1.2, 1.5, 1.0, 1.5, 1.9, 1.4, 1.6, 1.1, 2.0, 1.3],
                        [1.3, 1.6, 1.1, 1.6, 2.0, 1.5, 1.7, 1.2, 2.2, 1.4],
                        [1.2, 1.5, 1.0, 1.6, 1.9, 1.5, 1.6, 1.1, 2.1, 1.3]])

    A_care = np.array([10, 13, 22, 19, 21])
    C_tilde_care = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
    U_care = np.array([1.5, 1.8, 1.2, 1.9, 2.2, 1.8, 1.9, 1.4, 2.4, 1.6])
    
    if scale_dim_problem > 1:
        max_deviation = 0.10 # represents max deviation from care values
        C = np.block([[np.random.random_sample(size = (5,10)) * (C_care*(1+max_deviation) - C_care*(1-max_deviation)) + (C_care*(1-max_deviation)) for i in range(scale_dim_problem)] 
                      for j in range(scale_dim_problem)])
        
        A = np.round(np.block([np.random.random_sample(size = 5) * (A_care*(1+max_deviation) - A_care*(1-max_deviation)) + (A_care*(1-max_deviation)) for i in range(scale_dim_problem)]))
        
        C_tilde = np.block([np.random.random_sample(size = 10) * (C_tilde_care*(1+max_deviation) - C_tilde_care*(1-max_deviation)) + (C_tilde_care*(1-max_deviation)) for i in range(scale_dim_problem)])
        
        U = np.block([np.random.random_sample(size = 10) * (U_care*(1+max_deviation) - U_care*(1-max_deviation)) + (U_care*(1-max_deviation)) for i in range(scale_dim_problem)])
        param_dict = {'C':C, 'A':A, 'C_tilde': C_tilde, 'U': U}
    else:
        param_dict = {'C':C_care, 'A':A_care, 'C_tilde': C_tilde_care, 'U': U_care}
    
    return param_dict

def solve_SCP(S, **kwargs):
    # get fixed parameter values
    C = kwargs['C']
    A = kwargs['A']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    
    # unzip uncertain parameters
    d = np.array([i[0] for i in S])
    p = np.array([i[1] for i in S])
    
    # get dimensions of problem
    m,n = p[0].shape
    num_scen = len(d)
    
    # create variables
    theta = cp.Variable(1)
    y = cp.Variable((m, n), nonneg = True)
    
    # set up problem
    setup_time_start = time.time()
    fixed_costs = cp.sum(cp.multiply(C, y))
    constraints = []
    for s in range(num_scen):
        prod_s = cp.sum(cp.multiply(p[s], y), axis=0)
        unc_inv_cost_s = C_tilde.T @ cp.pos(prod_s - d[s])
        unc_rev_s = U.T @ cp.minimum(prod_s, d[s])

        constraints.append(-(unc_rev_s - fixed_costs - unc_inv_cost_s) - theta <= 0)
    
    constraints.append(cp.sum(y, axis=1) <= A)
    obj = cp.Minimize(theta)
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    primal_solution = [theta.value, y.value]
    obj = prob.value
    if kwargs.get('get_dual_sol', False):
        dual_solution = []
        for s in range(len(S)):
            dual_solution.append(constraints[s].dual_value)
        return primal_solution, obj, dual_solution
    return primal_solution, obj

def unc_function(solution, data, **kwargs):
    # extract values
    C = kwargs['C']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    d = np.array([i[0] for i in data])
    p = np.array([i[1] for i in data])
    m,n = p[0].shape
    y = solution[1]
    
    # compute obj function value:
    fixed_cost = np.sum(np.multiply(C, y))
    prod = [np.einsum('jk,jk->k', p[s], y) for s in range(len(data))]
    inventory_cost = np.array([np.dot(C_tilde, np.maximum(prod[s] - d[s],0)) for s in range(len(data))]) 
    revenue = np.array([np.dot(U, np.minimum(prod[s], d[s])) for s in range(len(data))]) 
    
    return - (revenue - fixed_cost - inventory_cost)

def unc_constraint(solution, data, **kwargs):
    f_evals = unc_function(solution, data, **kwargs)
    theta = solution[0][0]
    return f_evals - theta   

def eval_x_OoS(solution, data, unc_function, risk_param_epsilon, numeric_precision=1e-6, **kwargs):
    f_evals = unc_function(solution, data, **kwargs)  
    theta = solution[0][0]
    p_vio = sum(f_evals-theta>-numeric_precision) / len(data) 
    VaR = np.quantile(-f_evals, risk_param_epsilon, method='inverted_cdf')
    return p_vio, VaR


    
    



