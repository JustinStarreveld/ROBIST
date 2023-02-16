# import external packages
import numpy as np
import cvxpy as cp
# import mosek
from sklearn.model_selection import train_test_split
import time

# import internal packages
import phi_divergence as phi
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

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
    data = list(zip(d,p))
    data = np.array(data, dtype=object)
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

def solve_P_SCP(S, **kwargs):
    # get fixed parameter values
    C = kwargs['C']
    A = kwargs['A']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    
    # unzip uncertain parameters
    d,p = data.T
    
    # get dimensions of problem
    m,n = p[0].shape
    num_scen = len(d)
    
    # create variables
    theta = cp.Variable(1)
    y = cp.Variable((m, n), nonneg = True)
    
    # set up problem
    setup_time_start = time.time()
    constraints = []
    for s in range(num_scen):
        prod_s = cp.sum(cp.multiply(p[s], y), axis=0)
        unc_inv_cost_s = C_tilde.T @ cp.pos(prod_s - d[s])
        unc_rev_s = U.T @ cp.minimum(prod_s, d[s])

        constraints.append(unc_inv_cost_s - unc_rev_s - theta <= 0)
    
    constraints.append(cp.sum(y, axis=1) <= A)
    
    fixed_costs = cp.sum(cp.multiply(C, y))
    obj = cp.Minimize(fixed_costs + theta)
    prob = cp.Problem(obj,constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)
    
#     prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    prob.solve(solver=cp.GUROBI, verbose=False, Threads=1, TimeLimit=time_limit)
    x_value = [theta.value, y.value] # Combine y and theta into 1 single solution vector
    return (x_value, prob.value)

def unc_obj_func(x, data, **kwargs):
    # extract values
    C = kwargs['C']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    d,p = data.T
    m,n = p[0].shape
    y = x[1]
    
    # compute obj function value:
    fixed_cost = np.sum(np.multiply(C, y))
    prod = [np.einsum('jk,jk->k', p[s], y) for s in range(len(data))]
    inventory_cost = np.array([np.dot(C_tilde, np.maximum(prod[s] - d[s],0)) for s in range(len(data))]) 
    revenue = np.array([np.dot(U, np.minimum(prod[s], d[s])) for s in range(len(data))]) 
    
    return fixed_cost + inventory_cost - revenue

def eval_x_OoS(x, obj, data, eval_unc_obj, **kwargs):
    unc_obj_func = eval_unc_obj['function']
    desired_rhs = eval_unc_obj['info']['desired_rhs']
    
    evals = unc_obj_func(x, data, **kwargs)  
    p_vio = sum(evals>(obj+(1e-6))) / len(data) 
    VaR = - np.quantile(evals, desired_rhs, method='inverted_cdf')
    return p_vio, VaR

random_seed = 0
TIME_LIMIT = 1
# TIME_LIMIT = 5*60*60
scale_dim_problem = 3
problem_instance = get_fixed_param_data(random_seed, scale_dim_problem=scale_dim_problem)
problem_instance['time_limit'] = TIME_LIMIT 

# classic approach:
random_seed = 0
N = 20
# N = 10580
# N = 34918 
data = generate_unc_param_data(random_seed, N, scale_dim_problem=scale_dim_problem)

start_time = time.time()
x, obj = solve_P_SCP(data, **problem_instance)
runtime_classic = time.time() - start_time
obj_classic = - obj

print(N, runtime_classic, obj_classic)

# random_seed = 1234
# N_OoS = int(1e5)
# # N_OoS = int(2)
# data_OoS = generate_unc_param_data(1234, N_OoS, scale_dim_problem=scale_dim_problem)

# eval_unc_obj = {'function': unc_obj_func,
#                     'info': {'risk_measure': 'probability'}}
# eval_unc_obj['info']['desired_rhs'] = 1 - 0.01

# start = time.time()
# p_vio_classic, VaR_classic = eval_x_OoS(x, obj, data_OoS, eval_unc_obj, **problem_instance)
# print(("v1", (time.time() - start)))

# print(N, runtime_classic, obj_classic, p_vio_classic, VaR_classic)






































