# import external packages
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import time
import math

# import internal packages
import phi_divergence as phi
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util


# Problem specific functions:
def generate_data(random_seed, N, **kwargs):
    k = kwargs['k']    
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

def solve_SCP(S, **kwargs):
    k = S.shape[1]
    x = cp.Variable(k, nonneg = True)
    theta = cp.Variable(1)
    constraints = [theta - (S @ x) <= 0, cp.sum(x) == 1]
    obj = cp.Minimize(-theta) # must formulate as min problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 2*60*60)    
#     prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    x_value = np.concatenate((theta.value,x.value)) # Combine x and theta into 1 single solution vector
    return(x_value, prob.value)

def unc_obj_func(x, data, **kwargs):
    return np.dot(data,x[1:]) # Assume that x[0] contains theta variable 

def eval_x_OoS(x, obj, data, eval_unc_obj, **kwargs):
    unc_obj_func = eval_unc_obj['function']
    desired_rhs = eval_unc_obj['info']['desired_rhs']
    
    evals = unc_obj_func(x, data, **kwargs)  
    p_vio = sum(evals<(-obj+(1e-6))) / len(data) 
    VaR = - np.quantile(evals, desired_rhs, method='inverted_cdf')
    return p_vio, VaR

# calafiore2012 method:
def solve_with_calafiore2012(solve_SCP, problem_instance, dim_x, data, risk_param_epsilon, conf_param_alpha, q=-1):
    start_time = time.time()
    # 1) given N, determine maximum q such that rhs of eq 12 is no greater than N
    N = len(data)
    z_tol_cal = risk_param_epsilon
    n_cal = dim_x
    beta_cal = conf_param_alpha
    
    if q == -1:
        def eval_eq_12_calafiore2012(z_tol_cal, beta_cal, q, n_cal):
            return 2/z_tol_cal * math.log(1/beta_cal) + 4/z_tol_cal * (q+n_cal)
        
        # do bisection search to find maximum q
        a = 0
        b = N - n_cal - 1
        f_b = eval_eq_12_calafiore2012(z_tol_cal, beta_cal, b, n_cal)
        
        if f_b <= N:
            q = b
        else:
            while True:
                if b-a == 1:
                    if eval_eq_12_calafiore2012(z_tol_cal, beta_cal, b, n_cal) <= N:
                        q = b
                        break
                    else:
                        q = a
                        break
                
                c = math.ceil((a+b)/2)
                f_c = eval_eq_12_calafiore2012(z_tol_cal, beta_cal, c, n_cal)
                if f_c > N:
                    b = c
                else:
                    a = c
    
    # 2) iteratively, using Lagrange multiplier-based rule, discard q scenarios
    def solve_SCP_w_duals(S, **kwargs):
        k = S.shape[1]
        x = cp.Variable(k, nonneg = True)
        theta = cp.Variable(1)
        constraints = [theta - (S @ x) <= 0, cp.sum(x) == 1]
        obj = cp.Minimize(-theta) # must formulate as min problem
        prob = cp.Problem(obj,constraints)
        time_limit = kwargs.get('time_limit', 2*60*60)    
        prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
        x_value = np.concatenate((theta.value,x.value)) # Combine x and theta into 1 single solution vector
        
        duals = constraints[0].dual_value
        return(x_value, prob.value, duals)
        
    # Start with all N scenarios and remove one-by-one
    num_removed = 0
    while num_removed < q:
        x, obj, duals = solve_SCP_w_duals(data, **problem_instance)
        scen_i = np.argmax(duals)
        data = np.delete(data, scen_i, axis=0)
        num_removed += 1
        
    # return final solution
    x, obj = solve_SCP(data, **problem_instance)
    return x, obj, (time.time() - start_time), q



# Set parameter values (as in Bertsimas paper)
k = 10
conf_param_alpha = 0.10
risk_param_epsilon = 0.10
N_total = 500 
problem_instance = {}
problem_instance['time_limit'] = 2*60*60 

# Generate extra out-of-sample (OoS) data
random_seed_OoS = 1234
N_OoS = int(1e5)
data_OoS = generate_data(random_seed_OoS, N_OoS, k=k)

# Get generated data
random_seed_settings = [i for i in range(1,11)]
q_max = -1
for random_seed in random_seed_settings:
# random_seed = 0
    data = generate_data(random_seed, N_total, k=k)
    
    x, obj, runtime, q = solve_with_calafiore2012(solve_SCP, problem_instance, k, data, risk_param_epsilon, conf_param_alpha, q=q_max)
    obj_cal2012 = - obj
    q_max = q
    
    eval_unc_obj = {'function': unc_obj_func,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                             'desired_rhs': 1-risk_param_epsilon}}
    
    p_vio_cal2012, VaR_cal2012 = eval_x_OoS(x, obj, data_OoS, eval_unc_obj, **problem_instance)
    print("seed: " + str(random_seed), N_total, q, runtime, obj_cal2012, p_vio_cal2012, VaR_cal2012)
































