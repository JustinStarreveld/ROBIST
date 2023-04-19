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

def generate_unc_param_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    num_stores = len(kwargs.get('locations'))
        
    # generate random demand (using budget unc set)
    max_demand = kwargs.get('max_demand', 20)
    budget_demand = kwargs.get('budget_demand', 20*math.sqrt(num_stores))
    d = np.empty((N, num_stores))

    # Naive MC sampling
    # valid_scenarios = 0
    # while valid_scenarios < N:
    #     d_gen = np.random.uniform(0, max_demand, num_stores)
    #     if np.sum(d_gen) <= budget_demand:
    #         d[valid_scenarios] = d_gen
    #         valid_scenarios += 1

    # Hit-and-run sampler (Smith, 1984)
    d_0 = np.zeros(num_stores)
    valid_scenarios = 0
    while valid_scenarios < N:
        rand_dir = np.random.uniform(0, 1, num_stores)
        L_max = (budget_demand - np.sum(d_0))/np.sum(rand_dir)        
        for j,dir_j in enumerate(rand_dir):
            L_max = min(L_max, (max_demand - d_0[j])/dir_j)
                
        rand_L = np.random.uniform(0, L_max)
        d_gen = d_0 + rand_dir * rand_L
        d[valid_scenarios] = d_gen
        valid_scenarios += 1

    # generate random transport costs from each node i to each node j
    # similar to euclidean distance, but with some variation
    rho = 0.10 # controls the magnitude of variation
    x,y = problem_instance['locations'].T
    v = np.empty((N, num_stores, num_stores))
    for i in range(num_stores-1):
        for j in range(i+1, num_stores):
            euc_dist = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            for scen in range(N):
                v[scen][i][j] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)
                v[scen][j][i] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)

    data = list(zip(d,v))
    return data

def get_fixed_param_data(num_stores, random_seed=0):
    # get normal instance data, from Bertsimas & de Ruiter (2016)
    np.random.seed(random_seed)
    
    # uniform on [0,100]^2
    locations_x = np.random.randint(0, 100, num_stores)
    locations_y = np.random.randint(0, 100, num_stores)
    locations = np.array(list(zip(locations_x, locations_y)), dtype=object)

    storage_cost = 20
    capacity_store = 20
    
    max_demand = 20
    budget_demand = 20*math.sqrt(num_stores)

    param_dict = {'locations':locations, 'storage_cost':storage_cost, 'capacity_store': capacity_store, 
                  'max_demand': max_demand, 'budget_demand': budget_demand}
    return param_dict

def solve_SCP(S, **kwargs):
    # get fixed parameter values
    storage_cost = kwargs['storage_cost']
    capacity_store = kwargs['capacity_store']
    
    # unzip uncertain parameters
    d = np.array([i[0] for i in S])
    v = np.array([i[1] for i in S])
    # d,v = S.T
    
    # get dimensions of problem
    num_stores = len(d[0])
    num_scen = len(d)

    # create variables
    x = cp.Variable(num_stores, nonneg = True)
    B = cp.Variable(1)
    y = {} # "hack" to allow 3-dimensional vars, see: https://github.com/cvxpy/cvxpy/issues/198
    for s in range(num_scen):
        y[s] = cp.Variable((num_stores, num_stores), nonneg = True)

    # set up problem
    setup_time_start = time.time()
    constraints = []
    for s in range(num_scen):
        transport_costs = cp.sum(cp.multiply(v[s], y[s]))
        constraints.append(transport_costs <= B)
    
    for s in range(num_scen):
        #TODO: check if correct
        y_in = cp.sum(y[s], axis = 0)
        y_out = cp.sum(y[s], axis = 1)
        constraints.append(x + y_in - y_out >= d[s])
    
    constraints.append(x <= capacity_store)
    init_allocation_costs = storage_cost * cp.sum(x)
    obj = cp.Minimize(init_allocation_costs + B)
    prob = cp.Problem(obj, constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)

    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    x_value = [x.value, B.value] # Combine x and B into 1 single solution vector
    
    duals = np.zeros(len(S))
    for s in range(len(S)):
        duals[s] = constraints[s].dual_value

    return x_value, prob.value, duals

def unc_con_func(x, data, **kwargs):
    # get fixed parameter values
    storage_cost = kwargs['storage_cost']
    capacity_store = kwargs['capacity_store']
    time_limit = kwargs.get('time_limit', 2*60*60)

    # unzip uncertain parameters
    d = np.array([i[0] for i in data])
    v = np.array([i[1] for i in data])
    
    # get dimensions of problem
    num_stores = len(d[0])
    num_scen = len(d)

    # for each scenario we re-solve an optimization problem to determine the adaptive decisions y
    recourse_value = np.zeros(num_scen)
    for s in range(num_scen):
        x_fixed = x[0]
        B_fixed = x[1][0]
        y = cp.Variable((num_stores, num_stores), nonneg = True)
        constraints = []
        y_in = cp.sum(y, axis = 0)
        y_out = cp.sum(y, axis = 1)
        constraints.append(x_fixed + y_in - y_out >= d[s])
        transport_costs = cp.sum(cp.multiply(v[s], y))
        obj = cp.Minimize(transport_costs)
        prob = cp.Problem(obj, constraints)

        prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
        if prob.value is None: # implies that the problem is infeasible, demand cannot be satisfied
            recourse_value[s] = float('inf')
        else:
            recourse_value[s] = prob.value - B_fixed
    
    return recourse_value

def eval_x_OoS(x, data, eval_unc_constr, **kwargs):
    unc_obj_func = eval_unc_constr['function']
    evals = unc_obj_func(x, data, **kwargs)  
    p_vio = sum(evals>1e-6) / len(data) 
    return p_vio


random_seed = 0
num_stores = 5
problem_instance = get_fixed_param_data(num_stores, random_seed=random_seed)


solve_SCP = solve_SCP
conf_param_alpha = 0.05
risk_param_epsilon = 0.01

eval_unc_obj = None

eval_unc_constr = {'function': unc_con_func,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}

# Generate extra out-of-sample (OoS) data
random_seed_OoS = 1234
N_OoS = int(1e3)
data_OoS = generate_unc_param_data(random_seed_OoS, N_OoS, **problem_instance)

N = 1
data = generate_unc_param_data(random_seed, N, **problem_instance)
x, obj, duals = solve_SCP(data, **problem_instance)
p_vio = eval_x_OoS(x, data_OoS, eval_unc_constr, **problem_instance)
print(N, obj, p_vio)

N = 100
data = generate_unc_param_data(random_seed, N, **problem_instance)
x, obj, duals = solve_SCP(data, **problem_instance)
p_vio = eval_x_OoS(x, data_OoS, eval_unc_constr, **problem_instance)
print(N, obj, p_vio)

# debug=1














