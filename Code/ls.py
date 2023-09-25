"""
Basic functions for the 2-stage adaptive lot-sizing problem
"""  
# external imports
import numpy as np
import cvxpy as cp
import math
import time
import itertools
import warnings

# internal imports
from scen_opt_methods import determine_cam2008_N_min

def generate_unc_param_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    num_stores = len(kwargs.get('locations'))
        
    # generate random demand (using budget unc set)
    demand_max = kwargs.get('demand_max')
    demand_budget = kwargs.get('demand_budget')
    demand = np.empty((N, num_stores))

    # Hit-and-run sampler (Smith, 1984)
    d_0 = np.zeros(num_stores)
    valid_scenarios = 0
    while valid_scenarios < N:
        rand_dir = np.random.uniform(0, 1, num_stores)
        L_max = (demand_budget - np.sum(d_0))/np.sum(rand_dir)        
        for j,dir_j in enumerate(rand_dir):
            L_max = min(L_max, (demand_max - d_0[j])/dir_j)
                
        rand_L = np.random.uniform(0, L_max)
        d_gen = d_0 + rand_dir * rand_L
        demand[valid_scenarios] = d_gen
        valid_scenarios += 1

    # generate random transport costs from each node i to each node j
    # similar to euclidean distance, but with +/- rho% variation
    rho = kwargs.get('rho')
    x,y = kwargs.get('locations').T
    transport_costs = np.zeros((N, num_stores, num_stores))
    for i in range(num_stores-1):
        for j in range(i+1, num_stores):
            euc_dist = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            for scen in range(N):
                transport_costs[scen][i][j] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)
                transport_costs[scen][j][i] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)

    data = list(zip(demand,transport_costs))
    return data

def get_known_param_data(num_stores, random_seed=0):
    # get normal instance data, from Bertsimas & de Ruiter (2016)
    np.random.seed(random_seed)
    
    # uniform on [0,100]^2
    locations_x = np.random.randint(0, 100, num_stores)
    locations_y = np.random.randint(0, 100, num_stores)
    locations = np.array(list(zip(locations_x, locations_y)), dtype=object)

    storage_cost = 20
    capacity_store = 20
    demand_max = 20
    demand_budget = 20*math.sqrt(num_stores)
    rho = 0.10 # controls the magnitude of variation

    param_dict = {'locations':locations, 'storage_cost':storage_cost, 'capacity_store': capacity_store, 
                  'demand_max': demand_max, 'demand_budget': demand_budget, 'rho': rho}
    return param_dict

def solve_SCP(S, **kwargs):
    # get known parameter values
    storage_cost = kwargs['storage_cost']
    capacity_store = kwargs['capacity_store']
    
    # unzip uncertain parameters
    demand = np.array([i[0] for i in S])
    transport_costs = np.array([i[1] for i in S])
    
    # get dimensions of problem
    num_stores = len(demand[0])
    num_scen = len(demand)

    # create variables
    init_allocation = cp.Variable(num_stores, nonneg = True)
    obj_theta= cp.Variable(1)
    y = {} 
    for s in range(num_scen):
        y[s] = cp.Variable((num_stores, num_stores), nonneg = True)

    # set up problem
    setup_time_start = time.time()
    init_allocation_costs = storage_cost * cp.sum(init_allocation)
    constraints = []
    for s in range(num_scen):
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y[s]))
        constraints.append(init_allocation_costs + total_transport_costs <= obj_theta)
    
    for s in range(num_scen):
        y_in = cp.sum(y[s], axis = 0)
        y_out = cp.sum(y[s], axis = 1)
        constraints.append(init_allocation + y_in - y_out >= demand[s])
    
    constraints.append(init_allocation <= capacity_store)
    
    obj = cp.Minimize(obj_theta)
    prob = cp.Problem(obj, constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)

    prob.solve(solver=cp.GUROBI)
    solution = [obj_theta.value, init_allocation.value]
    obj = prob.value

    return solution, obj

def unc_function(solution, data, **kwargs):
    # get known parameter values
    storage_cost = kwargs['storage_cost']    

    # unzip uncertain parameters
    demand = np.array([i[0] for i in data])
    transport_costs = np.array([i[1] for i in data])
    
    # get dimensions of problem
    num_stores = len(demand[0])
    num_scen = len(demand)

    # unpack solution
    obj_theta = solution[0][0]
    init_allocation = solution[1]
    init_allocation_costs = storage_cost * np.sum(init_allocation)

    # for each scenario we re-solve an optimization problem to determine the adaptive decisions y
    recourse_value = np.zeros(num_scen)
    for s in range(num_scen):
        y = cp.Variable((num_stores, num_stores), nonneg = True)
        constraints = []
        y_in = cp.sum(y, axis = 0)
        y_out = cp.sum(y, axis = 1)
        constraints.append(init_allocation + y_in - y_out >= demand[s])
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y))
        obj = cp.Minimize(total_transport_costs)
        prob = cp.Problem(obj, constraints)

        time_limit = kwargs.get('time_limit', 2*60*60)
        with warnings.catch_warnings(): # to prevent "infeasible" warning
            warnings.simplefilter("ignore")
            prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
        if prob.value is None: # implies that the problem is infeasible, demand cannot be satisfied
            recourse_value[s] = float('inf')
        else:
            recourse_value[s] = init_allocation_costs + prob.value - obj_theta
    
    return recourse_value

def eval_OoS(solution, data, unc_func, risk_param_epsilon, **kwargs):
    evals = unc_func(solution, data, **kwargs)  
    p_vio = sum(evals>1e-6) / len(data) 
    obj_theta = solution[0][0]
    VaR = np.quantile(evals+obj_theta, 1-risk_param_epsilon, method='inverted_cdf')
    
    p_demand_unmet = np.count_nonzero(evals==float('inf')) / len(data) 
    return p_vio, VaR, p_demand_unmet


def determine_N_vay2012(num_stores, risk_param_epsilon, conf_param_alpha, degree_dr):
    # first determine number of decision variables of decision rule approximation RB
    # we assume to only be using algebraic polynomials
    dim_uncertainty = num_stores + num_stores**2
    dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    num_recourse_vars = (num_stores**2) * dim_monomials
    total_num_vars = 1 + num_stores + num_recourse_vars
    
    # then determine minimum N required to achieve desired level of robustness
    N_min = determine_cam2008_N_min(total_num_vars, risk_param_epsilon, conf_param_alpha)
    return N_min, total_num_vars

def solve_SCP_vay2012(S, degree_dr, **kwargs):
    # get fixed parameter values
    storage_cost = kwargs['storage_cost']
    capacity_store = kwargs['capacity_store']
    
    # unzip uncertain parameters
    demand = np.array([i[0] for i in S])
    transport_costs = np.array([i[1] for i in S])
    
    # get dimensions of problem
    num_stores = len(demand[0])
    num_scen = len(demand)

    # create variables
    init_allocation = cp.Variable(num_stores, nonneg = True)
    obj_theta = cp.Variable(1)
    
    # we assume to only be using algebraic polynomials for decision rules
    dim_uncertainty = num_stores + num_stores**2
    dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    dr_coef_matrix = cp.Variable(shape = (num_stores**2, dim_monomials))
    demand_lb = 0
    demand_ub = kwargs['demand_max']
    transport_cost_lb = 0
    transport_cost_ub = (1+kwargs['rho'])*math.sqrt((100)**2 + (100)**2)
    y = {}
    for s in range(num_scen):
        demand_scaled = 2*(demand[s] - demand_lb)/(demand_ub - demand_lb) - 1
        scaled_v = 2*(transport_costs[s] - transport_cost_lb)/(transport_cost_ub - transport_cost_lb) - 1
        xi = np.concatenate((demand_scaled, scaled_v.reshape(-1)), axis=None)
        
        if degree_dr == 1:
            basis_vector = np.array([1] + [xi[i] for i in range(dim_uncertainty)])
        elif degree_dr == 2:
            basis_vector = [1]
            for i in range(dim_uncertainty):
                basis_vector.append(xi[i])
                basis_vector.append(xi[i]**2)
            options = [i for i in range(dim_uncertainty)]
            basis_vector +=[xi[comb[0]]*xi[comb[1]] for comb in itertools.combinations(options, 2)]
            basis_vector = np.array(basis_vector)
        else:
            print("ERROR: provided degree > 2")
        
        y[s] = cp.reshape(dr_coef_matrix @ basis_vector, shape=(num_stores, num_stores))
        
    # set up problem
    setup_time_start = time.time()
    init_allocation_costs = storage_cost * cp.sum(init_allocation)
    constraints = []
    for s in range(num_scen):
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y[s]))
        constraints.append(init_allocation_costs + total_transport_costs <= obj_theta)
        y_in = cp.sum(y[s], axis = 0)
        y_out = cp.sum(y[s], axis = 1)
        constraints.append(init_allocation + y_in - y_out >= demand[s])
        constraints.append(y[s] >= 0)
    
    constraints.append(init_allocation <= capacity_store)
    
    obj = cp.Minimize(obj_theta)
    prob = cp.Problem(obj, constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 1*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)

    start_solve = time.time()
    prob.solve(solver=cp.GUROBI, verbose=False)
    
    solve_time = time.time() - start_solve
    solution = [obj_theta.value, init_allocation.value, dr_coef_matrix.value]
    obj = prob.value
    return solution, obj, solve_time
