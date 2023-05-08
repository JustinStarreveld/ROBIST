# import external packages
import numpy as np
import cvxpy as cp
import gurobipy
from sklearn.model_selection import train_test_split
import time
import math
import itertools

# import internal packages
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

def generate_unc_param_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    num_stores = len(kwargs.get('locations'))
        
    # generate random demand (using budget unc set)
    max_demand = kwargs.get('max_demand')
    budget_demand = kwargs.get('budget_demand')
    demand = np.empty((N, num_stores))

    # Naive MC sampling
    # valid_scenarios = 0
    # while valid_scenarios < N:
    #     d_gen = np.random.uniform(0, max_demand, num_stores)
    #     if np.sum(d_gen) <= budget_demand:
    #         demand[valid_scenarios] = d_gen
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
        demand[valid_scenarios] = d_gen
        valid_scenarios += 1

    # generate random transport costs from each node i to each node j
    # similar to euclidean distance, but with some variation
    rho = kwargs.get('rho')
    x,y = problem_instance['locations'].T
    transport_costs = np.zeros((N, num_stores, num_stores))
    for i in range(num_stores-1):
        for j in range(i+1, num_stores):
            euc_dist = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            for scen in range(N):
                transport_costs[scen][i][j] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)
                transport_costs[scen][j][i] = np.random.uniform((1-rho)*euc_dist, (1+rho)*euc_dist)

    data = list(zip(demand,transport_costs))
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
    rho = 0.10 # controls the magnitude of variation

    param_dict = {'locations':locations, 'storage_cost':storage_cost, 'capacity_store': capacity_store, 
                  'max_demand': max_demand, 'budget_demand': budget_demand, 'rho': rho}
    return param_dict

def solve_SCP(S, **kwargs):
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
    budget = cp.Variable(1)
    y = {} 
    for s in range(num_scen):
        y[s] = cp.Variable((num_stores, num_stores), nonneg = True)

    # set up problem
    setup_time_start = time.time()
    constraints = []
    for s in range(num_scen):
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y[s]))
        constraints.append(total_transport_costs <= budget)
    
    for s in range(num_scen):
        y_in = cp.sum(y[s], axis = 0)
        y_out = cp.sum(y[s], axis = 1)
        constraints.append(init_allocation + y_in - y_out >= demand[s])
    
    constraints.append(init_allocation <= capacity_store)
    
    init_allocation_costs = storage_cost * cp.sum(init_allocation)
    obj = cp.Minimize(init_allocation_costs + budget)
    prob = cp.Problem(obj, constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)

    # env = gurobipy.Env()
    # env.setParam('TimeLimit', time_limit) # in seconds
    # env.setParam('Method', -1)

    # start_solve = time.time()
    prob.solve(solver=cp.GUROBI)
    # prob.solve(solver=cp.GUROBI, verbose=False, env=env)#, reoptimize=False)
    # solve_time = time.time() - start_solve
    solution = [init_allocation.value, budget.value]
    obj = prob.value
    
    duals = np.zeros(len(S))
    for s in range(len(S)):
        duals[s] = constraints[s].dual_value

    return solution, obj, duals#, solve_time

def unc_con_func(solution, data, **kwargs):
    # get fixed parameter values
    storage_cost = kwargs['storage_cost']
    capacity_store = kwargs['capacity_store']
    time_limit = kwargs.get('time_limit', 2*60*60)

    # unzip uncertain parameters
    demand = np.array([i[0] for i in data])
    transport_costs = np.array([i[1] for i in data])
    
    # get dimensions of problem
    num_stores = len(demand[0])
    num_scen = len(demand)

    # for each scenario we re-solve an optimization problem to determine the adaptive decisions y
    recourse_value = np.zeros(num_scen)
    for s in range(num_scen):
        x_fixed = solution[0]
        budget_fixed = solution[1][0]
        y = cp.Variable((num_stores, num_stores), nonneg = True)
        constraints = []
        y_in = cp.sum(y, axis = 0)
        y_out = cp.sum(y, axis = 1)
        constraints.append(x_fixed + y_in - y_out >= demand[s])
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y))
        obj = cp.Minimize(total_transport_costs)
        prob = cp.Problem(obj, constraints)

        prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
        if prob.value is None: # implies that the problem is infeasible, demand cannot be satisfied
            recourse_value[s] = float('inf')
        else:
            recourse_value[s] = prob.value - budget_fixed
    
    return recourse_value

def eval_x_OoS(solution, data, unc_func, **kwargs):
    evals = unc_func(solution, data, **kwargs)  
    p_vio = sum(evals>1e-6) / len(data) 
    return p_vio

def determine_N_vayanos2012(dim_uncertainty, risk_param_epsilon, conf_param_alpha, degree_dr):
    # first determine number of decision variables of decision rule approximation RB
    # we assume to only be using algebraic polynomials
    dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    num_recourse_vars = (num_stores**2) * dim_monomials
    total_num_vars = (num_stores+1) + num_recourse_vars
    
    # then determine minimum N required to achieve desired level of robustness
    N_min = util.determine_campi_N_min(total_num_vars, 1-risk_param_epsilon, conf_param_alpha)
    return N_min

def solve_SCP_vayanos2012(S, degree_dr, **kwargs):
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
    budget = cp.Variable(1)
    
    # we assume to only be using algebraic polynomials for decision rules
    dim_uncertainty = num_stores + num_stores**2
    dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    dr_coef_matrix = cp.Variable(shape = (num_stores**2, dim_monomials))
    l_d = 0
    u_d = kwargs['max_demand']
    l_v = 0
    u_v = (1+kwargs['rho'])*math.sqrt((100)**2 + (100)**2)
    y = {}
    for s in range(num_scen):
        scaled_d = 2*(demand[s] - l_d)/(u_d - l_d) - 1
        scaled_v = 2*(transport_costs[s] - l_v)/(u_v - l_v) - 1
        xi = np.concatenate((scaled_d, scaled_v.reshape(-1)), axis=None)
        
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
    constraints = []
    for s in range(num_scen):
        total_transport_costs = cp.sum(cp.multiply(transport_costs[s], y[s]))
        constraints.append(total_transport_costs <= budget)
        constraints.append(y[s] >= 0)
    
    for s in range(num_scen):
        y_in = cp.sum(y[s], axis = 0)
        y_out = cp.sum(y[s], axis = 1)
        constraints.append(init_allocation + y_in - y_out >= demand[s])
    
    constraints.append(init_allocation <= capacity_store)
    
    init_allocation_costs = storage_cost * cp.sum(init_allocation)
    obj = cp.Minimize(init_allocation_costs + budget)
    prob = cp.Problem(obj, constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 1*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)

    # env = gurobipy.Env()
    # env.setParam('TimeLimit', time_limit) # in seconds
    # # env.setParam('Method', 5)

    start_solve = time.time()
    # prob.solve(solver=cp.GUROBI, verbose=False, env=env)
    prob.solve(solver=cp.GUROBI, verbose=False)
    
    solve_time = time.time() - start_solve
    solution = [init_allocation.value, budget.value, dr_coef_matrix.value]
    obj = prob.value
    return solution, obj, solve_time


def eval_p_OoS_vayanos2012(solution, data, degree_dr, **kwargs):
    # unzip uncertain parameters
    demand = np.array([i[0] for i in data])
    transport_costs = np.array([i[1] for i in data])
    num_scen = len(demand)
    
    # unzip solution
    init_allocation = solution[0]
    budget = solution[1][0]
    dr_coef_matrix = solution[2]
    
    # eval ldr
    dim_uncertainty = num_stores + num_stores**2
    l_d = 0
    u_d = kwargs['max_demand']
    l_v = 0
    u_v = (1+kwargs['rho'])*math.sqrt((100)**2 + (100)**2)
    evals = np.zeros(num_scen)
    for s in range(num_scen):
        scaled_d = 2*(demand[s] - l_d)/(u_d - l_d) - 1
        scaled_v = 2*(transport_costs[s] - l_v)/(u_v - l_v) - 1
        xi = np.concatenate((scaled_d, scaled_v.reshape(-1)), axis=None)
        
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
        
        y = (dr_coef_matrix @ basis_vector).reshape((num_stores, num_stores))
        y_in = np.sum(y, axis = 0)
        y_out = np.sum(y, axis = 1)
        
        demand_con_max = np.max(demand[s] - (init_allocation + y_in - y_out))
        total_transport_costs = np.sum(np.multiply(transport_costs[s], y))
        evals[s] = max(demand_con_max, total_transport_costs - budget)
        
    p_vio = sum(evals>1e-6) / num_scen
    return p_vio
    

random_seed = 0

solve_SCP = solve_SCP
conf_param_alpha = 0.05
# risk_param_epsilon = 0.05
# risk_param_epsilon = 0.10


stop_criteria={'max_num_iterations': 100}
eval_unc_obj = None

# m_settings = [2]
num_stores = 2
risk_param_epsilon_settings = [0.10]#[0.05, 0.01]

# Generate extra out-of-sample (OoS) data
random_seed_OoS = 1234
N_OoS = int(10)

for risk_param_epsilon in risk_param_epsilon_settings:
    eval_unc_constr = [{'function': unc_con_func,
                        'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                                'desired_rhs': 1 - risk_param_epsilon}}]
    
    problem_instance = get_fixed_param_data(num_stores, random_seed=random_seed)
    
    data_OoS = generate_unc_param_data(random_seed_OoS, N_OoS, **problem_instance)
    
    dim_uncertainty = num_stores + num_stores**2

    degree_dr = 1
    
    dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    num_recourse_vars = (num_stores**2) * dim_monomials
    num_vars_dr1 = (num_stores+1) + num_recourse_vars
    
    # N_dr1 = determine_N_vayanos2012(dim_uncertainty, risk_param_epsilon, conf_param_alpha, degree_dr)
    N_dr1 = 10
    
    data_dr1 = generate_unc_param_data(random_seed, N_dr1, **problem_instance)
    
    # # print(dim_monomials, num_vars_dr1, N_dr1)
    
    x_dr1, obj_dr1, time_dr1 = solve_SCP_vayanos2012(data_dr1, degree_dr, **problem_instance)
    time_dr1 = format(round(time_dr1, 0), '.0f')
    # gap1 = 100* abs(obj_scp - obj_dr1) / obj_scp
    
    OoS_p_vio_dr1_dr_2 = eval_p_OoS_vayanos2012(x_dr1, data_dr1, degree_dr, **problem_instance)
    OoS_p_vio_dr1_dr = eval_p_OoS_vayanos2012(x_dr1, data_OoS, degree_dr, **problem_instance)
    
    # OoS_p_vio_dr1_fa = eval_x_OoS(x_dr1, data_OoS, unc_con_func, **problem_instance)

    print(num_vars_dr1, N_dr1, time_dr1, obj_dr1, OoS_p_vio_dr1_dr, OoS_p_vio_dr1_dr_2)
    # print()
    
    # degree_dr = 2
    
    # dim_monomials = math.comb(dim_uncertainty + degree_dr, degree_dr)
    # num_recourse_vars = (num_stores**2) * dim_monomials
    # num_vars_dr2 = (num_stores+1) + num_recourse_vars
    
    # N_dr2 = determine_N_vayanos2012(dim_uncertainty, risk_param_epsilon, conf_param_alpha, degree_dr)
    # data_dr2 = generate_unc_param_data(random_seed, N_dr2, **problem_instance)
    
    # # print(dim_monomials, num_vars_dr2, N_dr2)
    
    # x_dr2, obj_dr2, time_dr2 = solve_SCP_vayanos2012(data_dr2, degree_dr, **problem_instance)
    # time_dr2 = format(round(time_dr2, 0), '.0f')
    # # gap1 = 100* abs(obj_scp - obj_dr2) / obj_scp
    # # OoS_p_vio_dr2 = eval_p_OoS_vayanos2012(x_dr2, data_OoS, degree_dr, **problem_instance)
    # OoS_p_vio_dr2 = eval_x_OoS(x_dr2, data_OoS, unc_con_func, **problem_instance)
    
    
    # print(num_vars_dr2, N_dr2, time_dr2, obj_dr2, OoS_p_vio_dr2)
    # print("--------------------------------------------------------")
    
    # N = N_dr1
    # N_train = math.floor(N/2)
    # N_test = N - N_train
    
    # data = data_dr1   
    # data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)
    
    # robist = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
    #                                 data_train, data_test, conf_param_alpha=conf_param_alpha,
    #                                 verbose=False)
    
    # (best_sol, runtime_robist, num_iter, pareto_frontier, S_history, all_solutions_robist) = robist.run(stop_criteria=stop_criteria, store_all_solutions=True)
    
    # lb_robist = best_sol['feas'][0]
    # obj_robist = best_sol['obj']
    # OoS_p_vio_robist = eval_x_OoS(best_sol['sol'], data_OoS, unc_con_func, **problem_instance)
    
    # print(N_train, N_test, runtime_robist, obj_robist, lb_robist, OoS_p_vio_robist)
    
    # print(N, " ", num_vars_0, time_scp, 
    #       " ", num_vars_1, time_dr1, format(round(gap1,3),'.3f'))#,
    #       # " ", num_vars_1, time_dr1_2, format(round(gap1_2,3),'.3f'))
    #       # " ", num_vars_2, time_dr2, format(round(gap2,3),'.3f'))
  
    

















































