# import external packages
import numpy as np
import cvxpy as cp
import mosek
from sklearn.model_selection import train_test_split
import time

# import internal packages
import phi_divergence as phi
from iter_gen_and_eval_alg import iter_gen_and_eval_alg

# problem specific functions:
def generate_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    k = kwargs.get('k',2)
    data = np.random.uniform(-1,1,size = (N,k)) # generates N random scenarios    
    return data 

def generate_data_with_nominal(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    k = kwargs.get('k',2)
    data_nominal = np.array([[0] * k])
    data = np.random.uniform(-1,1,size = (N-1,k)) # generate N-1 scenarios
    data = np.concatenate((data_nominal,data)) # add nominal case to training data
    return data

def solve_P_SCP(S, **kwargs):
    k = kwargs.get('k', 2)
    x = cp.Variable(k, nonneg = True)
    constraints = [cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    for s in range(len(S)):
        constraints.append(cp.multiply(S[s], x) - 1 <= 0)
    obj = cp.Minimize(- cp.sum(x)) # have to make it a minimization problem!!
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 15*60)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def unc_func(x, data, **kwargs):
    return (np.dot(data,x)) - 1

def analytic_eval(x, problem_info):
    k = problem_info['k']
    return(1/2+1/(2*x[k-1]))
    
def get_true_prob(x, k):
    return(1/2+1/(2*x[k-1]))
    
def solve_toyproblem_true_prob(beta, k):
    x = cp.Variable(k, nonneg = True)
    constraints = [(1-2*beta)*x[k-1] + 1 >= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)


# set parameter values
k = 2
problem_instance = {'k': k, 'time_limit': 10*60}

# generate and split data into train and test
random_seed = 0
N_total = 10000
data = generate_data(random_seed, N_total, k=k)

N_train = N_total / 2
data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)

# set our own algorithm parameter values
conf_param_alpha = 0.05
add_strategy = 'random_vio'
remove_strategy = 'random_any'

# provide functions and other info for generating & evaluating solutions
solve_SCP = solve_P_SCP
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                             'desired_rhs': 1 - 0.10}}]

# run the algorithm
alg = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                            data_train, data_test, conf_param_alpha=conf_param_alpha,
                            add_strategy=add_strategy ,remove_strategy=remove_strategy,
                            verbose=True)

stop_criteria={'max_elapsed_time': 0.5*60} # in seconds (time provided to search algorithm)

(best_sol, runtime, num_iter, pareto_frontier, S_history) = alg.run(stop_criteria=stop_criteria)






















