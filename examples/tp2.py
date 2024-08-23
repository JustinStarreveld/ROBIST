# import external packages
import numpy as np
import cvxpy as cp
import time

# problem specific functions:
def generate_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    dim_z = kwargs.get('dim_x',2)
    data = np.random.uniform(-1,1,size = (N,dim_z))   
    return data 

def solve_SCP(S, **kwargs):
    setup_time_start = time.time()
    dim_x = kwargs.get('dim_x', 2)
    x = cp.Variable(dim_x, nonneg = True)
    constraints = []
    for s in range(len(S)):
        constraints.append(cp.sum(cp.multiply(S[s], x)) <= 1)
    constraints.append(x <= dim_x)
    constraints.append(1 + cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1])
    obj = cp.Minimize(- cp.sum(x)) # formulate as a minimization problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 5*60) - (time.time() - setup_time_start)
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    primal_solution = x.value
    obj = prob.value
    if kwargs.get('get_dual_sol', False):
        dual_solution = []
        for s in range(len(S)):
            dual_solution.append(constraints[s].dual_value)
        return primal_solution, obj, dual_solution
    return primal_solution, obj

def solve_toyproblem_true_prob(dim_x, risk_param_epsilon):
    x = cp.Variable(dim_x, nonneg = True)
    constraints = [(1-2*(1-risk_param_epsilon))*x[dim_x-1] + 1 >= 0, 
                   1 + cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1], 
                   x <= dim_x]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.GUROBI)
    return prob.value

def unc_func(x, data, **kwargs):
    return (np.dot(data,x)) - 1

def get_true_prob(x, dim_x):
    return (1/2+1/(2*x[dim_x-1]))

