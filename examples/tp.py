"""
Basic functions for toy problem
"""   
# external imports
import numpy as np
import cvxpy as cp
import time
import scipy.stats
import itertools

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
        constraints.append(cp.sum(cp.multiply(S[s], x)) - 1 <= 0)
    constraints.append(x<=1)
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

def unc_function(solution, data, **kwargs):
    return (np.dot(data,solution)) - 1

def eval_OoS(solution, data, numeric_precision=1e-6):
    f_evals = unc_function(solution, data)
    N_vio = sum(f_evals>-numeric_precision)
    N = len(data)
    p_feas = 1 - N_vio/N
    return p_feas


def solve_with_yan2013(dim_x, risk_param_epsilon, conf_param_alpha, data, m_j=10,
                        omega_init=0.1, step_size=0.01, store_all_solutions=False, 
                        store_pareto_solutions=False, verbose=False):
    
    def get_values_yanikoglu2013(dim_x, m_j, data):
        a = []
        for i in range(dim_x+1):
            if i == 0:
                a.append(np.array([0 for j in range(dim_x)]))
            else:
                temp = [0 for j in range(i-1)]
                temp.append(1)
                temp = temp + [0 for j in range(i, dim_x)]
                a.append(np.array(temp))
        b = 1
        lb = -1
        ub = 1
        cpt_arr = []
        for i in range(dim_x):
            cpt_arr.append(make_center(lb,ub,m_j))
        indices = np.asarray(list((itertools.product(np.arange(m_j), repeat = dim_x))))
        p = get_freq(data,lb,ub,m_j,indices)
        
        return a, b, lb, ub, cpt_arr, indices, p
    
    def make_center(lb,ub,m_j):
        delta = (ub-lb)/m_j
        center = np.arange(lb+delta/2,ub,delta)
        return(center)          
    
    def get_freq(data,lb,ub,m_j,indices):
        dim_x = len(data[0])
        num_cells = len(indices)
        Freq = np.zeros(num_cells)
        delta = (ub-lb)/m_j
        
        for i in range(len(data)):
            ind = 0
            for j in range(dim_x):
                index_j = int(np.floor((data[i][j]-lb)/delta))
                ind += m_j**(dim_x - 1 - j) * index_j
            Freq[ind] += 1
        return(Freq/len(data))

    def solve_rc(omega,a,b):
        d = len(a[0])
        x = cp.Variable(d, nonneg = True)
        z = cp.Variable(d)
        w = cp.Variable(d)
        constraints = [cp.norm(z,1)+omega*cp.norm(w,2) + a[0] @ x <= b]
        for i in range(d):
            constraints.append(z[i] + w[i] == -a[i+1] @ x) 
        constraints.append(x<=1)
        obj = cp.Minimize(- cp.sum(x))
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.GUROBI)
        return x.value, prob.value

    def lower_bound_yan2013_chi2(alpha,p,S,N,phi_dot=2):
        N_v = len(p)
        q = cp.Variable(N_v, nonneg = True)
        t = cp.Variable(N_v, nonneg = True)
        r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
        constraints = [cp.sum(q) == 1]
        f_obj = 0
        for i in range(N_v):
            if S[i] == 1:
                f_obj = f_obj + q[i]
            z = cp.vstack([2*(q[i]-p[i]),(t[i]-q[i])])
            constraints.append(cp.norm(z,2) <= (t[i]+q[i]))
        constraints.append(cp.sum(t) <= r)
        obj = cp.Minimize(f_obj)
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.GUROBI)
        return prob.value
    
    def lower_bound_yan2013_mod_chi2_LD(alpha,p,S,N,phi_dot=2):
        # see notation for problem (LD) in yanikoglu2013
        eta = cp.Variable(1, nonneg = True)
        lbda = cp.Variable(1)
        
        # store extra info
        N_v = len(p)
        rho = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
        sum_pi = sum(p[i] for i in range(N_v) if S[i] == 1)
    
        # supplementary variables to get in DCP form
        t1 = cp.Variable(1)
        w1 = cp.Variable(1, nonneg = True)
        t2 = cp.Variable(1)
        w2 = cp.Variable(1, nonneg = True)
        
        f_obj = -eta*rho - lbda - sum_pi*t1 - (1-sum_pi)*t2
        obj = cp.Maximize(f_obj)
        
        constraints = [cp.norm(cp.vstack([w1,t1/2]) , 2) <= (t1+2*eta)/2,
                       -(lbda+1)/2+eta <= w1,
                       cp.norm(cp.vstack([w2,t2/2]) , 2) <= (t2+2*eta)/2,
                       -(lbda)/2+eta <= w2]
    
        prob = cp.Problem(obj, constraints)
        prob.solve(solver = cp.GUROBI)
        return prob.value
        
    def cpt_feas(cpt_arr,x,a,b,indices):
        d = len(cpt_arr)
        S = np.zeros(len(indices))
        for i in range(len(S)):
            const = a[0]
            for j in range(d):
                const = const + cpt_arr[j][indices[i][j]] * a[j+1] 
            if const.dot(x) <= b:
                S[i] = 1
        return(S)
    
    N = len(data)
    a, b, lb, ub, cpt_arr, indices, p = get_values_yanikoglu2013(dim_x, m_j, data)
    start_runtime = time.time()
    omega = omega_init
    lowerbound = -np.inf
    num_iter = 0
    if store_all_solutions:
        all_solutions = []
    if store_pareto_solutions:
        pareto_solutions = []
    while lowerbound < 1-risk_param_epsilon:
        num_iter += 1
        x, obj = solve_rc(omega,a,b)
        S = cpt_feas(cpt_arr,x,a,b,indices)
        
        lb_yan2013 = lower_bound_yan2013_mod_chi2_LD(conf_param_alpha,p,S,N)
        lowerbound = lb_yan2013

        if verbose:
            print('omega:', omega)
            print('lowerbound yan2013:',lb_yan2013)
            print('objective:', -obj)
            print()
        omega = omega + step_size
        if store_all_solutions:
            all_solutions.append({'sol': x, 'obj': obj, 'feas': lowerbound})
        if store_pareto_solutions:
            if len(pareto_solutions) == 0:
                pareto_solutions.append((obj, [lowerbound]))
            elif lowerbound > pareto_solutions[-1][1][0]:
                pareto_solutions.append((obj, [lowerbound]))
        
    runtime = time.time() - start_runtime
    if store_all_solutions:
        return runtime, num_iter, x, obj, lowerbound, all_solutions
    elif store_pareto_solutions:
        return runtime, num_iter, x, obj, lowerbound, pareto_solutions
    return runtime, num_iter, x, obj, lowerbound