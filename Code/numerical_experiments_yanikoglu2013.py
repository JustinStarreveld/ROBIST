# import external packages
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import time
import math

# import internal packages
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

# problem specific functions:
def generate_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    dim_x = kwargs.get('dim_x',2)
    data = np.random.uniform(-1,1,size = (N,dim_x)) # generates N random scenarios    
    return data 

def generate_data_with_nominal(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    dim_x = kwargs.get('dim_x',2)
    data_nominal = np.array([[0] * dim_x])
    data = np.random.uniform(-1,1,size = (N-1,dim_x)) # generate N-1 scenarios
    data = np.concatenate((data_nominal,data)) # add nominal case to training data
    return data

# def solve_P_SCP(S, **kwargs):
#     dim_x = kwargs.get('dim_x', 2)
#     x = cp.Variable(dim_x, nonneg = True)
#     setup_time_start = time.time()
#     constraints = [cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1]-1, x<=10]
#     for s in range(len(S)):
#         constraints.append(cp.multiply(S[s], x) - 1 <= 0)
#     obj = cp.Minimize(- cp.sum(x)) # formulate as a minimization problem
#     prob = cp.Problem(obj,constraints)
#     time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
#     if time_limit < 0:
#         print("Error: did not provide sufficient time for setting up & solving problem")
#         return (None, None)
#     try:
# #         prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
#         prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
#     except cp.error.SolverError:
#         return (None, None)
#     return x.value, prob.value

def solve_P_SCP(S, **kwargs):
    dim_x = kwargs.get('dim_x', 2)
    x = cp.Variable(dim_x, nonneg = True)
    setup_time_start = time.time()
    constraints = [x<=1]
    for s in range(len(S)):
        constraints.append(cp.sum(cp.multiply(S[s], x)) - 1 <= 0)
    obj = cp.Minimize(- cp.sum(x)) # formulate as a minimization problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)
    try:
        prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    except cp.error.SolverError:
        return (None, None)
    return x.value, prob.value

def unc_func(x, data, **kwargs):
    return (np.dot(data,x)) - 1
    
# def get_true_prob(x, dim_x):
#     return 1/2+1/(2*x[dim_x-1])

def approx_true_prob(x, data, numeric_precision=1e-6):
    f_evals = (np.dot(data,x)) - 1
    N_vio = sum(f_evals>(0+numeric_precision))
    N = len(data)
    p_feas = 1 - N_vio/N
    return p_feas
    
# def solve_toyproblem_true_prob(desired_rhs, dim_x):
#     beta = desired_rhs
#     x = cp.Variable(dim_x, nonneg = True)
#     constraints = [(1-2*beta)*x[dim_x-1] + 1 >= 0, cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1]-1, x<=10]
#     obj = cp.Maximize(cp.sum(x))
#     prob = cp.Problem(obj,constraints)
# #     prob.solve(solver=cp.MOSEK)
#     prob.solve(solver=cp.GUROBI)
#     return x.value, prob.value

def lower_bound_robist(data, x, conf_param_alpha, numeric_precision=1e-6):
    f_evals = (np.dot(data,x)) - 1
    N_vio = sum(f_evals>(0+numeric_precision))
    N = len(data)
    p_feas = 1 - N_vio/N
    if p_feas == 0:
        return p_feas, 0
    elif p_feas == 1:
        return p_feas, 1
    return p_feas, util.compute_mod_chi2_lowerbound(p_feas, N, conf_param_alpha)

# yanikoglu2013-related functions:
import scipy.stats
import itertools

def make_center(lb,ub,m_j):
    delta = (ub-lb)/m_j
    center = np.arange(lb+delta/2,ub,delta)
    return center          

def get_freq_v2(data,lb,ub,m_j,indices):
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
    return Freq/len(data)

def solve_rc(omega,a,b):
    d = len(a[0])
    x = cp.Variable(d, nonneg = True)
    z = cp.Variable(d)
    w = cp.Variable(d)
    constraints = [cp.norm(z,1)+omega*cp.norm(w,2) + a[0] @ x <= b]
    for i in range(d):
        constraints.append(z[i] + w[i] == -a[i+1] @ x) 
    
    # add our additional constraints
    # constraints.append(cp.sum(x[0:(d-1)]) <= x[d-1]-1)
    # constraints.append(x<=10)
    constraints.append(x<=1)
    
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    # prob.solve(solver = cp.MOSEK)
    prob.solve(solver = cp.GUROBI)
    return x.value

def lower_bound_yanikoglu2013_chi2(alpha,p,S,N,phi_dot=2):
    # start_time = time.time()
    N_v = len(p)
    q = cp.Variable(N_v, nonneg = True)
    t = cp.Variable(N_v, nonneg = True)
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
    constraints = [cp.sum(q) == 1]
    f_obj = sum(q[i] for i in range(N_v) if S[i] == 1)
    for i in range(N_v):
        z = cp.vstack([2*(q[i]-p[i]),(t[i]-q[i])])
        constraints.append(cp.norm(z,2) <= (t[i]+q[i]))
    constraints.append(cp.sum(t) <= r)
    obj = cp.Minimize(f_obj)
    prob = cp.Problem(obj,constraints)
    # print("construction time:", round(time.time() - start_time,2))
    # start_time = time.time()
    # prob.solve(solver = cp.MOSEK)
    prob.solve(solver = cp.GUROBI)
    # print("solve time:", round(time.time() - start_time,2))
    return prob.value
    
def lower_bound_yanikoglu2013_chi2_LD(alpha,p,S,N,phi_dot=2):
    start_time = time.time()
    # see notation for problem (LD) in yanikoglu2013
    eta = cp.Variable(1, nonneg = True)
    lbda = cp.Variable(1)
    
    # store extra info
    N_v = len(p)
    rho = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
    sum_pi = sum(p[i] for i in range(N_v) if S[i] == 1)
    
    chi2_conj_1 = cp.Expression(2 - 2*cp.sqrt(1-(-lbda-1)))    
    phi_conj_1 = cp.perspective(chi2_conj_1,eta)
    
    chi2_conj_2 = cp.Expression(2 - 2*cp.sqrt(1-(-lbda)))    
    phi_conj_2 = cp.perspective(chi2_conj_2,eta)
    
    f_obj = -eta*rho - lbda - (phi_conj_1*sum_pi + phi_conj_2*(1-sum_pi))
    obj = cp.Maximize(f_obj)

    constraints = [-lbda-1 <= eta,
                   -lbda <= eta]

    # prob = cp.Problem(obj)
    prob = cp.Problem(obj,constraints)
    
    print("construction time:", round(time.time() - start_time,2))
    start_time = time.time()
    # prob.solve(solver = cp.MOSEK)
    prob.solve(solver = cp.GUROBI)
    print("solve time:", round(time.time() - start_time,2))
    return prob.value

def lower_bound_yanikoglu2013_mod_chi2_LD(alpha,p,S,N,phi_dot=2):
    start_time = time.time()
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
    
    # print("construction time:", round(time.time() - start_time,2))
    # start_time = time.time()
    # prob.solve(solver = cp.MOSEK)
    prob.solve(solver = cp.GUROBI)
    # print("solve time:", round(time.time() - start_time,2))
    return prob.value

def lower_bound_yanikoglu2013_mod_chi2(alpha,p,S,N,phi_dot=2):
    # start_time = time.time()
    N_v = len(p)
    q = cp.Variable(N_v, nonneg = True)
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
    constraints = [cp.sum(q) == 1]
    phi_cons = sum(1/p[i]*(q[i]-p[i])**2 for i in range(N_v) if p[i] > 0)
    constraints.append(phi_cons<=r)
    f_obj = sum(q[i] for i in range(N_v) if S[i] == 1)
    obj = cp.Minimize(f_obj)
    prob = cp.Problem(obj,constraints)
    # print("construction time:", round(time.time() - start_time,2))
    # start_time = time.time()
    # prob.solve(solver = cp.MOSEK)
    prob.solve(solver = cp.GUROBI)
    # print("solve time:", round(time.time() - start_time,2))
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
    return S

def get_values_yanikoglu2013(dim_x, m_j, data):
    # ORIGNAL PROBLEM: see notation from yanikoglu2013
    # a = [np.array([1,1]), np.array([1,0]), np.array([0,1])]
    # b = 10
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
    p = get_freq_v2(data,lb,ub,m_j,indices)
    
    return a, b, lb, ub, cpt_arr, indices, p

def solve_with_yanikoglu2013(dim_x,risk_param_epsilon,conf_param_alpha,data,m_j=10,
                             omega_init=0.1,step_size=0.01,use_robist_lb=False,
                             store_all_solutions=False,
                             verbose=False):
    # OLD CODE: Assumes independence
    # np.random.seed(random_seed) 
    # xi = np.random.uniform(size = (dim_x,N))*2-1
    # N = N**dim_x # assume data is indep and all combinations are taken
    # # to get all possible combinations of independent data:
    # data = np.array(np.meshgrid(*xi)).T.reshape(-1,dim_x)
    # indices = np.asarray(list((itertools.product(np.arange(m_j), repeat = dim_x))))
    # p = np.zeros(len(indices))
    # freq_ct = []
    # for i in range(dim_x):
    #     cpt_arr.append(make_center(lb,ub,m_j))
    #     freq_ct.append(get_freq(m_j, data.T[i],lb,ub))
    # for j in range(len(indices)):
    #     p[j] = 1
    #     for k in range(dim_x):
    #         p[j] = p[j] * freq_ct[k][indices[j][k]]
    
    # lower_bound_yanikoglu2013 = lower_bound_yanikoglu2013_mod_chi2
    lower_bound_yanikoglu2013 = lower_bound_yanikoglu2013_mod_chi2_LD
    
    N = len(data)
    a, b, lb, ub, cpt_arr, indices, p = get_values_yanikoglu2013(dim_x, m_j, data)
    
    start_runtime = time.time()
    omega = omega_init
    lowerbound = -np.inf
    num_iter = 0
    if store_all_solutions:
        all_solutions = []
    while lowerbound < 1-risk_param_epsilon:
        num_iter += 1
        x = solve_rc(omega,a,b)
        obj = np.sum(x)
        if use_robist_lb:
            p_robist, lb_robist = lower_bound_robist(data,x,conf_param_alpha)
            lowerbound = lb_robist
        else:
            S = cpt_feas(cpt_arr,x,a,b,indices)
            lb_yanikoglu2013 = lower_bound_yanikoglu2013(conf_param_alpha,p,S,N)
            lowerbound = lb_yanikoglu2013
            
            if verbose:
                print("iter :", num_iter)
                print("omega:", omega)
                print("lb   :", lowerbound)
                print("obj  :", obj)
                # print("mod_chi2     :", lower_bound_yanikoglu2013_mod_chi2(conf_param_alpha,p,S,N))
                # print("chi2:", lower_bound_yanikoglu2013_chi2(conf_param_alpha,p,S,N))
                # print("chi2_LD:", lower_bound_yanikoglu2013_chi2_LD(conf_param_alpha,p,S,N))
                print()
        
        omega = omega + step_size
        if store_all_solutions:
            all_solutions.append({'sol': x, 'obj': obj, 'feas': lowerbound})
            
    runtime = time.time() - start_runtime
    if store_all_solutions:
        return runtime, num_iter, x, obj, lowerbound, all_solutions
    
    return runtime, num_iter, x, obj, lowerbound

# set parameter values
risk_param_epsilon = 0.05
conf_param_alpha = 0.01
dim_x = 5
m_j = 10
m = m_j**dim_x
N_min = 5*m
N = 2*N_min

N_train = math.floor(N/2)
N_test = N - N_train

# opt_x_true, opt_obj_true = solve_toyproblem_true_prob(1-risk_param_epsilon, dim_x)

N_eval = 1000000
data_eval = generate_data(1234, N_eval, dim_x=dim_x)

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60*60 

# ROBIST settings:
# stop_criteria={'max_elapsed_time': 1*60} 
stop_criteria={'max_num_iterations': 500}
solve_SCP = solve_P_SCP
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                   'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

num_seeds = 10
random_seed_settings = [i for i in range(1,num_seeds+1)] #11
# random_seed_settings = [10]

output_file_name = f'tp_yanikoglu2013_mod_chi2_k={dim_x}_mj={m_j}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_seeds=1-{num_seeds}'

headers = ['seed', '$k$', '$m_j$', '$m$', '$N_{min}$', 
           'N', '$N_1$', '$N_2$', 
           '\#Iter.~(yanikoglu2013)', '\#Iter.~(\\texttt{add})', '\#Iter.~(\\texttt{remove})', 
           '$T$ (yanikoglu2013)', '$T$ (ROBIST)', 
           'obj. (yanikoglu2013)', 'obj. (ROBIST)',
           'probability bound (yanikoglu2013)', 'probability bound (ROBIST)', 
           'true probability (yanikoglu2013)', 'true probability (ROBIST)',
           'probability bound computation time (yanikoglu2013)', 'bound computation time (ROBIST)', 
           'probability bound MAE (yanikoglu2013)', 'probability bound MAE (ROBIST)',
           'probability bound reliability (yanikoglu2013)', 'probability bound reliability (ROBIST)',
           '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']

# Write headers to .txt file
with open(r'output/ToyProblem/results_v2/headers_'+output_file_name+'.txt','w+') as f:
    f.write(str(headers))

output_data = {}

run_count = 0
for random_seed in random_seed_settings:
    
    data = generate_data(random_seed, N, dim_x=dim_x)               
    data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)

    # yanikoglu2013:
    (runtime_yanikoglu2013, num_iter_yanikoglu2013, x, 
     obj_yanikoglu2013, lb_yanikoglu2013, all_solutions_yanikoglu2013) = solve_with_yanikoglu2013(dim_x,risk_param_epsilon,conf_param_alpha,data,m_j=m_j,
                                                                                                  omega_init=0.0,step_size=0.01,use_robist_lb=False, 
                                                                                                  store_all_solutions=True,verbose=False)
    # true_prob_yanikoglu2013 = get_true_prob(x, dim_x)
    true_prob_yanikoglu2013 = approx_true_prob(x, data_eval)
    
    # all_solutions_yanikoglu2013 = []
    # print("finished yanikoglu2013")
    
    # ROBIST:
    robist = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                data_train, data_test, conf_param_alpha=conf_param_alpha,
                                verbose=False)
    
    (best_sol, runtime_robist, num_iter, pareto_frontier, S_history, all_solutions_robist) = robist.run(stop_criteria=stop_criteria, store_all_solutions=True)
    
    lb_robist = best_sol['feas'][0]
    obj_robist = - best_sol['obj']
    # true_prob_robist = get_true_prob(best_sol['sol'], dim_x)
    true_prob_robist = approx_true_prob(best_sol['sol'], data_eval)
    S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
    S_max = max(len(S_i) for S_i in S_history)
    num_iter_add = num_iter['add']
    num_iter_remove = num_iter['remove']
    
    # print("finished robist")
    
    # compare lower bounds on all solutions from both methods
    time_compute_lb_yanikoglu2013 = 0
    time_compute_lb_robist = 0
    lb_AE_yanikoglu2013 = 0
    lb_AE_robist = 0
    lb_reliability_yanikoglu2013 = 0
    lb_reliability_robist = 0
    for sol_info in all_solutions_yanikoglu2013:
        temp_lb_yanikoglu2013 = sol_info['feas']
        start_time_compute_lb = time.time()
        p, temp_lb_robist = lower_bound_robist(data_test,sol_info['sol'],conf_param_alpha)
        time_compute_lb_robist += (time.time() - start_time_compute_lb)
        # true_prob = get_true_prob(sol_info['sol'], dim_x)
        true_prob = approx_true_prob(sol_info['sol'], data_eval)
        lb_AE_yanikoglu2013 += abs(true_prob - temp_lb_yanikoglu2013)
        lb_AE_robist += abs(true_prob - temp_lb_robist)
        if temp_lb_yanikoglu2013 <= true_prob:
            lb_reliability_yanikoglu2013 += 1
        if temp_lb_robist <= true_prob:
            lb_reliability_robist += 1
        
    # to compute lb_yanikoglu2013
    a, b, lb, ub, cpt_arr, indices, p = get_values_yanikoglu2013(dim_x, m_j, data)
    for sol_info in all_solutions_robist:
        start_time_compute_lb = time.time()
        S = cpt_feas(cpt_arr,sol_info['sol'],a,b,indices)
        temp_lb_yanikoglu2013 = lower_bound_yanikoglu2013_mod_chi2_LD(conf_param_alpha,p,S,N)
        time_compute_lb_yanikoglu2013 += time.time() - start_time_compute_lb
        temp_lb_robist = sol_info['feas'][0]        
        # true_prob = get_true_prob(sol_info['sol'], dim_x)
        true_prob = approx_true_prob(sol_info['sol'], data_eval)
        lb_AE_yanikoglu2013 += abs(true_prob - temp_lb_yanikoglu2013)
        lb_AE_robist += abs(true_prob - temp_lb_robist)
        if temp_lb_yanikoglu2013 <= true_prob:
            lb_reliability_yanikoglu2013 += 1
        if temp_lb_robist <= true_prob:
            lb_reliability_robist += 1

    # get averages
    avg_time_compute_lb_yanikoglu2013 = time_compute_lb_yanikoglu2013 / len(all_solutions_robist)
    avg_time_compute_lb_robist = time_compute_lb_robist / len(all_solutions_yanikoglu2013)
    # avg_time_compute_lb_robist = np.nan
    avg_lb_AE_yanikoglu2013 = lb_AE_yanikoglu2013 / (len(all_solutions_yanikoglu2013) + len(all_solutions_robist)) 
    avg_lb_AE_robist = lb_AE_robist / (len(all_solutions_yanikoglu2013) + len(all_solutions_robist)) 
    avg_lb_reliability_yanikoglu2013 = 100*lb_reliability_yanikoglu2013 / (len(all_solutions_yanikoglu2013) + len(all_solutions_robist))
    avg_lb_reliability_robist = 100 * lb_reliability_robist / (len(all_solutions_yanikoglu2013) + len(all_solutions_robist))

    # # To turn off yanikoglu2013 output
    # num_iter_yanikoglu2013 = np.nan
    # runtime_yanikoglu2013 = np.nan
    # obj_yanikoglu2013 = np.nan
    # lb_yanikoglu2013 = np.nan
    # true_prob_yanikoglu2013 = np.nan
    # avg_time_compute_lb_yanikoglu2013 = np.nan
    # avg_lb_AE_yanikoglu2013 = np.nan
    # avg_lb_reliability_yanikoglu2013 = np.nan

    output_data[(random_seed, dim_x, m_j)] = [m_j, m, N_min, N, N_train, N_test,
                                              num_iter_yanikoglu2013,
                                              num_iter_add, num_iter_remove,
                                              runtime_yanikoglu2013, runtime_robist, 
                                              obj_yanikoglu2013, obj_robist,
                                              lb_yanikoglu2013, lb_robist,
                                              true_prob_yanikoglu2013, true_prob_robist,
                                              avg_time_compute_lb_yanikoglu2013, avg_time_compute_lb_robist,
                                              avg_lb_AE_yanikoglu2013, avg_lb_AE_robist,
                                              avg_lb_reliability_yanikoglu2013, avg_lb_reliability_robist,
                                              S_avg, S_max]
        
    # output_file_name = 'new_output_data'
    with open(r'output/ToyProblem/results_v2/'+output_file_name+'.txt','w+') as f:
        f.write(str(output_data))
    
    run_count += 1
    print("completed run: " + str(run_count))
    # print()


import pandas as pd
df_output = pd.DataFrame.from_dict(output_data, orient='index')
print(df_output.mean())















