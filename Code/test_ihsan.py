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

def solve_P_SCP(S, **kwargs):
    dim_x = kwargs.get('dim_x', 2)
    x = cp.Variable(dim_x, nonneg = True)
    setup_time_start = time.time()
    constraints = [cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1]-1, x<=10]
    for s in range(len(S)):
        constraints.append(cp.multiply(S[s], x) - 1 <= 0)
    obj = cp.Minimize(- cp.sum(x)) # formulate as a minimization problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 2*60*60) - (time.time() - setup_time_start)
    if time_limit < 0:
        print("Error: did not provide sufficient time for setting up & solving problem")
        return (None, None)
    try:
#         prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
        prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def unc_func(x, data, **kwargs):
    return (np.dot(data,x)) - 1
    
def get_true_prob(x, dim_x):
    return(1/2+1/(2*x[dim_x-1]))
    
def solve_toyproblem_true_prob(desired_rhs, dim_x):
    beta = desired_rhs
    x = cp.Variable(dim_x, nonneg = True)
    constraints = [(1-2*beta)*x[dim_x-1] + 1 >= 0, cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
#     prob.solve(solver=cp.MOSEK)
    prob.solve(solver=cp.GUROBI)
    return(x.value, prob.value)


from phi_divergence import mod_chi2_cut
import scipy.stats
import itertools

def solve_with_ihsan2013(dim_x,risk_param_epsilon,conf_param_alpha,data,m_j=10,
                         omega_init=0.1,step_size=0.01,use_robist_lb=False,
                         store_all_solutions=False):
    def make_center(lb,ub,m_j):
        delta = (ub-lb)/m_j
        center = np.arange(lb+delta/2,ub,delta)
        return(center)          

    def get_freq(m_j,data,lb,ub):
        Freq = np.zeros(m_j)
        delta = (ub-lb)/m_j
        for i in range(len(data)):
            index = int(np.floor((data[i]-lb)/delta))
            Freq[index] = Freq[index] + 1
        return(Freq/len(data))
    
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
        return(Freq/len(data))

    def solve_rc(omega,a,b):
        d = len(a[0])
        x = cp.Variable(d, nonneg = True)
        z = cp.Variable(d)
        w = cp.Variable(d)
        constraints = [cp.norm(z,1)+omega*cp.norm(w,2) + a[0] @ x <= b]
        for i in range(d):
            constraints.append(z[i] + w[i] == -a[i+1] @ x) 
        
        # add our additional constraints
        constraints.append(cp.sum(x[0:(d-1)]) <= x[d-1]-1)
        constraints.append(x<=10)
        
        obj = cp.Maximize(cp.sum(x))
        prob = cp.Problem(obj,constraints)
        # prob.solve(solver = cp.MOSEK)
        prob.solve(solver = cp.GUROBI)
        return(x.value)

    def lower_bound(alpha,p,S,N,phi_dot=2):
        start_time = time.time()
        N_v = len(p)
        q = cp.Variable(N_v, nonneg = True)
        t = cp.Variable(N_v, nonneg = True)
        r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
        constraints = [cp.sum(q) == 1]
        f_obj = sum(q[i] for i in range(N_v) if S[i] == 1)
        # constraints.append(cp.norm(cp.vstack([2*(q-p),(t-q)]), 2) <= t+q)
        # f_obj = 0
        for i in range(N_v):
        #     if S[i] == 1:
        #         f_obj = f_obj + q[i]
            z = cp.vstack([2*(q[i]-p[i]),(t[i]-q[i])])
            constraints.append(cp.norm(z,2) <= (t[i]+q[i]))
        constraints.append(cp.sum(t) <= r)
        obj = cp.Minimize(f_obj)
        prob = cp.Problem(obj,constraints)
        # print("Construction time:", round(time.time() - start_time))
        start_time = time.time()
        # prob.solve(solver = cp.MOSEK)
        prob.solve(solver = cp.GUROBI)
        # print("Solve time:", round(time.time() - start_time))
        return(prob.value)
        
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
    
    def lower_bound_ROBIST(data, x, conf_param_alpha, phi_div=mod_chi2_cut, phi_dot=2, numeric_precision=1e-6):
        N = len(data)
        constr_evals = (np.dot(data,x)) - 1
        N_vio = sum(constr_evals>(0+numeric_precision))
        p_vio = N_vio/N
        if p_vio == 0:
            return 1
        elif p_vio == 1:
            return 0
        return util.compute_mod_chi2_lowerbound(1-p_vio, N, conf_param_alpha)
    
    def get_true_prob(x, dim_x):
        return(1/2+1/(2*x[dim_x-1]))
    
    # see notation from ihsan2013
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
    N = len(data)
    cpt_arr = []
    
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
    
    for i in range(dim_x):
        cpt_arr.append(make_center(lb,ub,m_j))
        
    indices = np.asarray(list((itertools.product(np.arange(m_j), repeat = dim_x))))
    p = get_freq_v2(data,lb,ub,m_j,indices)
    
    start_runtime = time.time()
    omega = omega_init
    lowerbound = -np.inf
    num_iter = 0
    if store_all_solutions:
        all_solutions = []
    while lowerbound < 1-risk_param_epsilon:
        num_iter += 1
        x = solve_rc(omega,a,b)
        if use_robist_lb:
            lb_robist = lower_bound_ROBIST(data,x,conf_param_alpha)
            lowerbound = lb_robist
        else:
            S = cpt_feas(cpt_arr,x,a,b,indices)
            lb_ihsan2013 = lower_bound(conf_param_alpha,p,S,N)
            lowerbound = lb_ihsan2013
        
        obj = np.sum(x)
        omega = omega + step_size
        if store_all_solutions:
            all_solutions.append({'sol': x, 'obj': obj, 'feas': lowerbound})
            
    runtime = time.time() - start_runtime
    if store_all_solutions:
        return runtime, num_iter, x, obj, lowerbound, all_solutions
    
    return runtime, num_iter, x, obj, lowerbound

output_file_name = 'tp_ihsan2013_k=3_mj=10_eps=0.05_alpha=0.01_seeds=1-10'

headers = ['seed', '$k$', '$m_j$', '$m$', '$N_{min}$', 
           'N', '$N_1$', '$N_2$', '$T$ (ishan2013)', '$T$ (ROBIST)', 
           'obj. (ishan2013)', 'obj. (ROBIST)', 'opt. obj. (true)',
           '$\gamma$ (ishan2013)', '$\gamma$ (ROBIST)', 
           'true prob. (ishan2013)', 'true prob. (ROBIST)',
           'prob. MAPE (ishan2013)', 'prob. MAPE (ROBIST)',
           '\#Iter.~(ishan2013)',
           '\#Iter.~(\\texttt{add})', '\#Iter.~(\\texttt{remove})', 
           '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']

# # Write headers to .txt file
# with open(r'output/ToyProblem/headers_'+output_file_name+'.txt','w+') as f:
#     f.write(str(headers))

output_data = {}

# set parameter values
risk_param_epsilon = 0.05
conf_param_alpha = 0.01
dim_x = 3
m_j = 5
m = m_j**dim_x
N_min = 5*m
N = 2*N_min

N_train = math.floor(N/2)
N_test = N - N_train

opt_x_true, opt_obj_true = solve_toyproblem_true_prob(1-risk_param_epsilon, dim_x)

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60*60 

# ROBIST settings:
stop_criteria={'max_elapsed_time': 1*60} 
solve_SCP = solve_P_SCP
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                   'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

random_seed_settings = [i for i in range(1,11)] #11
run_count = 0

avg_lb_gap_ihsan2013 = []
avg_lb_gap_robist = []
avg_lb_over_freq_ihsan2013 = []
avg_lb_over_freq_robist = []

for random_seed in random_seed_settings:
    
    data = generate_data(random_seed, N, dim_x=dim_x)               
    data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)

    # # ihsan2013:
    # runtime_ihsan2013, num_iter_ihsan2013, x, obj_ihsan2013, lb_ihsan2013, all_solutions = solve_with_ihsan2013(dim_x,risk_param_epsilon,conf_param_alpha,data,m_j=m_j,
    #                                                                                                             omega_init=0.0,step_size=0.01,use_robist_lb=False)
    # true_prob_ihsan2013 = get_true_prob(x, dim_x)

    # lb_gap_ihsan2013 = []
    # lb_gap_robist = []
    # lb_over_freq_ihsan2013 = 0
    # lb_over_freq_robist = 0
    # for sol_info in all_solutions:
    #     lb_ihsan2013 = sol_info['lb_ihsan2013']
    #     lb_robist = sol_info['lb_robist']
    #     true_prob = get_true_prob(sol_info['sol'], dim_x)
        
    #     lb_gap_ihsan2013.append(100 * abs(true_prob - lb_ihsan2013)/true_prob)
    #     lb_gap_robist.append(100 * abs(true_prob - lb_robist)/true_prob)
        
    #     if lb_ihsan2013 > true_prob:
    #         lb_over_freq_ihsan2013 += 1
    #     if lb_robist > true_prob:
    #         lb_over_freq_robist += 1
        
    # avg_lb_gap_ihsan2013.append(sum(lb_gap_ihsan2013) / len(lb_gap_ihsan2013))
    # avg_lb_gap_robist.append(sum(lb_gap_robist) / len(lb_gap_robist))
    # avg_lb_over_freq_ihsan2013.append(lb_over_freq_ihsan2013 / len(all_solutions) * 100)
    # avg_lb_over_freq_robist.append(lb_over_freq_robist / len(all_solutions)* 100)

    # ROBIST:
    alg = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                data_train, data_test, conf_param_alpha=conf_param_alpha,
                                verbose=False)
    
    (best_sol, runtime_alg, num_iter, pareto_frontier, S_history, all_solutions) = alg.run(stop_criteria=stop_criteria, store_all_solutions=True)
    
    lb_alg = best_sol['feas'][0]
    obj_alg = - best_sol['obj']
    true_prob_alg = get_true_prob(best_sol['sol'], dim_x)
    S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
    S_max = max(len(S_i) for S_i in S_history)
    num_iter_add = num_iter['add']
    num_iter_remove = num_iter['remove']
    
    
    lb_gap_robist = []
    lb_over_freq_robist = 0
    for sol_info in all_solutions:
        lb_robist = sol_info['feas'][0]
        true_prob = get_true_prob(sol_info['sol'], dim_x)
        lb_gap_robist.append(100 * abs(true_prob - lb_robist)/true_prob)
    
        if lb_robist > true_prob:
            lb_over_freq_robist += 1
        
    avg_lb_gap_robist.append(sum(lb_gap_robist) / len(lb_gap_robist))
    avg_lb_over_freq_robist.append(lb_over_freq_robist / len(all_solutions)* 100)

    # # # turn off:   
    # # (best_sol, runtime_alg, num_iter, pareto_frontier, S_history) = (np.nan, np.nan, np.nan, np.nan, np.nan)
    # # lb_alg = np.nan
    # # obj_alg = np.nan
    # # true_prob_alg = np.nan
    # # S_avg = np.nan
    # # S_max = np.nan
    # # num_iter_add = np.nan
    # # num_iter_remove = np.nan

    # output_data[(random_seed, dim_x, m_j)] = [m_j, m, N_min, N, N_train, N_test, 
    #                                           runtime_ihsan2013, runtime_alg, 
    #                                           obj_ihsan2013, obj_alg, opt_obj_true,
    #                                           lb_ihsan2013, lb_alg,
    #                                           true_prob_ihsan2013, true_prob_alg,
    #                                           avg_lb_gap_ihsan2013, avg_lb_gap_robist, 
    #                                           num_iter_ihsan2013,
    #                                           num_iter_add, num_iter_remove,
    #                                           S_avg, S_max]
    
    # output_file_name = 'new_output_data'
    # with open(r'output/ToyProblem/'+output_file_name+'.txt','w+') as f:
    #     f.write(str(output_data))
    
    run_count += 1
    # print()
    print("completed run: " + str(run_count))
    # print()

print()
# print("MAPE over Ihsan2013:", sum(avg_lb_gap_ihsan2013) / len(avg_lb_gap_ihsan2013))
print("MAPE over ROBIST:", sum(avg_lb_gap_robist) / len(avg_lb_gap_robist))
# print("Freq over Ihsan2013:", sum(avg_lb_over_freq_ihsan2013) / len(avg_lb_over_freq_ihsan2013))
print("Freq over ROBIST:", sum(avg_lb_over_freq_robist) / len(avg_lb_over_freq_robist))
















