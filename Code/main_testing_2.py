# import external packages
import numpy as np
import cvxpy as cp
import mosek
from sklearn.model_selection import train_test_split
import time

# import internal packages
import phi_divergence as phi
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

# problem specific functions:
def generate_unc_param_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    m = 5
    n = 10
    
    # generate demand vector param
    d_nom = (25, 38, 18, 39, 60, 35, 41, 22, 74, 30)
    d = np.random.default_rng().dirichlet(d_nom, N) * 382
    
    # generate production efficiency param
    p_nom = np.array([[5.0, 7.6, 3.6, 7.8, 12.0, 7.0, 8.2, 4.4, 14.8, 6.0],
                      [3.8, 5.8, 2.8, 6.0, 9.2, 5.4, 6.3, 3.4, 11.4, 4.6],
                      [2.3, 3.5, 1.6, 3.5, 5.5, 3.2, 3.7, 2.0, 6.7, 2.7],
                      [2.6, 4.0, 1.9, 4.1, 6.3, 3.7, 4.3, 2.3, 7.8, 3.2],
                      [2.4, 3.6, 1.7, 3.7, 5.7, 3.3, 3.9, 2.1, 7.0, 2.9]])
    p = np.random.random_sample(size = (N,m,n)) * (p_nom*1.05 - p_nom*0.95) + (p_nom*0.95)
    data = list(zip(d,p))
    return data

def get_fixed_param_data():
    # fixed parameter values from Care (2014)
    C = np.array([[1.8, 2.2, 1.5, 2.2, 2.6, 2.1, 2.2, 1.7, 2.8, 1.9],
                  [1.6, 1.9, 1.3, 1.9, 2.3, 1.9, 2.0, 1.5, 2.5, 1.7],
                  [1.2, 1.5, 1.0, 1.5, 1.9, 1.4, 1.6, 1.1, 2.0, 1.3],
                  [1.3, 1.6, 1.1, 1.6, 2.0, 1.5, 1.7, 1.2, 2.2, 1.4],
                  [1.2, 1.5, 1.0, 1.6, 1.9, 1.5, 1.6, 1.1, 2.1, 1.3]])

    A = np.array([10, 13, 22, 19, 21])
    C_tilde = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
    U = np.array([1.5, 1.8, 1.2, 1.9, 2.2, 1.8, 1.9, 1.4, 2.4, 1.6])
    param_dict = {'C':C, 'A':A, 'C_tilde': C_tilde, 'U': U}
    return param_dict

def solve_P_SCP(S, **kwargs):
    # get fixed parameter values
    C = kwargs['C']
    A = kwargs['A']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    
    # unzip uncertain parameters
    d,p = zip(*S)
    
    # get dimensions of problem
    m,n = p[0].shape
    num_scen = len(d)
    
    # create variables
    theta = cp.Variable(1)
    y = cp.Variable((m, n), nonneg = True)
    
    # set up problem
    constraints = []
    for s in range(num_scen):
        unc_inv_cost_s = sum(C_tilde[k] * cp.maximum(sum(p[s][j][k]*y[j][k] for j in range(m)) - d[s][k], 0)
                             for k in range(n))
        unc_rev_s = sum(U[k] * cp.minimum(sum(p[s][j][k]*y[j][k] for j in range(m)),d[s][k]) 
                        for k in range(n))
        constraints.append(unc_inv_cost_s - unc_rev_s - theta <= 0)
    
    constraints.append(cp.sum(y, axis=1) <= A)
    
    fixed_costs = cp.sum(cp.multiply(C, y))
    obj = cp.Minimize(fixed_costs + theta)
    prob = cp.Problem(obj,constraints)
    
    # solve problem
    time_limit = kwargs.get('time_limit', 2*60*60)    
    prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    x_value = [theta.value, y.value] # Combine y and theta into 1 single solution vector
    return (x_value, prob.value)

def unc_obj_func(x, data, **kwargs):
    # extract values
    C = kwargs['C']
    C_tilde = kwargs['C_tilde']
    U = kwargs['U']
    d,p = zip(*data)
    m,n = p[0].shape
    y = x[1]
    
    # compute obj function value:
    fixed_costs = np.sum(np.multiply(C, y))
    inv_cost = np.array([sum(C_tilde[k] * np.maximum(sum(p[s][j][k]*y[j][k] for j in range(m)) - d[s][k], 0)
                             for k in range(n)) for s in range(len(data))]) 
    rev = np.array([sum(U[k] * np.minimum(sum(p[s][j][k]*y[j][k] for j in range(m)),d[s][k]) 
                        for k in range(n)) for s in range(len(data))]) 
    return fixed_costs + inv_cost - rev

def eval_p_vio(x, obj, data, **kwargs):
    evals = unc_obj_func(x, data, **kwargs) - obj 
    p_vio = sum(evals>(0+(1e-6))) / len(data) 
    return p_vio


output_file_name = 'var_epsilon_original_wdp_Care2014_seeds=1-10'

headers = ['$\epsilon$', 'seed', 
           '$N^{Classic}$', '$T^{Classic}$', '$Obj.~(Classic)$', '$p_{vio}^{OoS}~(Classic)$', 
           '$N_1^{FAST}$', '$N_2^{FAST}$', '$T^{FAST}$', '$Obj.~(FAST)$', '$p_{vio}^{OoS}~(FAST)$', 
           '$N_1^{ISSuR}$', '$N_2^{ISSuR}$', '$T^{ISSuR}$', '$Obj.~(ISSuR)$', '$p_{vio}^{OoS}~(ISSuR)$', 
           '\#Iter.~(\texttt{add})', '\#Iter.~(\texttt{remove})', '$\mu_{|\mathcal{S}_i|}$']

# # Write headers to .txt file
# with open(r'output/WeightedDistributionProblem/headers_'+output_file_name+'.txt','w+') as f:
#     f.write(str(headers))

output_data = {}

epsilon_settings = [0.2, 0.1] #[0.01, 0.05, 0.10]
random_seed_settings = [i for i in range(1, 3)]

# fixed info:
solve_SCP = solve_P_SCP
problem_instance = get_fixed_param_data()
eval_unc_obj = {'function': unc_obj_func,
                    'info': {'risk_measure': 'probability'}}
eval_unc_constr = None
conf_param_alpha = 0.1#1e-9
dim_x = 5 * 10
N_1 = 1000

random_seed = 1234
N_OoS = int(1e6)
data_OoS = generate_unc_param_data(1234, N_OoS)

run_count = 0
for epsilon in epsilon_settings:
    
    eval_unc_obj['info']['desired_rhs'] = 1 - epsilon

    for random_seed in random_seed_settings:

        # # classic approach:
        # N_classic = util.determine_calafiore_N_min(dim_x, 1 - epsilon, conf_param_alpha)
        # data = generate_unc_param_data(random_seed, N_classic)
        # start_time = time.time()
        # x, obj = solve_P_SCP(data, **problem_instance)
        # runtime_classic = time.time() - start_time
        # obj_classic = - obj
        # p_vio_classic = eval_p_vio(x, obj, data_OoS, **problem_instance)
        
        # # FAST approach
        # x, obj, N_2, runtime = util.solve_with_care2014(solve_SCP, problem_instance, generate_unc_param_data, 
        #                                             eval_unc_obj, conf_param_alpha, dim_x, N_1=N_1,
        #                                             random_seed=random_seed)
        # runtime_FAST = runtime 
        # obj_FAST = - obj
        # p_vio_FAST = eval_p_vio(x, obj, data_OoS, **problem_instance)
        
        # Our method
        N_classic = util.determine_calafiore_N_min(dim_x, 1 - epsilon, conf_param_alpha)
        N_total = N_classic
        data = generate_unc_param_data(random_seed, N_total)
        N_train = N_total / 2
        data_train, data_test = train_test_split(data, train_size=(N_train/N_total), random_state=random_seed)
        
        alg = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                    data_train, data_test, conf_param_alpha=conf_param_alpha,
                                    verbose=False)
        
        stop_criteria={'max_elapsed_time': 1*60} # in seconds (time provided to search algorithm)

        (best_sol, runtime, num_iter, pareto_frontier, S_history) = alg.run(stop_criteria=stop_criteria)
        
        obj_ISSuR = - best_sol['obj']
        p_vio_ISSuR = eval_p_vio(best_sol['sol'], best_sol['obj'], data_OoS, **problem_instance)
        S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
        
        # output_data[(epsilon, random_seed)] = [N_classic, runtime_classic, obj_classic, p_vio_classic,
        #                                        N_1, N_2, runtime_FAST, obj_FAST, p_vio_FAST,
        #                                        N_train, (N_total-N_train), runtime, obj_ISSuR, p_vio_ISSuR,
        #                                        num_iter['add'], num_iter['remove'], S_avg]

        # output_file_name = 'new_output_data'
        # with open(r'output/WeightedDistributionProblem/'+output_file_name+'.txt','w+') as f:
        #     f.write(str(output_data))

        # run_count += 1
        # print("Completed run: " + str(run_count))


















