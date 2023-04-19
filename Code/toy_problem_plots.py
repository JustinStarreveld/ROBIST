# Import packages
import numpy as np
import cvxpy as cp
import time

# import internal packages
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

# Problem specific functions:
def generate_data(random_seed, N, **kwargs):
    np.random.seed(random_seed)
    dim_x = kwargs.get('dim_x',2)
    data = np.random.uniform(-1,1,size = (N,dim_x)) # generates N random scenarios    
    return data 

def solve_P_SCP(S, **kwargs):
    dim_x = kwargs.get('dim_x', 2)
    x = cp.Variable(dim_x, nonneg = True)
    setup_time_start = time.time()
    constraints = []
    for s in range(len(S)):
        constraints.append(cp.sum(cp.multiply(S[s], x)) - 1 <= 0)
    constraints.append(x<=1)
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
    
    duals = np.zeros(len(S))
    for s in range(len(S)):
        duals[s] = constraints[s].dual_value
    
    return x.value, prob.value, duals

def unc_func(x, data, **kwargs):
    return (np.dot(data,x)) - 1

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


import os
import matplotlib.pyplot as plt

# Matplotlib settings:
size_plots = 3.5
plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
# plt.rcParams['figure.figsize'] = [1.2*size_plots, size_plots]
plt.rcParams['figure.dpi'] = 1200 # can be increased for better quality

plt.rcParams.update({
    'font.size': 10,
    'text.usetex': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def plot_sol(iter_count, data, S_values, x, obj, p, lb, true_prob, save_plot, plot_type, show_legend,
              N, conf_param_alpha, unc_func=None):
    
    if unc_func is not None:
        f_evals = unc_func(x, data)
        vio = f_evals>(0+1e-6)
        plt.plot(data[vio==False,0],data[vio==False,1],ls='', color='tab:blue', marker=".",markersize=6, label = 'feasible scenarios')
        plt.plot(data[vio,0],data[vio,1],ls='', color='tab:red', marker="*",markersize=6, label = 'violated scenarios')
    else:
        plt.plot(data[:,0],data[:,1],ls='', color='tab:blue', marker=".",markersize=8, label = 'data')
    
    # plt.plot(S_values[0],S_values[1], color='black', marker='x', linestyle='',
    #          markersize=8, label = 'nominal scenario')
    
    if S_values is not None:
        plt.plot(S_values[:,0],S_values[:,1], color='black', marker='x', linestyle='',
                  markersize=8, label = 'sampled scenarios')
        
    # Add constraint to plot, given solution x
    constraint_x = np.linspace(-1.05, 1.05, 1000)
    constraint_y = (1 - x[0]*constraint_x) / x[1]
    # plt.plot(constraint_x, constraint_y, '--g', label = r'$z_1 \bar{x}_1 + z_2 \bar{x}_2 \leq 1$', alpha=1)
    # plt.plot(constraint_x, constraint_y, '--g', label = r'$z_1 x_1 + z_2 x_2 \leq 1$', alpha=1)
    plt.plot(constraint_x, constraint_y, '--g', label = f'${round(x[0],1)}z_1 + {round(x[1],1)}z_2 \leq 1$', alpha=1)
    
    # add shaded region
    # constraint_y_lower = np.linspace(-1.05, 1.05, 1000)
    plt.fill_between(constraint_x, -1.05, constraint_y, color='gray', alpha=0.25)

    # to make gray
    # plt.gray()

    # plt.title(r'Iteration '+str(num_iter)+r': $\mathbf{\bar{x}}_{' + str(num_iter) +'}$ = (' + str(round(x[0],2)) + ', ' 
    #            + str(round(x[1],2)) + r') $\Rightarrow$ ' + str(round(obj,2)) 
    #            + r', $\mathrm{\mathbb{P}^{*}}$(feasible) = ' + str(round(true_prob,2)), loc='left')
    
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.xticks(np.arange(-1.0, 1.05, 0.5))
    plt.yticks(np.arange(-1.0, 1.05, 0.5))
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'output/ToyProblem/figures/Illustrate_wConstraint_iter='+str(iter_count)+'_N=' + str(N) + '_alpha=' + str(conf_param_alpha)
        # plot_name = 'output/ToyProblem/figures/Illustrate_iter='+str(iter_count)+'_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon="+ str(risk_param_epsilon)
        # plot_name = 'output/ToyProblem/figures/Illustrate_wConstraint_iter='+str(iter_count)+'_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon="+ str(risk_param_epsilon) +"_nolegend"

        strFile = plot_name + '.' + plot_type
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()
    
    
def plot_pareto_curve(pareto_solutions, save_plot, plot_type, show_legend, N, conf_param_alpha, risk_param_epsilon):
    # first we convert the list of tuples to a numpy array to get data in proper format
    array = np.array([*pareto_solutions])
    sorted_array = array[np.argsort(-array[:, 0])]
    y = - sorted_array[:,0] # contains obj
    x = sorted_array[:,1] # contains lb
    for i, elem_x in enumerate(x):
        x[i] = elem_x[0]
        
    plt.plot(x, y, "-o")
    plt.axvline(1-risk_param_epsilon, ls = '--')
    
    # plt.xlabel("lower bound on probabilty that solution is feasible")
    plt.xlabel("feasibility certificate")
    plt.ylabel("objective value");
    
    plt.xticks(np.arange(0.75, 1, 0.05))
    plt.yticks(np.arange(1.2, 2.01, 0.2))
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'output/ToyProblem/figures/ParetoCurve_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon="+ str(risk_param_epsilon) + "_new"
        strFile = plot_name + '.' + plot_type
    
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')
    
    
# set parameter values
risk_param_epsilon = 0.10
conf_param_alpha = 0.01
dim_x = 2
N = 100
random_seed = 0
numeric_precision=1e-6

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60*60 

data = generate_data(random_seed, N, dim_x=dim_x)

# nominal_scenario = np.array([0,0])
# S_values = np.array([nominal_scenario])

# x_0, obj_scp = solve_P_SCP(nominal_scenario, **problem_instance)
# obj = - obj_scp
# p, lb = lower_bound_robist(data, x_0, conf_param_alpha)

# true_prob = None
# save_plot = True
# plot_type = 'pdf'
# show_legend = True
# # show_legend = False

# # plot_sol(0, data, nominal_scenario, x_0, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, risk_param_epsilon, unc_func=None)
# # plot_sol(0, data, nominal_scenario, x_0, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, unc_func=unc_func)

# # # added_scenario = np.array([0.5563135 , 0.7400243 ])
# # added_scenario = np.array([0.95723668, 0.59831713])
# added_scenario = data[10]
# S_values = np.append(S_values, [added_scenario], axis = 0)

# x_1, obj_scp = solve_P_SCP(S_values, **problem_instance)
# obj = - obj_scp
# p, lb = lower_bound_robist(data, x_1, conf_param_alpha)

# true_prob = None
# save_plot = True
# plot_type = 'pdf'
# show_legend = True
# # show_legend = False

# # plot_sol(1, data, nominal_scenario, x_1, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, risk_param_epsilon, unc_func=None)
# plot_sol(1, data, S_values, x_1, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, unc_func=unc_func)


data_train = data

random_seed_2 = 1
data_test = generate_data(random_seed_2, N, dim_x=dim_x)

stop_criteria={'max_num_iterations': 10} 
solve_SCP = solve_P_SCP
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

robist = iter_gen_and_eval_alg(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                data_train, data_test, conf_param_alpha=conf_param_alpha,
                                verbose=True)

(best_sol, runtime_robist, num_iter, pareto_frontier, S_history, all_solutions_robist) = robist.run(stop_criteria=stop_criteria, store_all_solutions=True)



# save_plot = True
# plot_type = 'pdf'
# show_legend = False
# # show_legend = False
# plot_pareto_curve(pareto_frontier, save_plot, plot_type, show_legend, N, conf_param_alpha, risk_param_epsilon)










