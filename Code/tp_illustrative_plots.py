"""
In this script we create illustrative plots for ROBIST applied to the 
toy problem of dimension 2
"""

# external imports
import numpy as np
import scipy
import math
import os
import matplotlib.pyplot as plt


# Matplotlib settings:
size_plots = 3.5
plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
plt.rcParams['figure.dpi'] = 1200

plt.rcParams.update({
    'font.size': 10,
    'text.usetex': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# internal imports
from tp import generate_data, solve_SCP, unc_function
from ROBIST import ROBIST

def eval_robustness(data, x, conf_param_alpha, unc_function, numeric_precision=1e-6):
    f_evals = unc_function(x, data)
    N_vio = sum(f_evals>(0+numeric_precision))
    N = len(data)
    p_feas = 1 - N_vio/N
    if p_feas == 0:
        return p_feas, 0
    elif p_feas == 1:
        return p_feas, 1
    
    def _compute_mod_chi2_lowerbound(p_feas, N, conf_param_alpha):
        r = 1/N*scipy.stats.chi2.ppf(1-conf_param_alpha, 1)
        q_feas = max(p_feas - math.sqrt(p_feas*(1-p_feas)*r),0)
        return q_feas

    return p_feas, _compute_mod_chi2_lowerbound(p_feas, N, conf_param_alpha)

def plot_data(data_train, data_test, save_plot, plot_type, show_legend):
    plt.rcParams['figure.figsize'] = [13/9 * size_plots, size_plots]
    
    plt.plot(data_train[:,0], data_train[:,1],ls='', color='tab:blue', marker="o",markersize=4, label = 'train')
    plt.plot(data_test[:,0], data_test[:,1],ls='', color='tab:orange', marker="s",markersize=4, label = 'test')
    
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
        plot_name = 'output/ToyProblem/figures/demo/Illustrate_data_split_N=' + str(len(data_train)+len(data_test))
        strFile = plot_name + '.' + plot_type
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()

def plot_sol(iter_count, data, S_values, x, obj, p, lb, true_prob, save_plot, plot_type, show_legend,
              N, conf_param_alpha, unc_func=None):
    plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    
    if unc_func is not None:
        f_evals = unc_func(x, data)
        vio = f_evals>(0+1e-6)
        plt.plot(data[vio==False,0],data[vio==False,1],ls='', color='tab:blue', marker=".",markersize=6, label = 'feasible scenarios')
        plt.plot(data[vio,0],data[vio,1],ls='', color='tab:red', marker="*",markersize=6, label = 'violated scenarios')
    else:
        plt.plot(data[:,0],data[:,1],ls='', color='tab:blue', marker=".",markersize=8, label = 'data')
    
    if type(S_values) is list:
        S_values = np.array([*S_values])
        plt.plot(S_values[:,0],S_values[:,1], color='black', marker='x', linestyle='',
                  markersize=8, label = 'sampled scenarios')
    else:
        plt.plot(S_values[0],S_values[1], color='black', marker='x', linestyle='',
                  markersize=8, label = 'nominal scenario')
        
    # Add constraint to plot, given solution x
    constraint_x = np.linspace(-1.05, 1.05, 1000)
    constraint_y = (1 - x[0]*constraint_x) / x[1]
    plt.plot(constraint_x, constraint_y, '--g', label = f'${round(x[0],1)}z_1 + {round(x[1],1)}z_2 \leq 1$', alpha=1)
    
    # add shaded region
    plt.fill_between(constraint_x, -1.05, constraint_y, color='gray', alpha=0.25)
    
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
        plot_name = 'output/ToyProblem/figures/demo/Illustrate_wConstraint_iter='+str(iter_count)+'_N=' + str(N) + '_alpha=' + str(conf_param_alpha)
        strFile = plot_name + '.' + plot_type
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()
    
    
def plot_tradeoff_curve(non_dominated_solutions, save_plot, plot_type, show_legend, N, conf_param_alpha, risk_param_epsilon, i_max):
    plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    
   # first we convert the list of tuples to a numpy array to get data in proper format
    array = np.array([*non_dominated_solutions])
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
        plot_name = 'output/ToyProblem/figures/demo/TradeOffCurve_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon=" + str(risk_param_epsilon) + "_iMax="+str(i_max) + "_new"
        strFile = plot_name + '.' + plot_type
    
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')
        
def plot_tradeoff_curves(plot_info, save_plot, plot_type, show_legend, N, 
                       conf_param_alpha, risk_param_epsilon, i_max):
    plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    
    # first we convert the list of tuples to a numpy array to get data in proper format
    i = 0
    li_markers = ['-o', '-s', 'X']
    for name_method, solutions_method in plot_info.items():
        array = np.array([*solutions_method], dtype=object)
        sorted_array = array[np.argsort(-array[:, 0])]
        y = - sorted_array[:,0] # contains obj
        x = sorted_array[:,1] # contains lb
        for i_x, elem_x in enumerate(x):
            x[i_x] = elem_x[0]
         
        plt.plot(x, y, li_markers[i], label=name_method)
        i = i + 1
     
    plt.axvline(1-risk_param_epsilon, color='grey', ls = '--', lw=1.0)
     
    # plt.xlabel("lower bound on probabilty that solution is feasible")
    plt.xlabel("feasibility certificate")
    plt.ylabel("objective value");
     
    plt.xticks(np.arange(0.75, 1.01, 0.05))
    plt.yticks(np.arange(1.2, 2.01, 0.2))
     
    if show_legend:
        # plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
        plt.legend(loc='upper right')
     
    plt.tight_layout()
     
    if save_plot:
        plot_name = 'output/ToyProblem/figures/demo/TradeOffCurves_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon=" + str(risk_param_epsilon) + "_iMax="+str(i_max) + "_new"
        strFile = plot_name + '.' + plot_type
         
        if os.path.isfile(strFile):
            os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')

# plot settings:
save_plot = True
# plot_type = 'pdf'
plot_type = 'png'
show_legend = True

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

data_train = generate_data(random_seed, N, dim_x=dim_x)

random_seed_2 = 1
data_test = generate_data(random_seed_2, N, dim_x=dim_x)

# plot train & test data sets
plot_data(data_train, data_test, save_plot, plot_type, show_legend)

nominal_scenario = np.array([0,0])
S_values = [nominal_scenario]

x_0, obj_scp = solve_SCP(nominal_scenario, **problem_instance)
obj = - obj_scp
p, lb = eval_robustness(data_train, x_0, conf_param_alpha, unc_function)

true_prob = None

plot_sol(0, data_train, nominal_scenario, x_0, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, unc_func=unc_function)

added_scenario = data_train[10]
S_values.append(added_scenario)

x_1, obj_scp = solve_SCP(S_values, **problem_instance)
obj = - obj_scp
p, lb = eval_robustness(data_train, x_1, conf_param_alpha, unc_function)

plot_sol(1, data_train, S_values, x_1, obj, p, lb, true_prob, save_plot, plot_type, show_legend, N, conf_param_alpha, unc_func=unc_function)

i_max = 1000
stop_criteria={'max_num_iterations': i_max} 
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

robist = ROBIST(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                data_train, data_test, conf_param_alpha=conf_param_alpha,
                verbose=False)

(best_sol, 
  runtime_robist, 
  num_iter, 
  non_dominated_solutions, 
  S_history, 
  all_solutions_robist) = robist.run(stop_criteria=stop_criteria, store_all_solutions=True)

# if only interested in test certificates
# save_plot = True
# plot_type = 'pdf'
# show_legend = False
# plot_tradeoff_curve(non_dominated_solutions, save_plot, plot_type, show_legend, 
#                   N, conf_param_alpha, risk_param_epsilon, i_max)

# re-compute proxy certificates for training data in order to add to plot
non_dominated_solutions_train = []
for sol_info in all_solutions_robist:
    is_non_dominated_yn = False
    for sol in non_dominated_solutions:
        if sol_info['obj'] == sol[0] and sol_info['feas'][0] == sol[1][0]:
            is_non_dominated_yn = True
            break
    if is_non_dominated_yn:
        train_certificate = eval_robustness(data_train, sol_info['sol'], conf_param_alpha, unc_function)[1]
        non_dominated_solutions_train.append((sol_info['obj'], [train_certificate]))

plot_info = {'train': non_dominated_solutions_train,
              'test': non_dominated_solutions}

save_plot = True
show_legend = True
plot_tradeoff_curves(plot_info, save_plot, plot_type, show_legend, 
                    N, conf_param_alpha, risk_param_epsilon, i_max)



