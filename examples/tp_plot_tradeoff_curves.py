"""
In this script we plot trade-off curves for C&C, Y&dH and ROBIST
when applied to the toy problem of dimension=2
"""

# external imports
from sklearn.model_selection import train_test_split
import math
import time

# internal imports
from tp import generate_data, solve_SCP, unc_function, solve_with_yan2013
from robist import Robist
from scen_opt_methods import determine_cam2008_N_min

def plot_pareto_curves(plot_info, save_plot, plot_type, show_legend, N, 
                       conf_param_alpha, risk_param_epsilon):
    import numpy as np
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
    
    plt.xlabel("feasibility certificate")
    plt.ylabel("objective value");
    
    plt.xticks(np.arange(0.80, 1.01, 0.05))
    plt.yticks(np.arange(1.2, 2.01, 0.2))
    
    if show_legend:
        # plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'output/ToyProblem/figures/ParetoCurves_compare_cal2005_yan2013_robist_N=' + str(N) + '_alpha=' + str(conf_param_alpha) + "_epsilon="+ str(risk_param_epsilon) + "_new"
        strFile = plot_name + '.' + plot_type
    
        if os.path.isfile(strFile):
           os.remove(strFile)
        plt.savefig(strFile, bbox_inches='tight')



dim_x = 2
risk_param_epsilon = 0.05
conf_param_alpha = 0.01
m_j = 10
N = 1000
N_train = math.floor(N/2)
N_test = N - N_train

random_seed = 12345

data = generate_data(random_seed, N, dim_x=dim_x)               
data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60 

# ROBIST settings: 
stop_criteria={'max_num_iterations': 105}
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability',
                            'desired_rhs': 1 - risk_param_epsilon}}]

algorithm = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                    data_train, data_test, conf_param_alpha=conf_param_alpha,
                    use_dual_sol=True, verbose=False)

(best_sol, 
  runtime_robist, 
  num_iter, 
  pareto_robist, 
  S_history) = algorithm.run(stop_criteria=stop_criteria, 
                            store_all_solutions=False,
                            random_seed=random_seed)
                             
print("runtime ROBIST:", round(runtime_robist, 1))
             
# Y&dH method:   
(runtime_yan2013, 
  num_iter_yan2013,
  x_yan2013, 
  obj_yan2013, 
  lb_yan2013,
  pareto_yan2013) = solve_with_yan2013(dim_x, risk_param_epsilon,
                                          conf_param_alpha, data, store_pareto_solutions=True)

print("runtime Y&dH:", round(runtime_yan2013, 1))

# C&C method:
N_cam2008 = determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
data = generate_data(random_seed, N_cam2008, dim_x=dim_x)
problem_instance['get_dual_sol'] = False
start_time = time.time()
x, obj = solve_SCP(data, **problem_instance)
print("runtime C&C:", round(time.time() - start_time, 1))

plot_info = {
    'ROBIST': pareto_robist,
    'Y&dH': pareto_yan2013,
    'C&C' : [(obj, [1-risk_param_epsilon])]
        }
                      
                                         
plot_pareto_curves(plot_info, True, 'pdf', True, N, conf_param_alpha, risk_param_epsilon)
    
    



