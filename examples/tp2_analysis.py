# import external packages
from sklearn.model_selection import train_test_split
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyprojroot import here

# import internal packages
from tp2 import generate_data, solve_SCP, solve_toyproblem_true_prob, unc_func, get_true_prob

# from robist import Robist
# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

output_folder = here("examples/output/ToyProblem2/")
output_folder.mkdir(parents=True, exist_ok=True)

def make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings, 
                 y_axis_output, y_axis_label, name_plot, save_plot, plot_type='pdf', log_scale_yn=False):
    # Matplotlib settings:
    size_plots = 3.5
    # plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    plt.rcParams['figure.figsize'] = [1.2*size_plots, size_plots]
    plt.rcParams['figure.dpi'] = 1200 # can be increased for better quality

    plt.rcParams.update({
        'font.size': 12,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    if log_scale_yn:
        plt.yscale("log")
    
    fig, ax = plt.subplots()
    ax.boxplot(y_axis_output, showfliers=False)
    plt.xticks([i for i in range(1, len(N_settings)+1)], [r'$10^{'+str(int(round(math.log(i)/math.log(10))))+'}$' for i in N_settings])
    
    plt.xlabel(r"$N$")
    plt.ylabel(y_axis_label)
    
    plt.tight_layout()
    
    name = "boxplot_N_"+name_plot
    if save_plot:
        strFile = output_folder / f'{name}_k={dim_x}_alpha={conf_param_alpha}_epsilon={risk_param_epsilon}_num_seeds={num_seeds}_new.{plot_type}'
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()

"""
Analysis of optimality gap as N increases
"""                            
# set parameter values
dim_x = 2
risk_param_epsilon = 0.05
conf_param_alpha = 0.10

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60 

i_max = 1000
stop_criteria={'max_num_iterations': i_max}
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                    'info': {'risk_measure': 'probability',
                            'desired_rhs': 1 - risk_param_epsilon}}]

num_seeds = 100
N_settings = [100, 1000, 10000, 100000, 1000000] 
opt_gap_output = []
reliability_output = []
MAE_output = []

for N in N_settings:
    
    N_train = math.floor(N/2)
    N_test = N - N_train
    
    print("------------------------------------------------------------------------")
    print("N="+str(N))
    print()    

    output_file_name = f'tp_optgap_as_N_increases_k={dim_x}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_imax={i_max}_N={N}_seeds=1-{num_seeds}'

    headers = ['seed', '$k$', 
                '$N_1$', '$N_2$', 
                '\# iter.~(\\texttt{add})', '\# iter.~(\\texttt{remove})', 
                'runtime (ROBIST)', 
                'obj. (true prob)', 'obj. (ROBIST)', 'opt gap (%)'
                '$\gamma$ (ROBIST)', 'true prob (ROBIST)', 
                '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$',
                'reliability', 'MAE']
    
    # Write headers to .txt file
    with open(output_folder / f'headers_{output_file_name}.txt','w+') as f:
        f.write(str(headers))
    
    output_data = {}
    
    random_seed_settings = [i+1 for i in range(num_seeds)]
    run_count = 0
    for random_seed in random_seed_settings:
        
        data = generate_data(random_seed, N, dim_x=dim_x)               
        data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)
        
        # ROBIST:        
        algorithm = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                            data_train, data_test, conf_param_alpha=conf_param_alpha,
                            use_dual_sol=True, verbose=False)
        
        (best_sol, 
          runtime_robist, 
          num_iter, 
          pareto_frontier, 
          S_history,
          all_solutions) = algorithm.run(stop_criteria=stop_criteria, 
                                    store_all_solutions=True,
                                    random_seed=random_seed)
                                               
        lb_robist = best_sol['feas'][0]
        true_prob_robist = get_true_prob(best_sol['sol'], dim_x)
        obj_robist = - best_sol['obj']
        S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
        S_max = max(len(S_i) for S_i in S_history)
        num_iter_add = num_iter['add']
        num_iter_remove = num_iter['remove']
        
        true_prob_obj = solve_toyproblem_true_prob(dim_x, risk_param_epsilon)
        
        true_best_obj_robist = 0
        for sol in all_solutions:
            gamma_i = sol['feas'][0]
            if gamma_i >= 1-risk_param_epsilon: 
                x_i = sol['sol']
                if get_true_prob(x_i, dim_x) >= 1-risk_param_epsilon:
                    obj_i = - sol['obj']
                    if obj_i > true_best_obj_robist:
                        true_best_obj_robist = obj_i
        
        opt_gap = 100 * (true_prob_obj-true_best_obj_robist) / true_prob_obj
        
        num_bounds_reliable = 0
        sum_AE = 0
        for sol in all_solutions:
            x_i = sol['sol']
            gamma_i = sol['feas'][0]
            true_prob_i = get_true_prob(x_i, dim_x)
            
            sum_AE += abs(gamma_i - true_prob_i)
            if gamma_i <= true_prob_i:
                num_bounds_reliable += 1
        
        reliability = 100 * num_bounds_reliable / len(all_solutions)
        MAE = sum_AE / len(all_solutions)
                                           
        output_data[(random_seed, dim_x)] = [N_train, N_test, 
                                            num_iter_add, num_iter_remove,
                                            runtime_robist, 
                                            true_prob_obj, obj_robist, opt_gap,
                                            lb_robist, true_prob_robist,
                                            S_avg, S_max,
                                            reliability, MAE]
        
        # output_file_name = 'new_output_data'
        with open(output_folder / f'results_{output_file_name}.txt','w+') as f:
            f.write(str(output_data))
        
        run_count += 1
        # print("Completed run: " + str(run_count))
    
    df_output = pd.DataFrame.from_dict(output_data, orient='index')
    opt_gap_output.append(df_output[7])
    reliability_output.append(df_output[12])
    MAE_output.append(df_output[13])
    
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings, opt_gap_output, "optimality gap (%)", 'optgap', True, plot_type='pdf')
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings, reliability_output, "certificate reliability (%)", 'reliability', True, plot_type='pdf')
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings, MAE_output, "certificate MAE", 'MAE', True, plot_type='pdf')
    
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings[1:], opt_gap_output[1:], "optimality gap (%)", 'optgap_1000', True, plot_type='pdf')
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings[1:], reliability_output[1:], "certificate reliability (%)", 'reliability_1000', True, plot_type='pdf')
make_boxplot_N(dim_x, conf_param_alpha, risk_param_epsilon, num_seeds, N_settings[1:], MAE_output[1:], "certificate MAE", 'MAE_1000', True, plot_type='pdf')


"""
Analysis of computational efficiency as k increases
"""                            
def make_boxplot_k(dim_x_settings, conf_param_alpha, risk_param_epsilon, num_seeds, 
                   y_axis_output, y_axis_label, name_plot, save_plot, plot_type='pdf', log_scale_yn=False):
    # Matplotlib settings:
    size_plots = 3.5
    # plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    plt.rcParams['figure.figsize'] = [1.2*size_plots, size_plots]
    plt.rcParams['figure.dpi'] = 1200 # can be increased for better quality

    plt.rcParams.update({
        'font.size': 12,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    if log_scale_yn:
        plt.yscale("log")
    
    fig, ax = plt.subplots()
    ax.boxplot(y_axis_output, showfliers=False)
    plt.xticks([i+1 for i in range(len(dim_x_settings))], [r'$'+str(k)+'$' for k in dim_x_settings])
    
    plt.xlabel(r"$k$")
    plt.ylabel(y_axis_label)
    
    plt.tight_layout()
    
    name = "boxplot_k_"+name_plot
    if save_plot:
        strFile = output_folder / f'{name}_alpha={conf_param_alpha}_epsilon={risk_param_epsilon}_num_seeds={num_seeds}_new.{plot_type}'
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()

def make_boxplot_iter(iter_settings, conf_param_alpha, risk_param_epsilon, num_seeds, 
                      y_axis_output, y_axis_label, name_plot, save_plot, plot_type='pdf'):
    # Matplotlib settings:
    size_plots = 3.5
    # plt.rcParams['figure.figsize'] = [16/9 * size_plots, size_plots]
    plt.rcParams['figure.figsize'] = [1.2*size_plots, size_plots]
    plt.rcParams['figure.dpi'] = 1200 # can be increased for better quality

    plt.rcParams.update({
        'font.size': 12,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    fig, ax = plt.subplots()
    ax.boxplot(y_axis_output, showfliers=False)
    plt.xticks([i+1 for i in range(len(iter_settings))], [str(i) for i in iter_settings])
    
    plt.xlabel("No. of iterations")
    plt.ylabel(y_axis_label)
    
    plt.tight_layout()
    
    name = "boxplot_iter_"+name_plot
    if save_plot:
        strFile = output_folder / f'{name}_alpha={conf_param_alpha}_epsilon={risk_param_epsilon}_num_seeds={num_seeds}_new.{plot_type}'
        plt.savefig(strFile, bbox_inches='tight')
    
    plt.show()

# Custom function to evaluate expressions and replace 'array' and 'nan'
def custom_eval(node):
    """
    A custom AST (Abstract Syntax Tree) evaluator to handle 'array' and 'nan' strings.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == 'array':  # Check if it's an 'array' call
            return np.array([custom_eval(arg) for arg in node.args[0].elts])
    elif isinstance(node, ast.Name):
        if node.id == 'nan':  # Replace 'nan' with np.nan
            return np.nan
    elif isinstance(node, ast.List):  # Handle nested lists
        return [custom_eval(elem) for elem in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(custom_eval(elem) for elem in node.elts)
    elif isinstance(node, ast.Dict):
        return {custom_eval(k): custom_eval(v) for k, v in zip(node.keys, node.values)}
    elif isinstance(node, ast.Constant):  # Handle numbers, strings, and constants (new in Python 3.8+)
        return node.value
    return node

def parse_text_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Use ast.parse to safely evaluate the content
    parsed_data = ast.parse(data, mode='eval')

    # Evaluate the AST nodes using the custom evaluator
    return custom_eval(parsed_data.body)

# set parameter values
risk_param_epsilon = 0.05
conf_param_alpha = 0.10

N = int(10**6)
N_train = math.floor(N/2)
N_test = N - N_train

problem_instance = {}
problem_instance['time_limit'] = 5*60 

eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                    'info': {'risk_measure': 'probability',
                            'desired_rhs': 1 - risk_param_epsilon}}]

num_seeds = 100
delta_settings = [0.05, 0.01, 0.005, 0.001]
delta_min = delta_settings[-1]
num_iter_output = {}
S_max_output = {}
for delta in delta_settings:
    num_iter_output[delta] = []
    S_max_output[delta] = []
    
num_iter_feas_output = []
    
iter_settings = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
gap_at_iterations = {}
for i in iter_settings:
    gap_at_iterations[i] = []
    
dim_x_settings = [2, 20, 200, 2000]
for dim_x in dim_x_settings:
    problem_instance['dim_x'] = dim_x
    
    print("------------------------------------------------------------------------")
    print("k="+str(dim_x))
    print()    

    output_file_name = f'tp_computation_as_k_increases_k={dim_x}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_N={N}_seeds=1-{num_seeds}'

    headers = ['seed', '$k$', 
                '$N_1$', '$N_2$',
                'feasible num iter']
    for delta in delta_settings:
        headers.append("delta="+str(delta)+' num iter')
        headers.append("delta="+str(delta)+' $\max_{i}|\mathcal{S}_i|$')
    
    for i in iter_settings:
        headers.append("gap_after_iter="+str(i))
    
    # Write headers to .txt file
    with open(output_folder / f'headers_{output_file_name}.txt','w+') as f:
        f.write(str(headers))
    
    output_data = {}
    
    true_prob_obj = solve_toyproblem_true_prob(dim_x, risk_param_epsilon)
    obj_stop_threshold = - (1-delta_min) *  true_prob_obj
    
    stop_criteria={'obj_stop': obj_stop_threshold, 'max_num_iterations': 1000}
    
    random_seed_settings = [i+1 for i in range(num_seeds)]
    run_count = 0
    for random_seed in random_seed_settings:
        
        data = generate_data(random_seed, N, dim_x=dim_x)               
        data_train, data_test = train_test_split(data, train_size=(N_train/N), random_state=random_seed)
        
        # ROBIST:        
        algorithm = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                            data_train, data_test, conf_param_alpha=conf_param_alpha,
                            use_dual_sol=True, verbose=False)
        
        (best_sol, 
          runtime_robist, 
          num_iter, 
          pareto_frontier, 
          S_history,
          all_solutions) = algorithm.run(stop_criteria=stop_criteria, 
                                          store_all_solutions=True,
                                          random_seed=random_seed)
                                         
        output_data[(random_seed, dim_x)] = [N_train, N_test]
        
        for i, sol in enumerate(all_solutions):
            gamma_i = sol['feas'][0]
            if gamma_i >= 1-risk_param_epsilon: 
                output_data[(random_seed, dim_x)].append(i+1)
                break
        
        for delta in delta_settings:
            obj_threshold_delta = (1-delta) * true_prob_obj
            for i, sol in enumerate(all_solutions):
                gamma_i = sol['feas'][0]
                if gamma_i >= 1-risk_param_epsilon: 
                    obj_i = - sol['obj']
                    if obj_i >= obj_threshold_delta:
                        S_max_i = max(len(S_history[i2]) for i2 in range(i+1))
                        output_data[(random_seed, dim_x)].append(i+1)
                        output_data[(random_seed, dim_x)].append(S_max_i) 
                        break
                    
        best_obj = None            
        for i, iter_i in enumerate(iter_settings):
            if i == 0:
                i_begin = 0
            else:
                i_begin = iter_settings[i-1]
            
            for sol in all_solutions[i_begin:iter_i]:
                gamma_i = sol['feas'][0]
                if gamma_i >= 1-risk_param_epsilon: 
                    obj_i = - sol['obj']
                    if best_obj is None or obj_i > best_obj:
                        best_obj = obj_i
            
            if best_obj is not None:
                opt_gap = 100*(true_prob_obj - best_obj) / true_prob_obj
                gap_at_iterations[iter_i].append(opt_gap)
                output_data[(random_seed, dim_x)].append(opt_gap)
            else:
                output_data[(random_seed, dim_x)].append(None)
            
                
        
        # output_file_name = 'new_output_data'
        with open(output_folder / f'results_{output_file_name}.txt','w+') as f:
            f.write(str(output_data))
        
        run_count += 1
        # print("Completed run: " + str(run_count))

for dim_x in dim_x_settings:
    problem_instance['dim_x'] = dim_x
    output_file_name = f'tp_computation_as_k_increases_k={dim_x}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_N={N}_seeds=1-{num_seeds}'
    # Read from .txt file
    file_path = output_folder / f'results_{output_file_name}.txt'
    output_data = parse_text_file(file_path)
    
    df_output = pd.DataFrame.from_dict(output_data, orient='index')
    
    num_iter_feas_output.append(df_output[2])
    
    for i,delta in enumerate(delta_settings):
        num_iter_output[delta].append(df_output[3 + (i*2)].dropna())
        S_max_output[delta].append(df_output[4 + (i*2)].dropna())
        
        
make_boxplot_k(dim_x_settings, conf_param_alpha, risk_param_epsilon, num_seeds, 
               num_iter_feas_output, 'No. of iterations', 'N='+str(N)+'_feas_numiter', True, plot_type='pdf', log_scale_yn=False)
        
for delta in delta_settings:
    make_boxplot_k(dim_x_settings, conf_param_alpha, risk_param_epsilon, num_seeds, 
                        num_iter_output[delta], 'No. of iterations', 'N='+str(N)+'_'+str(delta)+'_numiter', True, plot_type='pdf', log_scale_yn=False)
    make_boxplot_k(dim_x_settings, conf_param_alpha, risk_param_epsilon, num_seeds, 
                        S_max_output[delta], r'Maximum size $\mathcal{S}_i$', 'N='+str(N)+'_'+str(delta)+'_maxsizeS', True, plot_type='pdf', log_scale_yn=False)

iter_of_interest = [10, 20, 30, 40, 50]
gap_output = []
for iter_i in iter_of_interest:
    gap_output.append(pd.Series(gap_at_iterations[iter_i]))

make_boxplot_iter(iter_of_interest, conf_param_alpha, risk_param_epsilon, num_seeds, 
                  gap_output, 'optimality gap (%)', 'N='+str(N)+'_optgap_iter', True, plot_type='pdf')

iter_of_interest = [10, 100, 500, 1000]
gap_output = []
for iter_i in iter_of_interest:
    gap_output.append(pd.Series(gap_at_iterations[iter_i]))

make_boxplot_iter(iter_of_interest, conf_param_alpha, risk_param_epsilon, num_seeds, 
                  gap_output, 'optimality gap (%)', 'N='+str(N)+'_optgap_iter_v2', True, plot_type='pdf')





