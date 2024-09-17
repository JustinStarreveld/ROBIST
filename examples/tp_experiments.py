"""
Comparison with cal2005 and yan2013 as size of problem (dim_x) increases
"""   
# external imports
import argparse
from sklearn.model_selection import train_test_split
import time
import math
import ast
import numpy as np
import pandas as pd
from pyprojroot import here

# internal imports
from tp import generate_data, solve_SCP, unc_function, eval_OoS, solve_with_yan2013
from scen_opt_methods import determine_cam2008_N_min

# from robist import Robist
# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

output_folder = here("examples/output/ToyProblem/")
output_folder.mkdir(parents=True, exist_ok=True)

def main(args):
    if not args.read_previous:
        cal2005_yn = True
        yan2013_yn = True
        robist_yn = True

        risk_param_epsilon = 0.05
        conf_param_alpha = 0.01                  
        i_max = 200
        num_seeds = 100

        N_robist = 1000
        N1 = math.ceil(N_robist / 3)
        N2 = math.ceil((N_robist - N1) / 2)
        N3 = N_robist - N1 - N2

        OoS_prob_feas_cc = 0
        OoS_prob_feas_yan = 0
        OoS_prob_feas_robist = 0

        for dim_x in [2,3,4,5]:
            
            problem_instance = {}
            problem_instance['dim_x'] = dim_x
            problem_instance['time_limit'] = 1*60*60 
            
            N_campi2008 = determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
            
            random_seed_OoS = 1234
            N_OoS = int(1e6)
            data_OoS = generate_data(random_seed_OoS, N_OoS, dim_x=dim_x)           
            
            print("------------------------------------------------------------------------")
            print("k="+str(dim_x))
            # print("N_campi="+str(N_campi2008))
            print()    

            output_file_name = f'tp_cal2005_yan2013_robist_dim_x={dim_x}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_seeds=1-{num_seeds}'
            
            headers = ['$dim_x$', 'seed']
            
            headers = headers + ['N_campi2008', 'runtime (cal2005)', 'obj. (cal2005)', 'OoS prob. feasible (cal2005)']
            
            headers = headers + ['N_yan2013', 'num iter (yan2013)', 'runtime (yan2013)', 
                                'obj. (yan2013)', 'feas. certificate (yan2013)', 'OoS prob. feasible (yan2013)']
            
            headers = headers + ['$N$', '$N_1$', '$N_2$', '$N_3$', '# iter.~(\\texttt{add})', '\# iter.~(\\texttt{remove})', 
                                'runtime (ROBIST)', 'obj. (ROBIST)', 'feas. certificate (ROBIST)', 'OoS prob. (ROBIST)',
                                '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']
                
            # Write headers to .txt file
            with open(output_folder / f'headers_{output_file_name}.txt','w+') as f:
                f.write(str(headers))
            
            output_data = {}
            
            random_seed_settings = [i+1 for i in range(num_seeds)]
            
            run_count = 0
            for random_seed in random_seed_settings:
                
                output_data[(dim_x, random_seed)] = []
                
                if cal2005_yn:
                    data = generate_data(random_seed, N_campi2008, dim_x=dim_x)     
                    problem_instance['get_dual_sol'] = False          
                    start_time = time.time()
                    x, obj = solve_SCP(data, **problem_instance)
                    runtime_classic = time.time() - start_time
                    obj_classic = - obj
                    OoS_prob_feas_cal2005 = eval_OoS(x, data_OoS)
                    results_classic = [N_campi2008, runtime_classic, obj_classic, OoS_prob_feas_cal2005]
                else:
                    results_classic = [np.nan, np.nan, np.nan, np.nan]
                    
                output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_classic
                
                if yan2013_yn:
                    m_j = 10
                    m = m_j**dim_x
                    N_min = 5*m
                    N_yan2013 = 2*N_min
                    data = generate_data(random_seed, N_yan2013, dim_x=dim_x)  
                    (runtime_yan2013, 
                    num_iter_yan2013,
                    x_yan2013, 
                    obj_yan2013, 
                    lb_yan2013) = solve_with_yan2013(dim_x, risk_param_epsilon,
                                                        conf_param_alpha, data)
                    
                    obj_yan2013 = - obj_yan2013
                    true_prob_yan2013 = eval_OoS(x_yan2013, data_OoS)
                    
                    results_yan2013 = [N_yan2013, num_iter_yan2013, runtime_yan2013, 
                                        obj_yan2013, lb_yan2013, true_prob_yan2013]
                else:
                    results_yan2013 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    
                output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_yan2013
                
                if robist_yn:
                    stop_criteria={'max_num_iterations': i_max}
                    eval_unc_obj = None
                    eval_unc_constr = [{'function': unc_function,
                                        'info': {'risk_measure': 'probability',
                                                'desired_rhs': 1 - risk_param_epsilon}}]
                    
                    
                    data = generate_data(random_seed, N_robist, dim_x=dim_x)     
                    data_train, rest = train_test_split(data, train_size=(N1/N_robist), random_state=random_seed)
                    data_valid, data_test = train_test_split(rest, train_size=(N2/(N2+N3)), random_state=random_seed)
                    
                    algorithm = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                        data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                                        use_dual_sol=True, verbose=False)
                    

                    (best_sol, 
                    runtime_robist, 
                    num_iter, 
                    pareto_frontier, 
                    S_history) = algorithm.run(stop_criteria=stop_criteria,
                                                random_seed=random_seed)
                                                        
                    lb_robist = best_sol['feas'][0]
                    OoS_prob_feas_robist = eval_OoS(best_sol['sol'], data_OoS)
                    obj_robist = - best_sol['obj']
                    S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
                    S_max = max(len(S_i) for S_i in S_history)
                    num_iter_add = num_iter['add']
                    num_iter_remove = num_iter['remove']
                    
                    results_robist = [N_robist, N1, N2, N3, num_iter_add, num_iter_remove,
                                    runtime_robist, obj_robist, lb_robist, OoS_prob_feas_robist,
                                    S_avg, S_max]
                else:
                    results_robist = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    
                output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_robist
                
                with open(output_folder / f'results_{output_file_name}_new.txt','w+') as f:
                    f.write(str(output_data))
                
                run_count += 1
                print("Completed run: " + str(run_count))
    
    else:
        # Read in previous output from .txt file
        file_path = output_folder / f'results_{output_file_name}_new.txt'
        output_data = parse_text_file(file_path)
    
    # Aggregate data to get avg across random seed runs
    df_output = pd.DataFrame.from_dict(output_data, orient='index')
    print(df_output.mean())
    
    OoS_prob_feas_cc += df_output.mean()[3]
    OoS_prob_feas_yan += df_output.mean()[9]
    OoS_prob_feas_robist += df_output.mean()[18]
    

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-previous", action=argparse.BooleanOptionalAction, default=False,
                        help="Read results from previous experiment(s). Default: --no-read-previous.")
    args = parser.parse_args()
    main(args)
    
