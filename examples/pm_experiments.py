"""
Comparison with results presented in Bertsimas et al. (2018)
"""  
# external imports
import argparse
from sklearn.model_selection import train_test_split
import math
import ast
import numpy as np
import pandas as pd
from pyprojroot import here

# internal imports
from pm import generate_data_natarajan2008, solve_SCP, unc_function, eval_OoS
from scen_opt_methods import solve_with_cal2013

# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist
# from robist import Robist

output_folder = here("examples/output/PortfolioManagement/")
output_folder.mkdir(parents=True, exist_ok=True)

def main(args):
    if not args.read_previous:
        # set parameter values
        risk_param_epsilon = 0.10
        conf_param_alpha = 0.10
        dim_x = 10

        N = 2000
        N1 = math.ceil(N / 3)
        N2 = math.ceil((N - N1) / 2)
        N3 = N - N1 - N2

        generate_data = generate_data_natarajan2008
        # generate_data = generate_data_mohajerin2018

        problem_instance = {}
        problem_instance['dim_x'] = dim_x
        problem_instance['time_limit'] = 1*60 

        # ROBIST settings:
        i_max = 200
        stop_criteria={'max_num_iterations': i_max}
        eval_unc_obj = {'function': unc_function,
                        'info': {'risk_measure': 'probability',
                                'desired_rhs': 1-risk_param_epsilon}}

        eval_unc_constr = None

        # generate extra out-of-sample (OoS) data
        random_seed_OoS = 1234
        N_OoS = int(1e6)
        data_OoS = generate_data(random_seed_OoS, N_OoS, dim_x=dim_x)


        print("------------------------------------------------------------------------")
        print("N="+str(N))
        print()    

        num_seeds = 100
        random_seed_settings = [i+1 for i in range(num_seeds)]
        calafiore2013_yn = False

        output_file_name = f'pm_N={N}_dimx={dim_x}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_imax={i_max}_seeds=1-{num_seeds}'

        headers = ['seed', '$N$', 
                '$N_1$', '$N_2$', '$N_3$',
                '\# iter.~(\\texttt{add})', '\# iter.~(\\texttt{remove})', 
                    'runtime (ROBIST)', 'runtime (cal2013)', 
                    'sol. (ROBIST)', 'sol. (cal2013)', 
                    'obj. (ROBIST)', 'obj. (cal2013)', 
                    'p_vio_OoS (ROBIST)', 'p_vio_OoS (cal2013)', 
                    'VaR_OoS (ROBIST)', 'VaR_OoS (cal2013)',
                    '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']

        # Write headers to .txt file
        with open(output_folder / f'headers_{output_file_name}.txt','w+') as f:
            f.write(str(headers))

        output_data = {}
        run_count = 0
        for random_seed in random_seed_settings:

            data = generate_data(random_seed, N, dim_x=dim_x)               
            data_train, rest = train_test_split(data, train_size=(N1/N), random_state=random_seed)
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
            
            sol_robist = best_sol['sol']
            obj_robist = - best_sol['obj']
            S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
            S_max = max(len(S_i) for S_i in S_history)
                                                
            p_vio_ROBIST, VaR_ROBIST, CVaR_ROBIST = eval_OoS(sol_robist, best_sol['obj'], data_OoS, eval_unc_obj, **problem_instance)
            
            
            if calafiore2013_yn: # calafiore2013 method    
                q_max = -1
                sol_cal2013, obj, runtime_cal2013, q = solve_with_cal2013(solve_SCP, problem_instance, dim_x, data, risk_param_epsilon,
                                                                    conf_param_alpha, q=q_max)
                obj_cal2013 = - obj
                q_max = q
                p_vio_cal2013, VaR_cal2013, CVaR_cal2013 = eval_OoS(sol_cal2013, obj, data_OoS, eval_unc_obj, **problem_instance)
            else:
                q, runtime_cal2013, sol_cal2013, obj_cal2013, p_vio_cal2013, VaR_cal2013 = np.nan, np.nan, [np.nan], np.nan, np.nan, np.nan
                
            
            output_data[(random_seed, N)] = [N1, N2, N3, 
                                            num_iter['add'], num_iter['remove'], q,
                                            runtime_robist, runtime_cal2013,
                                            sol_robist[0], sol_cal2013[0],
                                            obj_robist, obj_cal2013,
                                            p_vio_ROBIST, p_vio_cal2013,
                                            VaR_ROBIST, VaR_cal2013,
                                            S_avg, S_max]
            
            
            with open(output_folder / f'results_{output_file_name}_new.txt','w+') as f:
                f.write(str(output_data))
            
            run_count += 1
            print("Completed run: " + str(run_count))

    else:
        # Read in previous output from .txt file
        file_path = output_folder / f'results_{output_file_name}_new.txt'
        output_data = parse_text_file(file_path)

    if args.plot_results:
        df_Sol = pd.DataFrame({key: pd.Series(val[8]) for key, val in output_data.items()})
        df_Sol = df_Sol.T
        df_Sol.drop(0, axis=1, inplace=True)
        plot_portfolio_holdings(df_Sol)

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

def plot_portfolio_holdings(solutions):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [5, 3]
    plt.rcParams['figure.dpi'] = 800 # can be increased for better quality
    
    # Follow Bertsimas et al. (2018) by plotting the 10 & 90% quantiles with average (over 100 runs)
    # get quantiles and mean for each asset seperately
    q_10 = []
    means = []
    q_90 = []
    for i in solutions.columns:
        means.append(np.mean(solutions.loc[:,i]))
        q_10.append(np.percentile(solutions.loc[:,i], 10))
        q_90.append(np.percentile(solutions.loc[:,i], 90))
    
    x = ["x" + str(i) for i in solutions.columns]

    asymmetric_error = [np.array(means)-np.array(q_10), np.array(q_90)-np.array(means)]

    plt.errorbar(x, means, yerr=asymmetric_error,
                marker='o', markersize=2,
                linestyle='dotted')

    plt.xticks(x) # to ensure that all assets are shown on x-axis
    #plt.xlabel("Assets")
    plt.ylabel("Holding (%)")
    plt.tight_layout()
    plt.show()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-previous", action=argparse.BooleanOptionalAction, default=False,
                        help="Read results from previous experiment(s). Default: --no-read-previous.")
    parser.add_argument("--plot-results", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot results (shows quantiles of portfolio holdings / solution). Default: --no-plot-results.")
    args = parser.parse_args()
    main(args)
