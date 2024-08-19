"""
In this script we examine the impact of upsilon for ROBIST when applied to the 
toy problem
"""

# external imports
import numpy as np
import math
import statistics
from sklearn.model_selection import train_test_split
from pyprojroot import here

# internal imports
from tp2 import generate_data, solve_SCP, solve_toyproblem_true_prob, unc_func, get_true_prob

# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

# from robist import Robist

output_folder = here("examples/output/ToyProblem2/")
output_folder.mkdir(parents=True, exist_ok=True)

# set parameter values
risk_param_epsilon = 0.05
conf_param_alpha = 0.01
numeric_precision = 1e-6

i_max = 200
stop_criteria={'max_num_iterations': i_max}
eval_unc_obj = None
eval_unc_constr = [{'function': unc_func,
                    'info': {'risk_measure': 'probability',
                            'desired_rhs': 1 - risk_param_epsilon}}]

random_seed = 123
num_reps = 10
dim_x_settings = [20, 200, 2000]
N_settings = [3000, 30000]
alg_settings = ['random_vio', 'random_any', 'largest_vio']

i_robust_results = {}
opt_gap_results = {}
S_max_results = {}

for dim_x in dim_x_settings:
  problem_instance = {}
  problem_instance['dim_x'] = dim_x
  problem_instance['time_limit'] = 1*60*60 

  # solve toy problem exactly
  true_prob_obj = solve_toyproblem_true_prob(dim_x, risk_param_epsilon)

  for N in N_settings:
    N1 = math.ceil(N / 3)
    N2 = math.ceil((N - N1) / 2)
    N3 = N - N1 - N2

    i_robust_results[(dim_x, N)] = {strategy: [] for strategy in alg_settings}
    opt_gap_results[(dim_x, N)] = {strategy: [] for strategy in alg_settings}
    S_max_results[(dim_x, N)] = {strategy: [] for strategy in alg_settings}

    output_file_name = f'tp2_addStrategy_k={dim_x}_N={N}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_imax={i_max}_seeds=1-{num_reps}'
    output_data = {}

    for rep in range(num_reps):
      random_seed_rep = random_seed * (random_seed+rep)
      data = generate_data(random_seed_rep, N, dim_x=dim_x)     
      data_train, rest = train_test_split(data, train_size=(N1/N), random_state=random_seed_rep)
      data_valid, data_test = train_test_split(rest, train_size=(N2/(N2+N3)), random_state=random_seed_rep)

      for strategy in alg_settings:
        robist = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                        data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                        add_strategy=strategy,
                        verbose=False)

        (best_sol, 
          runtime_robist, 
          num_iter, 
          non_dominated_solutions, 
          S_history,
          all_solutions) = robist.run(stop_criteria=stop_criteria, 
                                      return_all_solutions=True)
        
        # gather info
        S_max = max(len(S_i) for S_i in S_history)

        i_robust = i_max
        true_best_obj_robist = 0
        for i,sol in enumerate(all_solutions):
              x_i = sol['sol']
              if get_true_prob(x_i, dim_x) >= 1-risk_param_epsilon:
                  if i_robust == i_max:
                     i_robust = i
                  obj_i = - sol['obj']
                  if obj_i > true_best_obj_robist and sol['cert_valid'][0] >= 1-risk_param_epsilon:
                      true_best_obj_robist = obj_i
        
        opt_gap = 100 * (true_prob_obj-true_best_obj_robist) / true_prob_obj

        # store info
        i_robust_results[(dim_x, N)][strategy].append(i_robust)
        opt_gap_results[(dim_x, N)][strategy].append(opt_gap)
        S_max_results[(dim_x, N)][strategy].append(S_max)

        output_data[(dim_x, N, rep, strategy)] = {'i_robust': i_robust, 
                                            'opt_gap': opt_gap, 
                                            'S_max': S_max}
    
    with open(output_folder / f'results_{output_file_name}_new.txt', 'w+') as f:
      f.write(str(output_data))

    # print averaged results
    print()
    print(f'(averaged) results for k={dim_x} and N={N}: ')
    for strategy in alg_settings:
        print(f'{strategy} : {statistics.fmean(i_robust_results[(dim_x, N)][strategy]):.1f}, {statistics.fmean(S_max_results[(dim_x, N)][strategy]):.1f}, {statistics.fmean(opt_gap_results[(dim_x, N)][strategy]):.2f}')



def make_table(dim_x_settings, N_settings, conf_param_alpha, risk_param_epsilon, num_reps, i_max, alg_settings, i_robust_results, opt_gap_results, S_max_results):
  table = "\\begin{table}[]\n"
  table += f"\\caption{{Analysis of impact of the addition strategy on performance of ROBIST (average over {num_reps} replications). We compare adding a scenario in three ways: (i) RV: random from amongst scenarios that violate the uncertain constraint, (ii) RA: random from amongst all scenarios in training set and (iii) MV: pick a scenario with maximal constraint violation. Settings: $\\epsilon$={risk_param_epsilon}, $\\alpha$={conf_param_alpha}, $i_{{max}}$={i_max}.}}\n"
  table += "\\label{tab:analyze_addstrategy}\n"
  table += "\\centering\n"
  table += "\\resizebox{0.95\\textwidth}{!}{%\n"
  table += "\\begin{tabular}{@{}llrrrrrrrrrrrr@{}}\n"
  table += "\\toprule\n"
  table += "&  & \\multicolumn{1}{l}{} & \\multicolumn{3}{c}{$i_{\\mathrm{robust}}$} & \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{$|\\mathcal{S}_i|^{\\max}$} & \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{Opt. Gap (\\%)} \\\\ \\cmidrule(lr){4-6} \\cmidrule(lr){8-10} \\cmidrule(l){12-14} \n"
  table += "&  & \\multicolumn{1}{l}{Addition Strategy}  & \\multicolumn{1}{c}{RV} & \\multicolumn{1}{c}{RA} & \\multicolumn{1}{c}{MV} & \\multicolumn{1}{c}{} & \\multicolumn{1}{c}{RV} & \\multicolumn{1}{c}{RA} & \\multicolumn{1}{c}{MV} & \\multicolumn{1}{c}{} & \\multicolumn{1}{c}{RV} & \\multicolumn{1}{c}{RA} & \\multicolumn{1}{c}{MV} \\\\ \n"
  table += "$k$ & $N$ & \\multicolumn{1}{l}{} &  &  &  &  &  &  &  &  &  &  &  \\\\ \\midrule \n"

  dim_x_prev = 0
  for dim_x in dim_x_settings:
      for N in N_settings:
          if dim_x != dim_x_prev:
              if dim_x != dim_x_settings[0]:
                  table +=  "&  &  &  &  &  &  &  &  &  &  &  &  & \\\\\n"
              table += f"{dim_x} & {N} & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(i_robust_results[(dim_x, N)][strategy]):.1f}"
              table += " & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(S_max_results[(dim_x, N)][strategy]):.1f}"
              table += " & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(opt_gap_results[(dim_x, N)][strategy]):.2f}"
              table += " \\\\\n"
          else:
              table += f"  & {N} & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(i_robust_results[(dim_x, N)][strategy]):.1f}"
              table += " & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(S_max_results[(dim_x, N)][strategy]):.1f}"
              table += " & "
              for strategy in alg_settings:
                 table += f" & {statistics.fmean(opt_gap_results[(dim_x, N)][strategy]):.2f}"
              table += " \\\\\n"
          dim_x_prev = dim_x


  table += "\\bottomrule\n"
  table += "\\end{tabular}%\n"
  table += "}\n"
  table += "\\end{table}\n"

  return table

print()
print(make_table(dim_x_settings, N_settings, conf_param_alpha, risk_param_epsilon, num_reps, i_max, alg_settings, i_robust_results, opt_gap_results, S_max_results))

        


      
    










