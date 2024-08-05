"""
In this script we compare ROBIST to ALSO-X methods for the linear CCP problem, see Jiang and Xie (2022)
"""

# external imports
import math
import numpy as np
from pyprojroot import here

# internal imports
from tp2_jiang2022 import get_param_data, generate_unc_param_data, solve_SCP, unc_function, eval_OoS, solve_with_jiang2022, solve_with_jiang2022_plus

# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

# from robist import Robist

output_folder = here("examples/output/ToyProblem_Jiang2022/")
output_folder.mkdir(parents=True, exist_ok=True)
output_data = {}

# set parameter values
risk_param_epsilon = 0.05
conf_param_alpha = 0.01
numeric_precision = 1e-6

i_max = 100
stop_criteria={'max_num_iterations': i_max} 
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

random_seed = 45 # np.random.randint(0, 999999) #942179
N_OoS = 10000

num_reps = 5
dim_x_settings = [20, 40, 100]
N_settings = [400, 600, 1000]

for dim_x in dim_x_settings:

  OoS_data = generate_unc_param_data(random_seed, dim_x=dim_x, N=N_OoS)
  
  for N in N_settings:
    output_file_name = f'compare_Jiang2022_dimx={dim_x}_N={N}_epsilon={risk_param_epsilon}_alpha={conf_param_alpha}_numReps={num_reps}_seed={random_seed}'  

    if 

    time_robist = []
    obj_robist = []
    cert_robist = []
    p_OoS_robist = []
    
    time_jiang2022 = []
    obj_jiang2022 = []
    p_OoS_jiang2022 = []

    time_jiang2022_plus = []
    obj_jiang2022_plus = []
    p_OoS_jiang2022_plus = []

    N1 = math.ceil(N / 3)
    N2 = math.ceil((N - N1) / 2)
    N3 = N - N1 - N2

    for rep in range(num_reps):
      random_seed_rep = random_seed * (random_seed+rep)
      problem_instance = get_param_data(random_seed, dim_x=dim_x, N=N)
      problem_instance['dim_x'] = dim_x
      problem_instance['time_limit'] = 5*60 
      
      uncertain_data = problem_instance['a'][0]
      splitted_data = np.split(uncertain_data, [N1, N1+N2, N1+N2+N3])
      data_train = splitted_data[0]
      data_valid = splitted_data[1]
      data_test = splitted_data[2]

      # OPTION 3: use 3 data sets, where third is only used for single final evaluation
      robist_opt3 = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                      data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                      verbose=False)

      (best_sol, 
        runtime_robist, 
        num_iter, 
        non_dominated_solutions, 
        S_history, 
        all_solutions_robist) = robist_opt3.run(stop_criteria=stop_criteria, return_all_solutions=True)
      
      time_robist.append(runtime_robist)
      obj_robist.append(-best_sol['obj'])
      cert_robist.append(best_sol['feas'][0])
      p_OoS_robist.append(eval_OoS(best_sol['sol'], OoS_data, **problem_instance))

      time, iter_jiang2022, obj, sol = solve_with_jiang2022(risk_param_epsilon, N, problem_instance, verbose=False) 
      
      time_jiang2022.append(time)
      obj_jiang2022.append(-obj)
      p_OoS_jiang2022.append(eval_OoS(sol, OoS_data, **problem_instance))

      time, iter_jiang2022_plus, obj, sol = solve_with_jiang2022_plus(risk_param_epsilon, N, problem_instance, verbose=False) 
      
      time_jiang2022_plus.append(time)
      obj_jiang2022_plus.append(-obj)
      p_OoS_jiang2022_plus.append(eval_OoS(sol, OoS_data, **problem_instance))

    # print()
    # print(f'results for dim_x={dim_x} and N={N}:')
    # print(f'ROBIST (option3): time={round(sum(time_robist)/len(time_robist),2)}, obj={round(sum(obj_robist)/len(obj_robist),2)}, p={round(sum(p_OoS_robist)/len(p_OoS_robist),3)}, with lb={round(sum(cert_robist)/len(cert_robist),3)}')
    # print(f'ALSO-X          : time={round(sum(time_jiang2022)/len(time_jiang2022),2)}, obj={round(sum(obj_jiang2022)/len(obj_jiang2022),2)}, p={round(sum(p_OoS_jiang2022)/len(p_OoS_jiang2022),3)}')
    # print(f'ALSO-X+         : time={round(sum(time_jiang2022_plus)/len(time_jiang2022_plus),2)}, obj={round(sum(obj_jiang2022_plus)/len(obj_jiang2022_plus),2)}, p={round(sum(p_OoS_jiang2022_plus)/len(p_OoS_jiang2022_plus),3)}')
    
    output_data[(dim_x, N)] = {
            "time_robist": sum(time_robist)/len(time_robist),
            "obj_robist": sum(obj_robist)/len(obj_robist),
            "p_OoS_robist": sum(p_OoS_robist)/len(p_OoS_robist),
            "time_jiang2022": sum(time_jiang2022)/len(time_jiang2022),
            "obj_jiang2022": sum(obj_jiang2022)/len(obj_jiang2022),
            "p_OoS_jiang2022": sum(p_OoS_jiang2022)/len(p_OoS_jiang2022),
            "time_jiang2022_plus": sum(time_jiang2022_plus)/len(time_jiang2022_plus),
            "obj_jiang2022_plus": sum(obj_jiang2022_plus)/len(obj_jiang2022_plus),
            "p_OoS_jiang2022_plus": sum(p_OoS_jiang2022_plus)/len(p_OoS_jiang2022_plus)
        }

    with open(output_folder / f'results_{output_file_name}_new.txt', 'w+') as f:
      f.write(str(output_data[(dim_x, N)]))

def make_table(dim_x_settings, N_settings, alpha, epsilon, num_reps, num_iter, output_data):
    table = "\\begin{table}[]\n"
    table += "\\centering\n"
    table += "\\resizebox{0.95\\textwidth}{!}{%\n"
    table += "\\begin{tabular}{@{}llrrrrrrrrrrr@{}}\n"
    table += "\\toprule\n"
    table += " &  & \\multicolumn{3}{c}{ALSO-X} &  & \\multicolumn{3}{c}{ALSO-X+} &  & \\multicolumn{3}{c}{ROBIST} \\\\ \\cline{3-5} \\cline{7-9} \\cline{11-13} \n"
    table += "N & dim & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{P(feasible)} &  & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{P(feasible)} &  & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{P(feasible)} \\\\ \\midrule\n"

    N_prev = 0
    for N in N_settings:
        for dim_x in dim_x_settings:
            time_jiang2022 = round(output_data[(dim_x, N)]["time_jiang2022"], 3)
            obj_jiang2022 = round(output_data[(dim_x, N)]["obj_jiang2022"], 3)
            p_OoS_jiang2022 = round(output_data[(dim_x, N)]["p_OoS_jiang2022"], 3)
            time_jiang2022_plus = round(output_data[(dim_x, N)]["time_jiang2022_plus"], 3)
            obj_jiang2022_plus = round(output_data[(dim_x, N)]["obj_jiang2022_plus"], 3)
            p_OoS_jiang2022_plus = round(output_data[(dim_x, N)]["p_OoS_jiang2022_plus"], 3)
            time_robist = round(output_data[(dim_x, N)]["time_robist"], 3)
            obj_robist = round(output_data[(dim_x, N)]["obj_robist"], 3)
            p_OoS_robist = round(output_data[(dim_x, N)]["p_OoS_robist"], 3)
            
            if N != N_prev:
                if dim_x != dim_x_settings[0]:
                    table +=  "&  &  &  &  &  &  &  &  &  &  &  &  \\\\\n"
                table += f"{N} & {dim_x} & {time_jiang2022:.2f} & {obj_jiang2022:.2f} & {p_OoS_jiang2022:.3f} &  & {time_jiang2022_plus:.2f} & {obj_jiang2022_plus:.2f} & {p_OoS_jiang2022_plus:.3f} & & {time_robist:.2f} & {obj_robist:.2f} & {p_OoS_robist:.3f}  \\\\\n"
            else:
                table += f"{N} & {dim_x} & {time_jiang2022:.2f} & {obj_jiang2022:.2f} & {p_OoS_jiang2022:.3f} &  & {time_jiang2022_plus:.2f} & {obj_jiang2022_plus:.2f} & {p_OoS_jiang2022_plus:.3f} & & {time_robist:.2f} & {obj_robist:.2f} & {p_OoS_robist:.3f}  \\\\\n"
            N_prev = N

    table += "\\bottomrule\n"
    table += "\\end{tabular}%\n"
    table += "}\n"
    table += f"\\caption{{Comparison between ALSO-X, ALSO-X+ and ROBIST (average over {num_reps} random instances). Settings: epsilon={epsilon}, alpha={alpha}, i_max={num_iter}.}}\n"
    table += "\\label{tab:compare_jiang2022}\n"
    table += "\\end{table}\n"

    return table

print(make_table(dim_x_settings, N_settings, conf_param_alpha, risk_param_epsilon, num_reps, i_max, output_data))


