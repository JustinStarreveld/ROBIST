"""
In this script we compare ROBIST to ALSO-X methods for the linear CCP problem, see Jiang and Xie (2022)
"""

# external imports
import math
import numpy as np
from pyprojroot import here

# internal imports
from jiang2022 import get_param_data, generate_unc_param_data, solve_SCP, unc_function, eval_OoS, solve_with_jiang2022, solve_with_jiang2022_plus, solve_SAA_with_MIO

# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

# from robist import Robist

output_folder = here("examples/output/ToyProblem_Jiang2022/")
output_folder.mkdir(parents=True, exist_ok=True)
output_data_avg = {}

# set parameter values
risk_param_epsilon = 0.10
conf_param_alpha = 0.01
numeric_precision = 1e-6

i_max = 200
stop_criteria={'max_num_iterations': i_max} 
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

random_seed = 45 
N_OoS = 1000000
random_seed_OoS = 451516

num_reps = 1
dim_x_settings = [100, 200, 300]
N_settings = [1000, 5000, 10000, 50000]

for dim_x in dim_x_settings:

  OoS_data = generate_unc_param_data(random_seed_OoS, dim_x=dim_x, N=N_OoS)
  
  for N in N_settings:
    output_file_name = f'compare_jiang2022_mio_robist_dimx={dim_x}_N={N}_epsilon={risk_param_epsilon}_alpha={conf_param_alpha}_numReps={num_reps}_seed={random_seed}'  
    output_data = {}

    time_mio = []
    obj_mio = []
    p_OoS_mio = []

    time_jiang2022 = []
    obj_jiang2022 = []
    p_OoS_jiang2022 = []

    time_jiang2022_plus = []
    obj_jiang2022_plus = []
    p_OoS_jiang2022_plus = []

    N1 = math.ceil(N / 3)
    N2 = math.ceil((N - N1) / 2)
    N3 = N - N1 - N2
    time_robist = []
    obj_robist = []
    cert_robist = []
    p_OoS_robist = []

    for rep in range(num_reps):
      random_seed_rep = random_seed * (random_seed+rep)
      problem_instance = get_param_data(random_seed_rep, dim_x=dim_x, N=N)
      uncertain_data = problem_instance['a'][0]
      problem_instance['dim_x'] = dim_x

      splitted_data = np.split(uncertain_data, [N1, N1+N2, N1+N2+N3])
      data_train = splitted_data[0]
      data_valid = splitted_data[1]
      data_test = splitted_data[2]

      robist = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                      data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                      verbose=False)

      (best_sol, 
        runtime_robist, 
        num_iter, 
        non_dominated_solutions, 
        S_history) = robist.run(stop_criteria=stop_criteria)
      
      time_robist.append(runtime_robist)
      obj_robist.append(-best_sol['obj'])
      cert_robist.append(best_sol['feas'][0])
      p_OoS_robist.append(eval_OoS(best_sol['sol'], OoS_data, **problem_instance))
      
      time, obj, sol = solve_SAA_with_MIO(risk_param_epsilon, uncertain_data, problem_instance, time_limit=runtime_robist, verbose=False)
      time_mio.append(time)
      obj_mio.append(-obj)
      p_OoS_mio.append(eval_OoS(sol, OoS_data, **problem_instance))

      time, iter_jiang2022, obj, sol = solve_with_jiang2022(risk_param_epsilon, N, problem_instance, verbose=False) 
      time_jiang2022.append(time)
      obj_jiang2022.append(-obj)
      p_OoS_jiang2022.append(eval_OoS(sol, OoS_data, **problem_instance))

      time, iter_jiang2022_plus, obj, sol = solve_with_jiang2022_plus(risk_param_epsilon, N, problem_instance, verbose=False) 
      time_jiang2022_plus.append(time)
      obj_jiang2022_plus.append(-obj)
      p_OoS_jiang2022_plus.append(eval_OoS(sol, OoS_data, **problem_instance))

      print()
      print(f'results for dim_x={dim_x}, N={N}, rep={rep}:')
      print(f'MIO             : time={time_mio[rep]:.1f}, obj={obj_mio[rep]:.2f}, p={p_OoS_mio[rep]:.3f}')
      print(f'ALSO-X          : time={time_jiang2022[rep]:.1f}, obj={obj_jiang2022[rep]:.2f}, p={p_OoS_jiang2022[rep]:.3f}')
      print(f'ALSO-X+         : time={time_jiang2022_plus[rep]:.1f}, obj={obj_jiang2022_plus[rep]:.2f}, p={p_OoS_jiang2022_plus[rep]:.3f}')
      print(f'ROBIST          : time={time_robist[rep]:.1f}, obj={obj_robist[rep]:.2f}, p={p_OoS_robist[rep]:.3f}, with lb={cert_robist[rep]:.3f}')

      output_data[(dim_x, N, rep)] = {
              "time_mio": time_mio[rep],
              "obj_mio": obj_mio[rep],
              "p_OoS_mio": p_OoS_mio[rep],
              "time_jiang2022": time_jiang2022[rep],
              "obj_jiang2022": obj_jiang2022[rep],
              "p_OoS_jiang2022": p_OoS_jiang2022[rep],
              "time_jiang2022_plus": time_jiang2022_plus[rep],
              "obj_jiang2022_plus": obj_jiang2022_plus[rep],
              "p_OoS_jiang2022_plus": p_OoS_jiang2022_plus[rep],
              "time_robist": time_robist[rep],
              "obj_robist": obj_robist[rep],
              "p_OoS_robist": p_OoS_robist[rep],
              "cert_robist": cert_robist[rep]
          }

      with open(output_folder / f'results_{output_file_name}_new.txt', 'w+') as f:
        f.write(str(output_data))

    print()
    print(f'averaged results for dim_x={dim_x}, N={N}:')
    print(f'MIO             : time={round(sum(time_mio)/len(time_mio),1)}, obj={round(sum(obj_mio)/len(obj_mio),2)}, p={round(sum(p_OoS_mio)/len(p_OoS_mio),3)}')
    print(f'ALSO-X          : time={round(sum(time_jiang2022)/len(time_jiang2022),1)}, obj={round(sum(obj_jiang2022)/len(obj_jiang2022),2)}, p={round(sum(p_OoS_jiang2022)/len(p_OoS_jiang2022),3)}')
    print(f'ALSO-X+         : time={round(sum(time_jiang2022_plus)/len(time_jiang2022_plus),1)}, obj={round(sum(obj_jiang2022_plus)/len(obj_jiang2022_plus),2)}, p={round(sum(p_OoS_jiang2022_plus)/len(p_OoS_jiang2022_plus),3)}')
    print(f'ROBIST          : time={round(sum(time_robist)/len(time_robist),1)}, obj={round(sum(obj_robist)/len(obj_robist),2)}, p={round(sum(p_OoS_robist)/len(p_OoS_robist),3)}, with lb={round(sum(cert_robist)/len(cert_robist),3)}')

    output_data_avg[(dim_x, N)] = {
            "time_mio": sum(time_mio)/len(time_mio),
            "obj_mio": sum(obj_mio)/len(obj_mio),
            "p_OoS_mio": sum(p_OoS_mio)/len(p_OoS_mio),
            "time_jiang2022": sum(time_jiang2022)/len(time_jiang2022),
            "obj_jiang2022": sum(obj_jiang2022)/len(obj_jiang2022),
            "p_OoS_jiang2022": sum(p_OoS_jiang2022)/len(p_OoS_jiang2022),
            "time_jiang2022_plus": sum(time_jiang2022_plus)/len(time_jiang2022_plus),
            "obj_jiang2022_plus": sum(obj_jiang2022_plus)/len(obj_jiang2022_plus),
            "p_OoS_jiang2022_plus": sum(p_OoS_jiang2022_plus)/len(p_OoS_jiang2022_plus),
            "time_robist": sum(time_robist)/len(time_robist),
            "obj_robist": sum(obj_robist)/len(obj_robist),
            "p_OoS_robist": sum(p_OoS_robist)/len(p_OoS_robist),
            "cert_robist": sum(cert_robist)/len(cert_robist)
        }

    with open(output_folder / f'avg_results_{output_file_name}_new.txt', 'w+') as f:
      f.write(str(output_data_avg[(dim_x, N)]))

def make_table(dim_x_settings, N_settings, alpha, epsilon, num_reps, num_iter, output_data_avg):
    table = "\\begin{table}[]\n"
    table += "\\centering\n"
    table += "\\resizebox{0.95\\textwidth}{!}{%\n"
    table += "\\begin{tabular}{@{}llrrrrrrrrrrrrrrrr@{}}\n"
    table += "\\toprule\n"
    table += " &  & \\multicolumn{3}{c}{MIO} &  & \\multicolumn{3}{c}{ALSO-X} &  & \\multicolumn{3}{c}{ALSO-X+} &  & \\multicolumn{4}{c}{ROBIST} \\\\ \\cline{3-5} \\cline{7-9} \\cline{11-13} \\cline{15-18} \n"
    table += "$d_{\mathbf{x}}/d_{\mathbf{z}}$~ & $N$ & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{$\\mathbb{P}(\\text{feas.})$} & & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{$\\mathbb{P}(\\text{feas.})$} &  & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{$\\mathbb{P}(\\text{feas.})$} &  & \\multicolumn{1}{r}{Time (s)} & \\multicolumn{1}{r}{Obj.} & \\multicolumn{1}{r}{$\\mathbb{P}(\\text{feas.})$} & \\multicolumn{1}{r}{Cert.} \\\\ \\midrule\n"

    dim_x_prev = 0
    for dim_x in dim_x_settings:
        for N in N_settings:
            time_mio = round(output_data_avg[(dim_x, N)]["time_mio"], 3)
            obj_mio = round(output_data_avg[(dim_x, N)]["obj_mio"], 3)
            p_OoS_mio = round(output_data_avg[(dim_x, N)]["p_OoS_mio"], 3)
            time_jiang2022 = round(output_data_avg[(dim_x, N)]["time_jiang2022"], 3)
            obj_jiang2022 = round(output_data_avg[(dim_x, N)]["obj_jiang2022"], 3)
            p_OoS_jiang2022 = round(output_data_avg[(dim_x, N)]["p_OoS_jiang2022"], 3)
            time_jiang2022_plus = round(output_data_avg[(dim_x, N)]["time_jiang2022_plus"], 3)
            obj_jiang2022_plus = round(output_data_avg[(dim_x, N)]["obj_jiang2022_plus"], 3)
            p_OoS_jiang2022_plus = round(output_data_avg[(dim_x, N)]["p_OoS_jiang2022_plus"], 3)
            time_robist = round(output_data_avg[(dim_x, N)]["time_robist"], 3)
            obj_robist = round(output_data_avg[(dim_x, N)]["obj_robist"], 3)
            p_OoS_robist = round(output_data_avg[(dim_x, N)]["p_OoS_robist"], 3)
            cert_robist = round(output_data_avg[(dim_x, N)]["cert_robist"], 3)
            
            if dim_x != dim_x_prev:
                if dim_x != dim_x_settings[0]:
                    table +=  "&  &  & &  &  &  &  &  &  &  &  &  &  &  &  & \\\\\n"
                table += f"{dim_x} & {N} & {time_mio:.1f} & {obj_mio:.2f} & {p_OoS_mio:.3f} & & {time_jiang2022:.1f} & {obj_jiang2022:.2f} & {p_OoS_jiang2022:.3f} &  & {time_jiang2022_plus:.1f} & {obj_jiang2022_plus:.2f} & {p_OoS_jiang2022_plus:.3f} & & {time_robist:.1f} & {obj_robist:.2f} & {p_OoS_robist:.3f} & {cert_robist:.3f}  \\\\\n"
            else:
                table += f" & {N} & {time_mio:.1f} & {obj_mio:.2f} & {p_OoS_mio:.3f} & & {time_jiang2022:.1f} & {obj_jiang2022:.2f} & {p_OoS_jiang2022:.3f} &  & {time_jiang2022_plus:.1f} & {obj_jiang2022_plus:.2f} & {p_OoS_jiang2022_plus:.3f} & & {time_robist:.1f} & {obj_robist:.2f} & {p_OoS_robist:.3f} & {cert_robist:.3f}  \\\\\n"
            dim_x_prev = dim_x

    table += "\\bottomrule\n"
    table += "\\end{tabular}%\n"
    table += "}\n"
    table += f"\\caption{{Comparison between MIO, ALSO-X, ALSO-X+ and ROBIST (average over {num_reps} random instances). Settings: $\\epsilon$={epsilon}, $\\alpha$={alpha}, $i_{{max}}$={num_iter}, $N_{{OoS}}=10^5$.}}\n"
    table += "\\label{tab:compare_jiang2022}\n"
    table += "\\end{table}\n"

    return table

print()
print(make_table(dim_x_settings, N_settings, conf_param_alpha, risk_param_epsilon, num_reps, i_max, output_data_avg))


