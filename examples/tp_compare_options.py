"""
In this script we compare two options for ROBIST when applied to the 
toy problem
"""

# external imports
import numpy as np
from pyprojroot import here

# internal imports
from tp import generate_data, solve_SCP, unc_function

# to make sure that we're using the local .py version of robist
import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

# from robist import Robist

# set parameter values
risk_param_epsilon = 0.10
conf_param_alpha = 0.01
numeric_precision = 1e-6

i_max = 100
stop_criteria={'max_num_iterations': i_max} 
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

num_reps = 10
random_seed = np.random.randint(0, 999999) #942179

for dim_x in [5, 25, 50]:
  problem_instance = {}
  problem_instance['dim_x'] = dim_x
  problem_instance['time_limit'] = 1*60*60 
  for N in [100, 1000]:
    obj_opt3 = []
    cert_opt3 = []
    obj_opt1 = []
    cert_opt1 = []

    for rep in range(num_reps):
      random_seed = random_seed*rep+123
      data_train = generate_data(random_seed, N, dim_x=dim_x)

      random_seed_2 = random_seed*2
      data_valid = generate_data(random_seed_2, N, dim_x=dim_x)

      random_seed_3 = random_seed*3
      data_test = generate_data(random_seed_3, N, dim_x=dim_x)

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
      
      obj_opt3.append(-best_sol['obj'])
      cert_opt3.append(best_sol['feas'][0])

      # OPTION 1: use 2 data sets, where third is only used for single final evaluation
      data_valid = np.concatenate((data_valid, data_test))
      data_test = None

      robist_opt1 = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                      data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                      verbose=False)

      (best_sol, 
        runtime_robist, 
        num_iter, 
        non_dominated_solutions, 
        S_history, 
        all_solutions_robist) = robist_opt1.run(stop_criteria=stop_criteria, return_all_solutions=True)
      
      obj_opt1.append(-best_sol['obj'])
      cert_opt1.append(best_sol['feas'][0])


    print()
    print(f'results for dim_x={dim_x} and N={N}:')
    print(f'option1: obj={round(sum(obj_opt1)/len(obj_opt1),2)} with lb={round(sum(cert_opt1)/len(cert_opt1),2)}')
    print(f'option3: obj={round(sum(obj_opt3)/len(obj_opt3),2)} with lb={round(sum(cert_opt3)/len(cert_opt3),2)}')
    










