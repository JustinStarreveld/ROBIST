"""
In this script we compare two options for ROBIST when applied to the 
toy problem of dimension 2
"""

# external imports
import numpy as np
from pyprojroot import here

# internal imports
from tp import generate_data, solve_SCP, unc_function

import sys
mypath = "C:/Users/JSTARRE/OneDrive - UvA/Research/ROBIST/Code/ROBIST/"
sys.path.insert(0, mypath)
from src.robist.robist import Robist

# set parameter values
risk_param_epsilon = 0.10
conf_param_alpha = 0.01
dim_x = 2
N = 1000
numeric_precision=1e-6

problem_instance = {}
problem_instance['dim_x'] = dim_x
problem_instance['time_limit'] = 1*60*60 

random_seed = np.random.randint(0, 999999) #942179
data_train = generate_data(random_seed, N, dim_x=dim_x)

random_seed_2 = random_seed*2
data_valid = generate_data(random_seed_2, N, dim_x=dim_x)

random_seed_3 = random_seed*3
data_test = generate_data(random_seed_3, N, dim_x=dim_x)



i_max = 100
stop_criteria={'max_num_iterations': i_max} 
eval_unc_obj = None
eval_unc_constr = [{'function': unc_function,
                    'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                            'desired_rhs': 1 - risk_param_epsilon}}]

robist = Robist(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                data_train, data_valid, data_test, conf_param_alpha=conf_param_alpha,
                verbose=False)

(best_sol, 
  runtime_robist, 
  num_iter, 
  non_dominated_solutions, 
  S_history, 
  all_solutions_robist) = robist.run(stop_criteria=stop_criteria, return_all_solutions=True)

...



