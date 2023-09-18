"""
Comparison with cal2005 and yan2013 as size of problem (dim_x) increases
"""   
# external imports
from sklearn.model_selection import train_test_split
import time
import math
from numpy import nan

# internal imports
from tp import generate_data, solve_SCP, unc_function, eval_OoS, solve_with_yan2013
from ROBIST import ROBIST
from scen_opt import determine_cam2008_N_min

cal2005_yn = True
yan2013_yn = True
robist_yn = True

risk_param_epsilon = 0.05
conf_param_alpha = 0.01                  
num_seeds = 100

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
    
    headers = headers + ['$N$', '$N_1$', '$N_2$', '# iter.~(\\texttt{add})', '\# iter.~(\\texttt{remove})', 
                        'runtime (ROBIST)', 'obj. (ROBIST)', 'feas. certificate (ROBIST)', 'OoS prob. (ROBIST)',
                        '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']
        
    # Write headers to .txt file
    with open(r'output/ToyProblem/headers_'+output_file_name+'.txt','w+') as f:
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
            results_classic = [nan, nan, nan, nan]
            
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
            results_yan2013 = [nan, nan, nan, nan, nan, nan]
            
        output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_yan2013
        
        if robist_yn:
            stop_criteria={'max_num_iterations': 500}
            eval_unc_obj = None
            eval_unc_constr = [{'function': unc_function,
                                'info': {'risk_measure': 'probability',
                                        'desired_rhs': 1 - risk_param_epsilon}}]
            
            N_robist = 1000
            N_train = math.floor(N_robist/2)
            N_test = N_robist - N_train
            data = generate_data(random_seed, N_robist, dim_x=dim_x)               
            data_train, data_test = train_test_split(data, train_size=(N_train/N_robist), random_state=random_seed)
            
            algorithm = ROBIST(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                data_train, data_test, conf_param_alpha=conf_param_alpha,
                                use_dual_sol=True, verbose=False)
            

            (best_sol, 
              runtime_robist, 
              num_iter, 
              pareto_frontier, 
              S_history) = algorithm.run(stop_criteria=stop_criteria,
                                        store_all_solutions=False,
                                        random_seed=random_seed)
                                                   
            lb_robist = best_sol['feas'][0]
            OoS_prob_feas_robist = eval_OoS(best_sol['sol'], data_OoS)
            obj_robist = - best_sol['obj']
            S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
            S_max = max(len(S_i) for S_i in S_history)
            num_iter_add = num_iter['add']
            num_iter_remove = num_iter['remove']
            
            results_robist = [N_robist, N_train, N_test, num_iter_add, num_iter_remove,
                              runtime_robist, obj_robist, lb_robist, OoS_prob_feas_robist,
                              S_avg, S_max]
        else:
            results_robist = [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
            
        output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_robist
        
        
        # output_file_name = 'new_output_data'
        with open(r'output/ToyProblem/results_'+output_file_name+'_new.txt','w+') as f:
            f.write(str(output_data))
        
        run_count += 1
        print("Completed run: " + str(run_count))
        # print()
    

    # # Read in previous output from .txt file
    # file_path = 'output/ToyProblem/results_'+output_file_name+'_new.txt'
    # dic = ''
    # with open(file_path,'r') as f:
    #     for i in f.readlines():
    #         dic+=i
    # output_data_read = eval(dic)
    # output_data = output_data_read
    
    # Aggregate data to get avg across random seed runs
    import pandas as pd
    df_output = pd.DataFrame.from_dict(output_data, orient='index')
    print()
    print(df_output.mean())
    
    # OoS_prob_feas_cc += df_output.mean()[3]
    # OoS_prob_feas_yan += df_output.mean()[9]
    # OoS_prob_feas_yan += df_output.mean()[18]
    


    
