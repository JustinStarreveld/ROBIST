"""
Numerical experiments on WDP
"""  
# external imports
from sklearn.model_selection import train_test_split
import time
import math
from numpy import nan

# internal imports
from wdp import generate_unc_param_data, get_fixed_param_data, solve_SCP, unc_function, unc_constraint, eval_x_OoS
from ROBIST import ROBIST
import scen_opt_methods as scen_opt

cal2005_yn = run_cal2005_yn = True
car2014_yn = run_car2014_yn = True
cal2016_yn = run_cal2016_yn = True
gar2022_yn = run_gar2022_yn = True
robist_yn = run_robist_yn = True

for scale_dim_problem in [1,2,3]:

    print("-------------------------------------------------------------------")
    print("scale=", scale_dim_problem)    
    print("-------------------------------------------------------------------")

    dim_x = 5*scale_dim_problem * 10*scale_dim_problem + 1
    conf_param_alpha = 1e-9
    risk_param_epsilon = 0.01
    
    generate_data_kwargs = {'scale_dim_problem': scale_dim_problem}
    
    # Generate extra out-of-sample (OoS) data
    random_seed_OoS = 1234
    N_OoS = int(1e6)
    data_OoS = generate_unc_param_data(random_seed_OoS, N_OoS, scale_dim_problem=scale_dim_problem)
    
    if cal2005_yn:
        N_cam2008 = scen_opt.determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    
    if car2014_yn:
        N_1_car2014, N_2_car2014 = scen_opt.determine_N_car2014(dim_x, risk_param_epsilon, conf_param_alpha)
    
    if cal2016_yn:
        time_determine_N_cal2016, N_cal2016 = scen_opt.determine_N_cal2016(dim_x, risk_param_epsilon)
        time_determine_N_eval_cal2016, N_eval_cal2016 = scen_opt.determine_N_oracle_cal2016(dim_x, risk_param_epsilon, conf_param_alpha, N_cal2016)
        
    if gar2022_yn:
        time_determine_N_gar2022, set_sizes_gar2022 = scen_opt.gar2022_determine_set_sizes(dim_x, risk_param_epsilon, conf_param_alpha)
    
    if robist_yn:
        N_robist = 3000
        i_max = 200
        stop_criteria={'max_num_iterations': i_max}
        eval_unc_obj = {'function': unc_function,
                        'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                                  'desired_rhs': 1 - risk_param_epsilon}}
    
        eval_unc_constr = None
    
    num_seeds = 10
    random_seed_settings = [i+1 for i in range(num_seeds)]
    
    output_file_name = f'wdp_scale={scale_dim_problem}_eps={risk_param_epsilon}_alpha={conf_param_alpha}'
    headers = ['$dim(\mathbf{x})$', 'seed']
    
    if cal2005_yn:
        output_file_name = output_file_name + '_cal2005'
        headers = headers + ['$N$ (cal2005)', '$T$ (cal2005)', '$Obj.$ (cal2005)', '$p_{vio}^{OoS}$ (cal2005)', '$VaR^{OoS}$ (cal2005)']
    if car2014_yn:
        output_file_name = output_file_name + '_car2014'
        headers = headers + ['$N_1$ (car2014)', '$N_2$ (car2014)', '$T$ (car2014)', '$Obj.$ (car2014)', '$p_{vio}^{OoS}$ (car2014)', '$VaR^{OoS}$ (car2014)']
    if cal2016_yn:
        output_file_name = output_file_name + '_cal2016'
        headers = headers + ['N_cal2016', 'N_eval_cal2016', 'total_train_data_used_cal2016', 'total_test_data_used_cal2016',
                              'runtime_cal2016', 'obj_cal2016', 'p_vio_cal2016', 'VaR_cal2016', 'iter_cal2016']
    if gar2022_yn:
        output_file_name = output_file_name + '_gar2022'
        headers = headers + ['N_gar2022', 'runtime_gar2022', 'obj_gar2022', 'p_vio_gar2022', 'VaR_gar2022']
    if robist_yn:
        output_file_name = output_file_name + '_robist'
        headers = headers + ['$N_1$ (robist)', '$N_2$ (robist)', '$T$ (robist)', '$Obj.$ (robist)', '$p_{vio}^{OoS}$ (robist)', '$VaR^{OoS}$ (robist)',
                            '\#Iter.~(\\texttt{add})', '\#Iter.~(\\texttt{remove})', 
                            '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']
    
    output_file_name = output_file_name + f'_seeds=1-{num_seeds}'
    
    # Write headers to .txt file
    with open(r'output/WeightedDistribution/headers_'+output_file_name+'.txt','w+') as f:
        f.write(str(headers))
    
    output_data = {}
    
    run_count = 0
    for random_seed in random_seed_settings:
        output_data[(dim_x, random_seed)] = []
        
        problem_instance = get_fixed_param_data(random_seed, scale_dim_problem=scale_dim_problem)
        problem_instance['time_limit'] = 1*60*60 
        
        # classic approach:
        if cal2005_yn:
            if run_cal2005_yn:
                data = generate_unc_param_data(random_seed, N_cam2008, scale_dim_problem=scale_dim_problem)
                start_time = time.time()
                x, obj = solve_SCP(data, **problem_instance)
                runtime_cal2005 = time.time() - start_time
                obj_cal2005 = - obj
                p_vio_cal2005, VaR_cal2005 = eval_x_OoS(x, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
                results_cal2005 = [N_cam2008, runtime_cal2005, obj_cal2005, p_vio_cal2005, VaR_cal2005]
            else:
                results_cal2005 = [nan, nan, nan, nan, nan]
                
            output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_cal2005
            
            print("Finished cal2005 in", runtime_cal2005, "seconds")
            if runtime_cal2005 > problem_instance['time_limit']:
                run_cal2005_yn = False
    
        # FAST approach
        if car2014_yn:
            if run_car2014_yn:
                (x, 
                  obj, 
                  runtime) = scen_opt.solve_with_car2014(dim_x, risk_param_epsilon, conf_param_alpha, 
                                                        solve_SCP, unc_function, problem_instance, 
                                                        generate_unc_param_data, generate_data_kwargs,
                                                        N_1=N_1_car2014, N_2=N_2_car2014, 
                                                        random_seed=random_seed)
                runtime_car2014 = runtime 
                obj_car2014 = - obj
                p_vio_car2014, VaR_car2014 = eval_x_OoS(x, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
                results_car2014 = [N_1_car2014, N_2_car2014, runtime_car2014, obj_car2014, p_vio_car2014, VaR_car2014]
            else:
                results_car2014 = [nan, nan, nan, nan, nan, nan]
                
            output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_car2014
            
            print("Finished car2014 in", runtime_car2014, "seconds")
            if runtime_car2014 > problem_instance['time_limit']:
                run_car2014_yn = False
            
        # Calafiore2016
        if cal2016_yn:
            if run_cal2016_yn:
                (x_cal2016, 
                  obj, 
                  iter_k, 
                  total_train_data_used_cal2016, 
                  total_test_data_used_cal2016, 
                  runtime_cal2016) = scen_opt.solve_with_cal2016(N_cal2016, N_eval_cal2016, dim_x, risk_param_epsilon, conf_param_alpha, 
                                                                  solve_SCP, unc_constraint, problem_instance, 
                                                                  generate_unc_param_data, generate_data_kwargs, random_seed, 
                                                                  numeric_precision=1e-6, verbose=False)
                obj_cal2016 = - obj
                p_vio_cal2016, VaR_cal2016 = eval_x_OoS(x_cal2016, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
                results_cal2016 = [N_cal2016, N_eval_cal2016, total_train_data_used_cal2016, total_test_data_used_cal2016,
                                                    runtime_cal2016, obj_cal2016, p_vio_cal2016, VaR_cal2016, iter_k]
            else:
                results_cal2016 = [nan, nan, nan, nan, nan, nan, nan, nan, nan]
                
            output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_cal2016
            
            print("Finished cal2016 in", runtime_cal2016, "seconds")
            if runtime_cal2016 > problem_instance['time_limit']:
                run_cal2016_yn = False
        
        # gar2022
        if gar2022_yn:
            if run_gar2022_yn:
                (x_gar2022, 
                  obj, 
                  j_gar2022, 
                  s_j_gar2022, 
                  set_sizes_gar2022, 
                  time_main_solves_gar2022, 
                  time_determine_supp_gar2022) = scen_opt.solve_with_gar2022(dim_x, set_sizes_gar2022, solve_SCP, unc_constraint, 
                                                                              generate_unc_param_data, random_seed, problem_instance, 
                                                                              scale_dim_problem=scale_dim_problem)
                
                N_gar2022 = set_sizes_gar2022[s_j_gar2022]
                runtime_gar2022 = time_main_solves_gar2022 + time_determine_supp_gar2022
                obj_gar2022 = - obj
                p_vio_gar2022, VaR_gar2022 = eval_x_OoS(x_gar2022, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
                results_gar2022 = [N_gar2022, runtime_gar2022, obj_gar2022, p_vio_gar2022, VaR_gar2022]
            else:
                results_gar2022 = [nan, nan, nan, nan, nan]
                
            output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_gar2022
        
            print("Finished gar2022 in", runtime_gar2022, "seconds")
            if runtime_gar2022 > problem_instance['time_limit']:
                gar2022_yn = False
        
        # ROBIST
        if robist_yn:
            if run_robist_yn:
                data = generate_unc_param_data(random_seed, N_robist, scale_dim_problem=scale_dim_problem)
                N_train = math.floor(N_robist / 2)
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
                
                obj_robist = - best_sol['obj']
                p_vio_robist, VaR_robist = eval_x_OoS(best_sol['sol'], data_OoS, unc_function, risk_param_epsilon, **problem_instance)
                S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
                S_max = max(len(S_i) for S_i in S_history)
            
                results_robist = [N_train, (N_robist-N_train), runtime_robist, obj_robist, p_vio_robist, VaR_robist,
                                  num_iter['add'], num_iter['remove'], S_avg, S_max]
            else:
                results_robist = [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
            
            output_data[(dim_x, random_seed)] = output_data[(dim_x, random_seed)] + results_robist
            
            print("Finished robist in", runtime_robist, "seconds")
    
        with open(r'output/WeightedDistribution/results_'+output_file_name+'_new.txt','w+') as f:
            f.write(str(output_data))
    
        run_count += 1
        print("Completed run: " + str(run_count))
        print()
        
        
    # # Read in previous output from .txt file
    # file_path = 'output/WeightedDistribution/results_'+output_file_name+'_new.txt'
    # dic = ''
    # with open(file_path,'r') as f:
    #     for i in f.readlines():
    #         dic+=i
    # output_data_read = eval(dic)
    # output_data = output_data_read 
        
        
    # Aggregate data to get avg across random seed runs
    import pandas as pd
    df_output = pd.DataFrame.from_dict(output_data, orient='index')
    print(df_output.mean())
    
    
    
    
    
    
    
    





