"""
Numerical experiments on LS problem
"""  
# external imports
from sklearn.model_selection import train_test_split
import math

# internal imports
from ls import generate_unc_param_data, get_known_param_data, solve_SCP, unc_function, eval_OoS
from ls import determine_N_vay2012, solve_SCP_vay2012
from ROBIST import ROBIST

robist_yn = True
vay2012_degree1_yn = True
vay2012_degree2_yn = False

for num_stores in [2,3,4,5]:

    risk_param_epsilon = 0.05
    conf_param_alpha = 0.05
    
    # robist info:
    N_robist = 1000
    N_train = math.floor(N_robist/2)
    N_test = N_robist - N_train
    i_max = 50
    stop_criteria={'max_num_iterations': i_max}
    eval_unc_obj = None
    eval_unc_constr = [{'function': unc_function,
                        'info': {'risk_measure': 'probability', # must be either 'probability' or 'expectation'
                                'desired_rhs': 1 - risk_param_epsilon}}]
    
    # vay2012 info:
    if vay2012_degree1_yn:
        N_d1, num_vars_d1 = determine_N_vay2012(num_stores, risk_param_epsilon, conf_param_alpha, 1)
    if vay2012_degree2_yn:
        N_d2, num_vars_d2 = determine_N_vay2012(num_stores, risk_param_epsilon, conf_param_alpha, 2)
    
    num_seeds = 10
    random_seed_settings = [i+1 for i in range(num_seeds)]
    
    output_file_name = f'ls_numstores={num_stores}_eps={risk_param_epsilon}_alpha={conf_param_alpha}_imax={i_max}_seeds=1-{num_seeds}'
    
    headers = ['num_stores', 'seed']
    if robist_yn:
        headers = headers + ['$N_1$ (robist)', '$N_2$ (robist)', '$T$ (robist)', '$Obj.$ (robist)', '$VaR^{OoS}$ (robist)',
                             '$p_{vio}^{OoS}$ (robist)', '$p_{vio_demand}^{OoS}$ (robist)',
                            '\#Iter.~(\\texttt{add})', '\#Iter.~(\\texttt{remove})', 
                            '$\mu_{|\mathcal{S}_i|}$', '$\max_{i}|\mathcal{S}_i|$']
    if vay2012_degree1_yn:
        headers = headers + ['num vars (vay2012_degree1)', '$N$ (vay2012_degree1)', '$T$ (vay2012_degree1)', 
                              '$Obj.$ (vay2012_degree1)', '$VaR^{OoS}$ (vay2012_degree1)',
                              '$p_{vio}^{OoS}$ (vay2012_degree1)', '$p_{vio_demand}^{OoS}$ (vay2012_degree1)']
    if vay2012_degree2_yn:
        headers = headers + ['num vars (vay2012_degree2)', '$N$ (vay2012_degree2)', '$T$ (vay2012_degree2)', 
                              '$Obj.$ (vay2012_degree2)', '$VaR^{OoS}$ (vay2012_degree2)',
                              '$p_{vio}^{OoS}$ (vay2012_degree2)', '$p_{vio_demand}^{OoS}$ (vay2012_degree2)']
    
    # Write headers to .txt file
    with open(r'output/LotSizing/headers_'+output_file_name+'.txt','w+') as f:
        f.write(str(headers))
    
    output_data = {}
    run_count = 0
    for random_seed in random_seed_settings:
        output_data[(num_stores, random_seed)] = []
        problem_instance = get_known_param_data(num_stores, random_seed=random_seed)
    
        # Generate extra out-of-sample (OoS) data
        random_seed_OoS = 1234
        N_OoS = int(10000)
        data_OoS = generate_unc_param_data(random_seed_OoS, N_OoS, **problem_instance)
            
        # robist
        if robist_yn:
            data = generate_unc_param_data(random_seed, N_robist, **problem_instance)               
            data_train, data_test = train_test_split(data, train_size=(N_train/N_robist), random_state=random_seed)
            
            algorithm = ROBIST(solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                                data_train, data_test, conf_param_alpha=conf_param_alpha,
                                use_dual_sol=False, verbose=False)
            
            (best_sol, 
              runtime_robist, 
              num_iter, 
              pareto_frontier, 
              S_history) = algorithm.run(stop_criteria=stop_criteria, store_all_solutions=False)
            
            obj_robist = best_sol['obj']
            OoS_p_vio_robist, OoS_VaR_robist, OoS_p_demand_robist = eval_OoS(best_sol['sol'], data_OoS, unc_function, risk_param_epsilon, **problem_instance)
            S_avg = sum(len(S_i) for S_i in S_history) / len(S_history)
            S_max = max(len(S_i) for S_i in S_history)
            
            results_robist = [N_train, (N_robist-N_train), runtime_robist, obj_robist,  OoS_VaR_robist,
                              OoS_p_vio_robist, OoS_p_demand_robist,
                              num_iter['add'], num_iter['remove'], S_avg, S_max]
            
            output_data[(num_stores, random_seed)] = output_data[(num_stores, random_seed)] + results_robist
        
            print("Completed robist")    
        
        # vay2012
        if vay2012_degree1_yn:
            degree_dr = 1
            data_d1 = generate_unc_param_data(random_seed, N_d1, **problem_instance) 
            x_d1, obj_d1, time_d1 = solve_SCP_vay2012(data_d1, degree_dr, **problem_instance)
            
            # OoS_p_vio_d1_dr = eval_p_OoS_vay2012(x_d1, data_OoS, degree_dr, **problem_instance)
            OoS_p_vio_d1, OoS_VaR_d1, OoS_p_demand_d1 = eval_OoS(x_d1, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
            
            results_d1 = [num_vars_d1, N_d1, time_d1, obj_d1, OoS_VaR_d1, OoS_p_vio_d1, OoS_p_demand_d1]
            
            output_data[(num_stores, random_seed)] = output_data[(num_stores, random_seed)] + results_d1
        
            print("Completed vay2012_degree1_yn")
        
        if vay2012_degree2_yn:
            degree_dr = 2
            data_d2 = generate_unc_param_data(random_seed, N_d2, **problem_instance) 
            x_d2, obj_d2, time_d2 = solve_SCP_vay2012(data_d2, degree_dr, **problem_instance)
            
            # OoS_p_vio_d1_dr = eval_p_OoS_vay2012(x_d1, data_OoS, degree_dr, **problem_instance)
            OoS_p_vio_d2, OoS_VaR_d2, OoS_p_demand_d2 = eval_OoS(x_d2, data_OoS, unc_function, risk_param_epsilon, **problem_instance)
            
            results_d2 = [num_vars_d2, N_d2, time_d2, obj_d2, OoS_VaR_d2, OoS_p_vio_d2, OoS_p_demand_d2]
            
            output_data[(num_stores, random_seed)] = output_data[(num_stores, random_seed)] + results_d2
            
            print("Completed vay2012_degree2_yn")
        
        with open(r'output/LotSizing/results_'+output_file_name+'_new.txt','w+') as f:
            f.write(str(output_data))
    
        run_count += 1
        print("Completed run: " + str(run_count))
        print()
        
    # # Read in previous output from .txt file
    # file_path = 'output/LotSizing/results_'+output_file_name+'_new.txt'
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