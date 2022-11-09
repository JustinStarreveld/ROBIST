# Import packages
import numpy as np
import cvxpy as cp
import scipy.stats
import time
import math
from k_means_constrained import KMeansConstrained

def gen_and_eval_alg_con(solve_P, unc_func, problem_info,
                        data_train, data_test, conf_param_alpha, 
                        bound_settings, phi_div, phi_dot,
                        stopping_cond, stop_info, compute_prob_add, 
                        add_strategy, remove_strategy, clean_strategy=None, 
                        use_tabu=False,
                        numeric_precision=1e-6, random_seed=0, 
                        data_eval=None, emp_eval=None, 
                        analytic_eval=None,
                        verbose=False, store_all_sol=False):

    # Extract info
    k = data_train.shape[1]
       
    # Store info
    num_iter = {'add':0, 'remove':0, 'clean':0}
    feas_solutions_str = set()
    feas_solutions = []
    all_solutions = []
    best_sol = {'sol': None}
    pareto_solutions = [] # concept of pareto solutions does not extend to uncertainty in obj
    S_past = []
    count_duplicate_S = 0
    prev_x = None
    prev_obj = None
    
    # Initialize algorithm
    start_time = time.time()
    S_val = np.array([data_train[0]]) # Assume first index contains nominal data
    S_ind = [0] # Tracks the indices of the scenarios in Z
    np.random.seed(random_seed) # Set seed for random strategies
    if problem_info['risk_measure'] == 'expectation':
        bound_settings['method'] = 'equal_size' # set as default method (may be changed in certain situations later)
    
    count_iter = 0
    while True:
        count_iter += 1
        
        # check if we have already found and evaluated this sample of scenarios
        copy_S = sorted(S_ind.copy())
        duplicate_sample = [copy_S == S for S in S_past]
        if any(duplicate_sample):
            count_duplicate_S += 1
            i = [i for i,x in enumerate(duplicate_sample) if x == True][0]
            S_ind = S_past[i]
            S_val = data_train[S_ind]
            for sol_info in all_solutions:
                if sol_info['scenario_set'] == sorted(S_ind):
                    x = sol_info['sol']
                    obj = sol_info['obj']
                    break
        else:
            [x, obj] = solve_P(k, S_val, problem_info)
            S_past.append(copy_S)
        
        if x is None and prev_x is not None: # something went wrong in solving P, revert back to previous solution
            x = prev_x
            obj = prev_obj
        else:
            prev_x = x
            prev_obj = obj
        
        bound_train = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                    data_train, x, unc_func, bound_settings, numeric_precision)
        
        bound_test = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                   data_test, x, unc_func, bound_settings, numeric_precision)
        
        if verbose:
            print("-----------------")
            print("iter     : " + f'{round(count_iter,0):.0f}')
            print("size_S   : " + f'{round(len(S_val),0):.0f}')
            print("obj_S    : " + f'{round(obj,3):.3f}')
            if emp_eval is not None:
                if data_eval is not None:
                    eval_true_obj = emp_eval(x, data_eval, problem_info)
                    print("obj_eval : " + f'{round(eval_true_obj,3):.3f}')
                eval_train_obj = emp_eval(x, data_train, problem_info)
                eval_test_obj = emp_eval(x, data_test, problem_info)
                print("obj_train: " + f'{round(eval_train_obj,3):.3f}')
                print("obj_test : " + f'{round(eval_test_obj,3):.3f}')
            if analytic_eval is not None:
                anal_eval = analytic_eval(x, problem_info)
                print("anal_eval: " + f'{round(anal_eval,3):.3f}')
            if data_eval is not None:
                if problem_info['risk_measure'] == 'expectation':
                    eval_p = (1/len(data_eval))*np.sum(unc_func(data_eval, x, problem_info))
                else:
                    eval_p =  sum(unc_func(data_eval, x, problem_info)>(0+numeric_precision))/len(data_eval)
                print('p_eval   : ' + f'{round(eval_p,3):.3f}')
            print("b_train  : " + f'{round(bound_train,3):.3f}')
            print("b_test   : " + f'{round(bound_test,3):.3f}')
            
        
        current_sol = {'sol': x, 'obj': obj, 'time': (time.time()-start_time), 
                    'bound_train': bound_train,  'bound_test': bound_test, 
                    'scenario_set': copy_S}
        
        if store_all_sol:           
            all_solutions.append(current_sol)
        
        if x.tostring() not in feas_solutions_str:
            # check if worthwhile to see if we can tighten bound (before evaluating feasibility)
            if problem_info['risk_measure'] == 'expectation':
                potential_bound_shift = 0.10*abs(bound_test)
                if bound_test - potential_bound_shift <= (0+numeric_precision):
                    if len(data_test) < 1000: # takes too long if many data points
                        bound_settings['method'] = 'kmeans'
                        bound_test_2 = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                                     data_test, x, unc_func, bound_settings, numeric_precision)
                        bound_settings['method'] = 'equal_size'
                        if bound_test_2 < bound_test:
                            bound_test = bound_test_2
                            current_sol['bound_test'] = bound_test
            
            # even if infeasible, might still be better than current "best"
            if best_sol['sol'] is None or (best_sol['bound_test']>0 and bound_test < best_sol['bound_test']):
                best_sol = current_sol
                
            # check if feasible solution
            if bound_test <= (0+numeric_precision):
                feas_solutions_str.add(x.tostring())
                feas_solutions.append(current_sol)
                
                if obj > best_sol['obj']:
                    best_sol = current_sol
                    
            # update list of Pareto efficient solutions
            if len(pareto_solutions) == 0:
                pareto_solutions.append((bound_test, obj))
            else:
                pareto_opt = True
                to_remove = []
                for i, (lb2, obj2) in enumerate(pareto_solutions):
                    if bound_test < lb2 and obj >= obj2:
                        to_remove.append(i)
                    elif not bound_test < lb2 and obj <= obj2:
                        pareto_opt = False
                for index in sorted(to_remove, reverse=True):
                    del pareto_solutions[index]
                if pareto_opt:
                    pareto_solutions.append((bound_test, obj))

        elapsed_time = (time.time()-start_time)
        num_solutions = len(feas_solutions)
        if stopping_cond(stop_info, elapsed_time=elapsed_time, num_solutions=num_solutions, num_iterations=count_iter):
            break
        
        if use_tabu == True: 
            tabu_add = get_tabu_add(S_ind, S_past)
            tabu_remove = get_tabu_remove(S_ind, S_past)
        else:
            tabu_add = set()
            if len(S_ind) > 0:
                tabu_add.add([i for i in S_ind][0]) # Not allowed to add scenarios that are already in current S
            tabu_remove = set()
        
        # Now we determine whether to add or remove scenarios using only the training data        
        lhs_constr = bound_train
        constr_train = unc_func(data_train, x, problem_info)
        constr_add, num_possible_additions = get_possible_additions(constr_train, tabu_add, numeric_precision)
        S_ind_rem, num_possible_removals = get_possible_removals(S_ind, tabu_remove)
        
        add_or_remove = determine_action(lhs_constr, compute_prob_add, 
                                         num_possible_additions, num_possible_removals)
        
        if add_or_remove is None:
            break
        elif add_or_remove == True:
            S_val, S_ind = add_scenarios(add_strategy, data_train, S_val, S_ind, 
                                         constr_train, constr_add, bound_train, numeric_precision) 
            num_iter['add'] += 1
        elif add_or_remove == False:
            S_val_rem = np.array([S_val[i] for i,e in enumerate(S_ind) if e in S_ind_rem])
            constr_S = unc_func(S_val_rem, x, problem_info) - obj
            S_val, S_ind = remove_scenarios(remove_strategy, S_val, S_ind, S_val_rem,
                                            constr_S, numeric_precision)
            num_iter['remove'] += 1
        
        if clean_strategy is not None and len(S_val) >= clean_strategy[0]: # Invoke clean strategy (to improve solve efficiency)
            S_val_rem = np.array([S_val[i] for i,e in enumerate(S_ind) if e in S_ind_rem])
            constr = unc_func(S_val_rem, x, problem_info) - obj
            S_val, S_ind = remove_scenarios(remove_strategy, S_val, S_ind, S_val_rem,
                                            constr, numeric_precision)
            num_iter['clean'] += 1  
    
    runtime = time.time() - start_time
    if store_all_sol:
        return runtime, num_iter, all_solutions, best_sol, pareto_solutions
    return runtime, num_iter, feas_solutions, best_sol, pareto_solutions

def gen_and_eval_alg_obj(solve_P, unc_func, problem_info,
                        data_train, data_test, conf_param_alpha, 
                        bound_settings, phi_div, phi_dot,
                        stopping_cond, stop_info, compute_prob_add, 
                        add_strategy, remove_strategy, clean_strategy=None, 
                        use_tabu=False,
                        numeric_precision=1e-6, random_seed=0, 
                        data_eval=None, emp_eval=None, 
                        analytic_eval=None,
                        verbose=False):

    # Extract info
    k = data_train.shape[1]
       
    # Store info
    num_iter = {'add':0, 'remove':0, 'clean':0}
    feas_solutions = set()
    feas_solution_info = []
    all_solutions = []
    best_sol = {'sol': None}
    pareto_solutions = None # concept of pareto solutions does not extend to uncertainty in obj
    S_past = []
    count_duplicate_S = 0
    prev_x = None
    prev_obj = None
    
    # Initialize algorithm
    start_time = time.time()
    S_val = np.array([data_train[0]]) # Assume first index contains nominal data
    S_ind = [0] # Tracks the indices of the scenarios in Z
    np.random.seed(random_seed) # Set seed for random strategies
    bound_settings['method'] = 'equal_size' # set as default method (may be changed in certain situations later)
    
    count_iter = 0
    while True:
        count_iter += 1
        # check if we have already found and evaluated this sample of scenarios
        duplicate_sample = [sorted(S_ind) == x for x in S_past]
        if any(duplicate_sample):
            count_duplicate_S += 1
            i = [i for i,x in enumerate(duplicate_sample) if x == True][0]
            S_ind = S_past[i]
            S_val = data_train[S_ind]
            for sol_info in all_solutions:
                if sol_info['scenario_set'] == sorted(S_ind):
                    x = sol_info['sol']
                    obj = sol_info['obj']
                    break
        else:
            # solve_start_time = time.time()
            [x, obj] = solve_P(k, S_val, problem_info)
            # solve_time = time.time() - solve_start_time
            
            all_solutions.append({'sol': x, 'obj': obj, 'scenario_set': sorted(S_ind.copy())})
            S_past.append(sorted(S_ind.copy()))
        
        
        if x is None and prev_x is not None: # something went wrong in solving P, revert back to previous solution
            x = prev_x
            obj = prev_obj
        else:
            prev_x = x
            prev_obj = obj
        
        bound_train = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                    data_train, x, unc_func, bound_settings, numeric_precision)
        
        bound_test = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                   data_test, x, unc_func, bound_settings, numeric_precision)
        
        if verbose:
            print("-----------------")
            print("iter     : " + f'{round(count_iter,0):.0f}')
            print("size_S   : " + f'{round(len(S_val),0):.0f}')
            print("obj_S    : " + f'{round(obj,3):.3f}')
            if emp_eval is not None:
                if data_eval is not None:
                    eval_true_obj = emp_eval(x, data_eval, problem_info)
                    print("obj_eval : " + f'{round(eval_true_obj,3):.3f}')
                eval_train_obj = emp_eval(x, data_train, problem_info)
                eval_test_obj = emp_eval(x, data_test, problem_info)
                print("obj_train: " + f'{round(eval_train_obj,3):.3f}')
                print("obj_test : " + f'{round(eval_test_obj,3):.3f}')
                
            if analytic_eval is not None:
                eval_true_obj = analytic_eval(x, problem_info)
                print("obj_anal : " + f'{round(eval_true_obj,3):.3f}')
            
            if data_eval is not None:
                eval_true_exp = (1/len(data_eval))*np.sum(unc_func(data_eval, x, problem_info))
                print('p_eval   : ' + f'{round(eval_true_exp,3):.3f}')
                
            print("b_train  : " + f'{round(bound_train,3):.3f}')
            print("b_test   : " + f'{round(bound_test,3):.3f}')
                   
        if x.tostring() not in feas_solutions:
            feas_solutions.add(x.tostring())
            sol_info = {'sol': x, 'obj': obj, 'time': (time.time()-start_time), 
                        'bound_train': bound_train,  'bound_test': bound_test, 'scenario_set': sorted(S_ind.copy())}
            feas_solution_info.append(sol_info)
            
            # Determine if best solution can be replaced
            potential_bound_shift = 0.20*abs(bound_test) #assume that we might be able to lower the upper bound by 20%
            if best_sol['sol'] is None or bound_test - potential_bound_shift < best_sol['bound_test']:
                if len(data_test) < 1000: # evaluate using more sophisticated kmeans bound method
                    bound_settings['method'] = 'kmeans'
                    bound_test_2 = compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                                                 data_test, x, unc_func, bound_settings, numeric_precision)
                    bound_settings['method'] = 'equal_size'
                    if bound_test_2 < bound_test:
                        bound_test = bound_test_2
                        sol_info['bound_test'] = bound_test
                
                if best_sol['sol'] is None or bound_test < best_sol['bound_test']:
                    best_sol = sol_info
        
        elapsed_time = (time.time()-start_time)
        num_solutions = len(feas_solution_info)
        if stopping_cond(stop_info, elapsed_time, num_solutions):
            break
        
        if use_tabu == True: 
            tabu_add = get_tabu_add(S_ind, S_past)
            tabu_remove = get_tabu_remove(S_ind, S_past)
        else:
            tabu_add = set()
            tabu_add.add([i for i in S_ind][0]) # Not allowed to add scenarios that are already in current S
            tabu_remove = set()
        
        # Now we determine whether to add or remove scenarios using only the training data        
        lhs_constr = bound_train - obj #
        constr_train = unc_func(data_train, x, problem_info) - obj
        constr_add, num_possible_additions = get_possible_additions(constr_train, tabu_add, numeric_precision)
        S_ind_rem, num_possible_removals = get_possible_removals(S_ind, tabu_remove)
        
        add_or_remove = determine_action(lhs_constr, compute_prob_add, 
                                         num_possible_additions, num_possible_removals)
        
        if add_or_remove == True:
            S_val, S_ind = add_scenarios(add_strategy, data_train, S_val, S_ind, 
                                         constr_train, constr_add, bound_train, numeric_precision) 
            num_iter['add'] += 1
        elif add_or_remove == False:
            S_val_rem = np.array([S_val[i] for i,e in enumerate(S_ind) if e in S_ind_rem])
            constr_S = unc_func(S_val_rem, x, problem_info) - obj
            S_val, S_ind = remove_scenarios(remove_strategy, S_val, S_ind, S_val_rem,
                                            constr_S, numeric_precision)
            num_iter['remove'] += 1
        else:
            break # Finished
        
        if clean_strategy is not None and len(S_val) >= clean_strategy[0]: # Invoke clean strategy (to improve solve efficiency)
            S_val_rem = np.array([S_val[i] for i,e in enumerate(S_ind) if e in S_ind_rem])
            constr = unc_func(S_val_rem, x, problem_info) - obj
            S_val, S_ind = remove_scenarios(remove_strategy, S_val, S_ind, S_val_rem,
                                            constr, numeric_precision)
            num_iter['clean'] += 1  
    
    runtime = time.time() - start_time
    return runtime, num_iter, feas_solution_info, best_sol, pareto_solutions

def compute_bound(problem_info, conf_param_alpha, phi_div, phi_dot, 
                  data, x, unc_func, bound_settings, numeric_precision):
    
    if problem_info['risk_measure'] == 'probability':
        return compute_bound_prob(problem_info, conf_param_alpha, phi_div, phi_dot, numeric_precision, data, x, unc_func)
    elif problem_info['risk_measure'] == 'expectation':
        return compute_bound_exp(problem_info, conf_param_alpha, phi_div, phi_dot, numeric_precision, data, x, unc_func, bound_settings)
    else:
        print("ERROR: do not recognize risk measure")
        return None

def compute_bound_prob(problem_info, alpha, phi_div, phi_dot, numeric_precision, data, x, unc_func):
    N = len(data)
    beta = problem_info.get('desired_prob_guarantee_beta', 0)
    constr = unc_func(data, x, problem_info)
    num_vio = sum(constr>(0+numeric_precision))
    p_vio = num_vio/N
    p = np.array([1-p_vio, p_vio])
    return compute_cc_lb(p, N, alpha, beta, phi_div, phi_dot)

def compute_cc_lb(p, N, alpha, beta, phi_div, phi_dot):
    if p[0] == 1:
        return beta - 1
    else:
        deg_of_freedom = 1
        r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, deg_of_freedom)
        q = cp.Variable(2, nonneg = True)
        constraints = [cp.sum(q) == 1]
        constraints = phi_div(p, q, r, None, constraints)
        obj = cp.Minimize(q[0])
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK)
        lb = prob.value
        return beta - lb

def compute_cc_lb_chi2_analytic(p, phi_dot, N, alpha, phi_div):
    import math
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, 1)
    q_l = p - math.sqrt(-r * (p)**2 + r*p)
    #q_u = p + math.sqrt(-r * (p)**2 + r*p)
    return q_l
    
def compute_bound_exp(problem_info, alpha, phi_div, phi_dot, numeric_precision, data, x, unc_func, bound_settings):
    min_num_obs_per_bin = bound_settings['min_num_obs_per_bin']
    num_bins_range = bound_settings['num_bins_range']
    
    # Evaluate the uncertain function f(u,x) on data
    f_evals = unc_func(data, x, problem_info)
    sorted_f_evals = np.sort(f_evals) 
    
    if bound_settings['method'] == 'equal_size':
        compute_bound = compute_exp_bound_ub_equalsize
    elif bound_settings['method'] == 'kmeans':
        compute_bound = compute_exp_bound_ub_kmeans
        sorted_f_evals = sorted_f_evals.reshape(-1,1)
    else:
        print("Error: unknown method specified in bound_settings") 
        return None
    
    # Golden-ratio search:
    gr = (math.sqrt(5) + 1) / 2
    a, b = num_bins_range
    f_a = compute_bound(alpha, phi_div, phi_dot, numeric_precision, a,
                        min_num_obs_per_bin, sorted_f_evals)
    f_b = compute_bound(alpha, phi_div, phi_dot, numeric_precision, b,
                        min_num_obs_per_bin, sorted_f_evals)
    c = math.floor(b - (b - a) / gr)
    if c == a:
        c += 1
    d = math.ceil(a + (b - a) / gr)
    if d == b:
        d -= 1
    f_c = compute_bound(alpha, phi_div, phi_dot, numeric_precision, c,
                        min_num_obs_per_bin, sorted_f_evals)
    f_d = compute_bound(alpha, phi_div, phi_dot, numeric_precision, d,
                        min_num_obs_per_bin, sorted_f_evals)
    while True:
        if abs(a-b) == 1:
            if f_a < f_b:
                return f_a
            else:
                return f_b
        if f_c < f_d:  # f(c) > f(d) to find the maximum
            b = d
            f_b = f_d
        else:
            a = c
            f_a = f_c
        c = math.floor(b - (b - a) / gr)
        if c == a:
            c += 1
        d = math.ceil(a + (b - a) / gr)
        if d == b:
            d -= 1
        f_c = compute_bound(alpha, phi_div, phi_dot, numeric_precision, c,
                            min_num_obs_per_bin, sorted_f_evals)
        f_d = compute_bound(alpha, phi_div, phi_dot, numeric_precision, d,
                            min_num_obs_per_bin, sorted_f_evals)

def compute_exp_bound_ub_equalsize(alpha, phi_div, phi_dot, numeric_precision, num_bins,
                                min_num_obs_per_bin, sorted_f_evals):
    N = len(sorted_f_evals)
    m = num_bins
    deg_of_freedom = m-1
    bins = np.array_split(sorted_f_evals, m)
    bin_thresholds = np.array([b[-1] for b in bins])
    p = np.array([(b.size/N) for b in bins])
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, deg_of_freedom)
    q = cp.Variable(m, nonneg = True)
    constraints = [cp.sum(q) == 1]
    constraints = phi_div(p, q, r, None, constraints)
    obj_sum = 0
    for i in range(m):
        obj_sum = obj_sum + q[i] * bin_thresholds[i]
    obj = cp.Maximize(obj_sum)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)
    return prob.value

def compute_exp_bound_ub_kmeans(alpha, phi_div, phi_dot, numeric_precision, num_bins,
                                min_num_obs_per_bin, sorted_f_evals):
    m = num_bins
    N = len(sorted_f_evals)
    clf = KMeansConstrained(n_clusters=m, size_min=min_num_obs_per_bin, random_state=0)
    clf.fit_predict(sorted_f_evals)
    deg_of_freedom = m - 1
    p = np.array([(sum(clf.labels_ == b)/N) for b in range(m)])
    bin_thresholds = np.array([max(sorted_f_evals[clf.labels_ == b]) for b in range(m)])
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, deg_of_freedom)
    q = cp.Variable(m, nonneg = True)
    constraints = [cp.sum(q) == 1]
    constraints = phi_div(p, q, r, None, constraints)
    obj_sum = 0
    for i in range(m):
        obj_sum = obj_sum + q[i] * bin_thresholds[i]
    obj = cp.Maximize(obj_sum)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)
    return prob.value  

def get_possible_additions(constr, tabu_add, numeric_precision):
    constr_add = constr.copy()
    if len(tabu_add) > 0:
        constr_add = np.delete(constr_add, list(tabu_add))
    return constr_add, sum(constr_add>(0+numeric_precision))

def get_possible_removals(S_ind, tabu_remove):
    S_ind_rem = S_ind.copy()
    for i in tabu_remove:
        S_ind_rem.remove(i)
    return S_ind_rem, len(S_ind_rem)

def determine_action(lhs_constr, compute_prob_add, num_possible_additions, 
                     num_possible_removals):    
    # Determines whether it will be an add (True) or remove (False) or break (None) 
    if num_possible_additions == 0 and num_possible_removals == 0:
        return None
    elif num_possible_additions == 0:
        return False
    elif num_possible_removals == 0:
        return True
    
    threshold = compute_prob_add(lhs_constr)
    # print("Prob. Add: " + f'{round(threshold,2):.2f}')
    draw = np.random.uniform()
    if draw < threshold:
        return True
    else:
        return False

def add_scenarios(add_strategy, data, S_val, S_ind, constr, constr_add, lhs_constr, numeric_precision):
    vio = constr_add[constr_add>(0+numeric_precision)]
    ind = pick_scenarios_to_add(add_strategy, len(data), constr, vio, lhs_constr, numeric_precision)
    S_ind.append(ind)
    scen_to_add = np.array([data[ind]])
    if len(S_val) > 0:
        S_val = np.append(S_val, scen_to_add, axis = 0)
    else:
        S_val = scen_to_add
    return S_val, S_ind

def pick_scenarios_to_add(add_strategy, N, constr, vio, lhs_constr, numeric_precision):
    if add_strategy == 'smallest_vio':   # the least violated scenario is added   
        return np.where(constr == np.min(vio))[0][0]
    elif add_strategy == 'random_vio':
        rand_vio = np.random.choice(vio)
        return np.where(constr == rand_vio)[0][0]
    elif add_strategy == 'N*(beta-lb)_smallest_vio':   # the N*(beta-lb)-th scenario is added
        rank = np.ceil(N*(-lhs_constr)).astype(int)
        if rank > len(vio):
            return np.where(constr == np.max(vio))[0][0]
        vio_sort = np.sort(vio) 
        vio_value = vio_sort[rank-1]     # -1 to correct for python indexing
        return np.where(constr == vio_value)[0][0]
    elif add_strategy == 'random_weighted_vio':
        vio_min = np.min(vio)
        vio_max = np.max(vio)
        vio_ideal = (-lhs_constr) * (vio_max - vio_min)
        weights = [(1 / (abs(vio_ideal - i))) for i in vio]
        sum_weights = sum(weights)
        probs = [i/sum_weights for i in weights]
        ind = np.random.choice(a = len(vio), p = probs)  
        vio_chosen = vio[ind]
        return np.where(constr == vio_chosen)[0][0]
    else:
        print("Error: did not provide valid addition strategy")
        return None

def remove_scenarios(remove_strategy, S_val, S_ind, S_val_rem, constr, numeric_precision):

    if remove_strategy == 'all_inactive':
        ind = np.where(constr < (0-numeric_precision))[0]
    elif remove_strategy == 'random_inactive':
        inactive = np.where(constr < (0-numeric_precision))[0]
        if len(inactive) > 0:
            ind = np.random.choice(inactive)
        else:
            ind = None
    elif remove_strategy == 'random_active':
        active = np.where(constr > (0-numeric_precision))[0]
        if len(active) > 0:
            ind = np.random.choice(active)
        else:
            ind = None
    elif remove_strategy == 'random_any':
        ind = np.random.choice(len(S_val_rem))
    else:
        print("Error: did not provide valid removal strategy")
    
    if ind is None:
        return S_val, S_ind
    elif isinstance(ind, np.ndarray):
        ind_set = set(ind.flatten())
        vals_to_delete = [S_val_rem[i] for i in ind_set]
        S_ind = [e for i,e in enumerate(S_ind) if not (np.any(np.all(S_val[i] == vals_to_delete, axis=1)))] 
        S_val = np.array([e for i,e in enumerate(S_val) if not (np.any(np.all(e == vals_to_delete, axis=1)))])
    elif isinstance(ind, int):
        val_to_delete = S_val_rem[ind]
        S_ind = [e for i,e in enumerate(S_ind) if not np.array_equal(S_val[i], val_to_delete)] 
        S_val = np.array([e for i,e in enumerate(S_val) if not np.array_equal(e, val_to_delete)])
    else:
        ind = ind.item()
        val_to_delete = S_val_rem[ind]
        S_ind = [e for i,e in enumerate(S_ind) if not np.array_equal(S_val[i], val_to_delete)] 
        S_val = np.array([e for i,e in enumerate(S_val) if not np.array_equal(e, val_to_delete)])
        
    return S_val, S_ind

def get_tabu_add(S_current, S_past):
    tabu_add = set()
    
    for S in S_past:
        if len(S) == len(S_current) + 1:
            if all(i in S for i in S_current):
                tabu_add.add([i for i in S if i not in S_current][0])
                        
    return tabu_add

def get_tabu_remove(S_current, S_past):
    tabu_remove = set()
    
    for S in S_past:
        if len(S) == len(S_current) - 1:
            if all(i in S_current for i in S):
                tabu_remove.add([i for i in S_current if i not in S][0])
                        
    return tabu_remove



