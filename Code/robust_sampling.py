# Import packages
import numpy as np
import cvxpy as cp
import scipy.stats
import time
import math

def gen_and_eval_alg(data_train, data_test, beta, alpha, time_limit_search, time_limit_solve, 
                        max_nr_solutions, add_strategy, remove_strategy, clean_strategy, 
                        add_remove_threshold, use_tabu,
                        phi_div, phi_dot, numeric_precision,
                        solve_SCP, uncertain_constraint, check_robust,
                        risk_measure, seed, 
                        num_obs_per_bin=None, data_eval=None, emp_eval_obj=None,
                        analytic_out_perf=None):

    # Get extra info
    k = data_train.shape[1]
    beta_l = beta - add_remove_threshold
    beta_u = beta + add_remove_threshold
    
    # Initialize algorithm
    start_time = time.time()
    S_val = np.array([data_train[0]]) # Assume first index contains nominal data
    S_ind = [0] # Tracks the indices of the scenarios in Z
    num_iter = {'add':0, 'remove':0, 'clean':0}
    feas_solutions = set()
    feas_solution_info = []
    all_solutions = []
    np.random.seed(seed) # Set seed for random strategies
    
    # Store best solution info
    best_sol = {'sol': None}
    pareto_solutions = []
    
    S_past = []
    count_duplicate_S = 0
    prev_x = None
    prev_obj = None
    
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
            solve_start_time = time.time()
            [x, obj] = solve_SCP(k, S_val, time_limit_solve)
            solve_time = time.time() - solve_start_time
            
            all_solutions.append({'sol': x, 'obj': obj, 'scenario_set': sorted(S_ind.copy())})
            S_past.append(sorted(S_ind.copy()))
        
        # something went wrong in solving SCP, revert back to previous solution
        if x is None and prev_x is not None:
            x = prev_x
            obj = prev_obj
        else:
            prev_x = x
            prev_obj = obj
        
        
        bound_train = compute_bound(risk_measure, alpha, phi_div, phi_dot, numeric_precision, 
                                    data_train, x, uncertain_constraint, num_obs_per_bin)
        
        
        bound_test = compute_bound(risk_measure, alpha, phi_div, phi_dot, numeric_precision,
                                   data_test, x, uncertain_constraint, num_obs_per_bin)
        
        
        if data_eval is not None and emp_eval_obj is not None:
            # eval_true_obj = emp_eval_obj(x, data_eval, 10, 0.2)
            eval_true_obj = analytic_out_perf(x, 10, 0.2)
            eval_true_exp = np.mean(uncertain_constraint(data_eval, x))
            print("---------------------------------------------")
            print("iter     : " + f'{round(count_iter,0):.0f}')
            print("size_S   : " + f'{round(len(S_val),0):.0f}')
            print("obj_SCP  : " + f'{round(-obj,3):.3f}')
            #print("obj_eval : " + f'{round(eval_true_obj,3):.3f}')
            print("obj_true : " + f'{round(eval_true_obj,3):.3f}')
            print("b_train  : " + f'{round(bound_train,3):.3f}')
            print("b_test   : " + f'{round(bound_test,3):.3f}')
            print('p_eval   : ' + f'{round(eval_true_exp,3):.3f}')
        
        x_satisfies_robust_condition = check_robust(bound_test, numeric_precision, beta)
        if x_satisfies_robust_condition and x.tostring() not in feas_solutions:
            feas_solutions.add(x.tostring())
            sol_info = {'sol': x, 'obj': obj, 'time': (time.time()-start_time), 
                        #'p_train':(1-p_vio_train), 'p_test': (1-p_vio_test),
                        'bound_train': bound_train,  'bound_test': bound_test, 'scenario_set': sorted(S_ind.copy())}
            feas_solution_info.append(sol_info)
            
            # Determine if best solution can be replaced
            if best_sol['sol'] is None or (not check_robust(best_sol['bound_test'], numeric_precision, beta) and 
                                           bound_is_better(risk_measure, bound_test, best_sol['bound_test'])):
                best_sol = sol_info
            elif (obj > best_sol['obj'] or (bound_is_better(risk_measure, bound_test, best_sol['bound_test']) 
                                            and obj >= best_sol['obj'])):
                best_sol = sol_info
        
        # Update list of Pareto efficient solutions
        if len(pareto_solutions) == 0:
            pareto_solutions.append((bound_test, obj))
        else:
            pareto_opt = True
            to_remove = []
            for i, (lb2, obj2) in enumerate(pareto_solutions):
                if bound_is_better(risk_measure, bound_test, lb2) and obj >= obj2:
                    to_remove.append(i)
                elif not bound_is_better(risk_measure, bound_test, lb2) and obj <= obj2:
                    pareto_opt = False
            for index in sorted(to_remove, reverse=True):
                del pareto_solutions[index]
            if pareto_opt:
                pareto_solutions.append((bound_test, obj))
        
        if len(feas_solutions) >= max_nr_solutions:
            break
        
        if use_tabu == True: 
            tabu_add = get_tabu_add(S_ind, S_past)
            tabu_remove = get_tabu_remove(S_ind, S_past)
        else:
            tabu_add = set()
            tabu_remove = set()
        
        constr_train = uncertain_constraint(data_train, x)
        constr_add, num_possible_additions = get_possible_additions(constr_train, tabu_add, numeric_precision)
        S_ind_rem, num_possible_removals = get_possible_removals(S_ind, tabu_remove)
        
        add_or_remove = determine_action(bound_train, beta, beta_l, beta_u, 
                                         num_possible_additions, num_possible_removals,
                                         check_robust, numeric_precision)
        if add_or_remove == True:
            S_val, S_ind = add_scenarios(add_strategy, data_train, S_val, S_ind, 
                                         constr_train, constr_add, beta, bound_train, numeric_precision) 
            num_iter['add'] += 1
        elif add_or_remove == False:
            S_val, S_ind = remove_scenarios(remove_strategy, S_val, S_ind, S_ind_rem,
                                            x, uncertain_constraint, numeric_precision)
            num_iter['remove'] += 1
        else:
            break # Finished
        
        if len(S_val) >= clean_strategy[0]: # Invoke clean strategy (to improve solve efficiency)
            S_val, S_ind = remove_scenarios(clean_strategy[1], S_val, S_ind, S_ind_rem,
                                                   x, uncertain_constraint, numeric_precision)
            num_iter['clean'] += 1
        
        if (time.time()-start_time) >= time_limit_search:
            break   
    
    runtime = time.time() - start_time

    #print("Duplicate samples encountered: " + str(count_duplicate_S))
    return runtime, num_iter, feas_solution_info, best_sol, pareto_solutions

def bound_is_better(risk_measure, bound1, bound2):
    if risk_measure == 'chance_constraint' or risk_measure == 'exp_constraint_geq':
        return (bound1 > bound2)
    elif risk_measure == 'exp_constraint_leq':
        return (bound1 < bound2)
    else:
        print("ERROR: do not recognize risk measure")
        return None

def compute_cc_lb(alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint):
    N = len(data)
    constr = uncertain_constraint(data, x)
    num_vio = sum(constr>(0+numeric_precision))
    p_vio = num_vio/N
    p = np.array([1-p_vio, p_vio])
    deg_of_freedom = 1
    
    if p_vio == 0:
        return 1
    else:
        r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, deg_of_freedom)
        q = cp.Variable(2, nonneg = True)
        constraints = [cp.sum(q) == 1]
        constraints = phi_div(p, q, r, None, constraints)
        obj = cp.Minimize(q[0])
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK)
        return prob.value

def compute_cc_lb_chi2_analytisch(p, phi_dot, N, alpha, par, phi_div):
    import sympy
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, 1)
    q = sympy.Symbol('q')
    sol = sympy.solvers.solve(p*((q/p) - 1)**2 + (1-p)*((1-q)/(1-p) - 1)**2 - r, q)
    return sol[0]
    
def compute_cc_lb_chi2_analytisch_2(p, phi_dot, N, alpha, par, phi_div):
    import math
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, 1)
    q_l = p - math.sqrt(-r * (p)**2 + r*p)
    #q_u = p + math.sqrt(-r * (p)**2 + r*p)
    return q_l
   
def compute_exp_lb(alpha, par, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint, min_obs_per_bin):
    N = len(data)
    constr = uncertain_constraint(data, x)
    constr_sort = np.sort(constr) 
    m = math.floor(N / min_obs_per_bin)
    deg_of_freedom = m-1
    
    bins = np.array_split(constr_sort, m)
    bin_thresholds = np.array([b[0] for b in bins])
    p = np.array([(b.size/N) for b in bins])
    
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, deg_of_freedom)
    q = cp.Variable(m, nonneg = True)
    constraints = [cp.sum(q) == 1]
    constraints = phi_div(p, q, r, None, constraints)
    obj_sum = 0
    for i in range(m):
        obj_sum = obj_sum + q[i] * bin_thresholds[i]
    obj = cp.Minimize(obj_sum)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)
    return prob.value
    
def compute_exp_ub(alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint, num_obs_per_bin):
    N = len(data)
    constr = uncertain_constraint(data, x)
    constr_sort = np.sort(constr) 
    m = math.floor(N / num_obs_per_bin)
    deg_of_freedom = m-1
    
    bins = np.array_split(constr_sort, m)
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

def compute_bound(risk_measure, alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint, num_obs_per_bin):
    if risk_measure == 'chance_constraint':
        return compute_cc_lb(alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint)
    elif risk_measure == 'exp_constraint_leq':
        return compute_exp_ub(alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint, num_obs_per_bin)
    elif risk_measure == 'exp_constraint_geq':
        return compute_exp_lb(alpha, phi_div, phi_dot, numeric_precision, data, x, uncertain_constraint, num_obs_per_bin)
    else:
        print("ERROR: do not recognize risk measure")
        return None
    

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

def determine_action(bound_train, beta, beta_l, beta_u, num_possible_additions, 
                     num_possible_removals, check_robust, numeric_precision):    
    # Determines whether it will be an add (True) or remove (False) or break (None) 
    if num_possible_additions == 0 and num_possible_removals == 0:
        return None
    elif num_possible_additions == 0:
        return False
    elif num_possible_removals == 0:
        return True
    
    if check_robust(bound_train, numeric_precision, beta_l) and not check_robust(bound_train, numeric_precision, beta_u):
        p_remove = (bound_train - beta_l) / (beta_u - beta_l)
        draw = np.random.uniform()
        if draw < p_remove:
            return False
        else:
            return True
    else:
        if check_robust(bound_train, numeric_precision, beta): # have achieved feasibility, now we remove some scenarios
            return False
        else: # Add scenario if lb still lower than beta
            return True

def add_scenarios(add_strategy, data, S_val, S_ind, constr, constr_add, beta, lb, numeric_precision):
    vio = constr_add[constr_add>(0+numeric_precision)]
    ind = pick_scenarios_to_add(add_strategy, len(data), constr, vio, beta, lb, numeric_precision)
    S_ind.append(ind)
    scen_to_add = np.array([data[ind]])
    if len(S_val) > 0:
        S_val = np.append(S_val, scen_to_add, axis = 0)
    else:
        S_val = scen_to_add
    return S_val, S_ind

def pick_scenarios_to_add(add_strategy, N, constr, vio, beta, lb, numeric_precision):
    if add_strategy == 'smallest_vio':   # the least violated scenario is added   
        return np.where(constr == np.min(vio))[0][0]
    elif add_strategy == 'random_vio':
        rand_vio = np.random.choice(vio)
        return np.where(constr == rand_vio)[0][0]
    elif add_strategy == 'N*(beta-lb)_smallest_vio':   # the N*(beta-lb)-th scenario is added
        rank = np.ceil(N*(beta-lb)).astype(int)
        if rank > len(vio):
            return np.where(constr == np.max(vio))[0][0]
        vio_sort = np.sort(vio) 
        vio_value = vio_sort[rank-1]     # -1 to correct for python indexing
        return np.where(constr == vio_value)[0][0]
    elif add_strategy == 'random_weighted_vio':
        vio_min = np.min(vio)
        vio_max = np.max(vio)
        vio_ideal = (beta-lb) * (vio_max - vio_min)
        weights = [(1 / (abs(vio_ideal - i))) for i in vio]
        sum_weights = sum(weights)
        probs = [i/sum_weights for i in weights]
        ind = np.random.choice(a = len(vio), p = probs)  
        vio_chosen = vio[ind]
        return np.where(constr == vio_chosen)[0][0]
    else:
        print("Error: did not provide valid addition strategy")
        return None

def remove_scenarios(remove_strategy, S_val, S_ind, S_ind_rem, x, uncertain_constraint, numeric_precision):
    
    S_val_rem = np.array([S_val[i] for i,e in enumerate(S_ind) if e in S_ind_rem])
    
    if remove_strategy == 'all_inactive':
        constr = uncertain_constraint(S_val_rem, x)
        ind = np.where(constr < (0-numeric_precision))[0]
        #S_val = np.delete(S_val, ind, axis=0)
    elif remove_strategy == 'random_inactive':
        constr = uncertain_constraint(S_val_rem, x)
        inactive = np.where(constr < (0-numeric_precision))[0]
        if len(inactive) > 0:
            ind = np.random.choice(inactive)
            #S_val = np.delete(S_val, ind, axis=0)
        else:
            ind = None
    elif remove_strategy == 'random_active':
        constr = uncertain_constraint(S_val_rem, x)
        active = np.where(constr > (0-numeric_precision))[0]
        if len(active) > 0:
            ind = np.random.choice(active)
            #S_val = np.delete(S_val, ind, axis=0)
        else:
            ind = None
    elif remove_strategy == 'random_any':
        ind = np.random.choice(len(S_val_rem))
        #S_val = np.delete(S_val, ind, axis=0)
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



