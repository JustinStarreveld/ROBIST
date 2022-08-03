# Import packages
import numpy as np
import cvxpy as cp
import scipy.stats
import time

def search_alg(data_train, beta, alpha, time_limit_search, time_limit_solve, 
               threshold_time_solve, max_nr_solutions, add_strategy, remove_strategy,
               improve_strategy, par, phi_div, phi_dot, numeric_precision,
               solve_SCP, uncertain_constraint):

    # Get extra info
    N_train = len(data_train)
    r = phi_dot/(2*N_train)*scipy.stats.chi2.ppf(1-alpha, 1)
    
    # Initialize algorithm
    start_time = time.time()
    Z_values = np.array([data_train[0]]) # Assume first index contains nominal data
    Z_indices = [0] # Tracks the indices of the scenarios in Z
    lb = -np.inf
    num_iter = {'add':0, 'remove':0, 'improve':0}
    solutions = []
    np.random.seed(1) # Set seed for random strategies
    
    while True:
        solve_start_time = time.time()
        [x, obj] = solve_SCP(Z_values, time_limit_solve)
        solve_time = time.time() - solve_start_time
            
        # Compute the lower bound on training data (to get a feel for feasibility)
        constr = uncertain_constraint(data_train, x)
        vio = constr[constr>(0+numeric_precision)]   
        p_vio = len(vio)/N_train
        p = np.array([1-p_vio, p_vio])
        
        if p_vio == 0:
            lb = 1
        else:
            lb = compute_lb(p, r, par, phi_div)
        
        solutions.append({'sol': x, 'obj': obj, 'time': (time.time()-start_time), 
                          'lb_train': lb, 'lb_test': np.nan, 'scenario_set': Z_indices.copy()})
                
        if len(solutions) >= max_nr_solutions:
            break
        
        if solve_time >= threshold_time_solve and len(Z_values) > 1: # Invoke removal scenarios (to improve solve efficiency)
            Z_values, Z_indices = remove_scenarios(remove_strategy, Z_values, Z_indices, 
                                                   x, uncertain_constraint, numeric_precision)
            num_iter['remove'] += 1
        elif lb >= beta: # have achieved feasibility, now we remove some scenarios
            Z_values, Z_indices = remove_scenarios(improve_strategy, Z_values, Z_indices, 
                                                   x, uncertain_constraint, numeric_precision)
            num_iter['improve'] += 1
        else: # Add scenario if lb still lower than beta
            Z_values, Z_indices = add_scenarios(add_strategy, data_train, Z_values, Z_indices, 
                                                constr, vio, beta, lb, numeric_precision) 
            num_iter['add'] += 1
        
        if (time.time()-start_time) >= time_limit_search:
            break   
    
    runtime = time.time() - start_time
    return runtime, num_iter, solutions  


def compute_lb(p, r, par, phi_div):
    q = cp.Variable(2, nonneg = True)
    constraints = [cp.sum(q) == 1]
    constraints = phi_div(p,q,r,par,constraints)
    obj = cp.Minimize(q[0])
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(prob.value)

def add_scenarios(add_strategy, data, Z_values, Z_indices, constr, vio, beta, lb, numeric_precision):
    ind = pick_scenarios_to_add(add_strategy, len(data), constr, vio, beta, lb, numeric_precision)
    Z_indices.append(ind)
    scen_to_add = np.array([data[ind]])
    Z_values = np.append(Z_values, scen_to_add, axis = 0)
    return Z_values, Z_indices

def pick_scenarios_to_add(add_strategy, N, constr, vio, beta, lb, numeric_precision):
    if add_strategy == 'smallest_vio':   # the least violated scenario is added   
        return np.where(constr == np.min(vio))[0][0]
    elif add_strategy == 'random_vio':
        return np.random.choice(np.where(constr > (0+numeric_precision))[0])
    elif add_strategy == 'N*(beta-lb)_smallest_vio':   # the N*(beta-lb)-th scenario is added
        rank = np.ceil(N*(beta-lb)).astype(int)
        if rank > len(vio):
            return np.where(constr == np.max(vio))[0][0]
        vio_sort = np.sort(vio) 
        vio_value = vio_sort[rank-1]     # -1 to correct for python indexing
        return np.where(constr == vio_value)[0][0]
    elif add_strategy == 'random_weighted_vio':
        rank = np.ceil(N*(beta-lb)).astype(int)
        if rank > len(vio) or rank < 2:
            return np.where(constr == np.max(vio))[0][0]
        vio_sort = np.sort(vio)  
        vio_ideal = (vio_sort[rank-1] + vio_sort[rank-2]) / 2    # -1 to correct for python indexing
        weights = [(1 / (abs(vio_ideal - i))) for i in vio]
        sum_weights = sum(weights)
        probs = [i/sum_weights for i in weights]
        ind = np.random.choice(a = len(vio), p = probs)  
        vio_chosen = vio[ind]
        return np.where(constr == vio_chosen)[0][0]
    else:
        print("Error: did not provide valid addition strategy")
        return None

def remove_scenarios(remove_strategy, Z_values, Z_indices, x, uncertain_constraint, numeric_precision):
    if remove_strategy == 'all_inactive':
        constr = uncertain_constraint(Z_values, x)
        ind = np.where(constr < (0-numeric_precision))[0]
        Z_values = np.delete(Z_values, ind, axis=0)
    elif remove_strategy == 'random_active':
        constr = uncertain_constraint(Z_values, x)
        active = np.where(constr > (0-numeric_precision))[0]
        ind = np.random.choice(active)
        Z_values = np.delete(Z_values, ind, axis=0)
    elif remove_strategy == 'random_any':
        ind = np.random.choice(len(Z_values))
        Z_values = np.delete(Z_values, ind, axis=0)
    else:
        print("Error: did not provide valid removal strategy")
    
    if isinstance(ind, np.ndarray):
        ind_set = set(ind.flatten())
        Z_indices = [i for j, i in enumerate(Z_indices) if j not in ind_set] 
    else:
        ind = ind.item()
        del Z_indices[ind]
    
    return Z_values, Z_indices

def evaluate_alg(solutions, data_test, beta, alpha, par, phi_div, phi_dot, 
                 uncertain_constraint, numeric_precision):
    start_time = time.time()
    
    # Get extra info
    N_test = len(data_test)
    r = phi_dot/(2*N_test)*scipy.stats.chi2.ppf(1-alpha, 1)
    
    # Store best solution info
    best_sol = {'sol': None}
    pareto_solutions = []
    
    for sol_info in solutions:
        x = sol_info['sol']
        obj = sol_info['obj']
        
        # Evaluate "real" lb on test data
        constr_test = uncertain_constraint(data_test, x)
        vio_test = constr_test[constr_test>(0+numeric_precision)]   
        p_vio = len(vio_test)/N_test
        p = np.array([1-p_vio, p_vio])
        
        if p_vio == 0:
            lb = 1
        else:
            lb = compute_lb(p, r, par, phi_div)

        sol_info['lb_test'] = lb
        
        # Determine if best solution can be replaced
        if best_sol['sol'] is None or (best_sol['lb_test'] < beta and lb > best_sol['lb_test']):
            best_sol = sol_info
        elif ((lb >= beta and obj > best_sol['obj']) 
              or (lb > best_sol['lb_test'] and obj >= best_sol['obj'])):
            best_sol = sol_info
            
        # Update list of Pareto efficient solutions
        if len(pareto_solutions) == 0:
            pareto_solutions.append((lb, obj))
        else:
            add_sol = True
            to_remove = []
            for i, (lb2, obj2) in enumerate(pareto_solutions):
                if lb >= lb2 and obj >= obj2:
                    to_remove.append(i)
                elif lb <= lb2 and obj <= obj2:
                    add_sol = False
            for index in sorted(to_remove, reverse=True):
                del pareto_solutions[index]
            if add_sol:
                pareto_solutions.append((lb, obj))
            
    runtime = time.time() - start_time
    return runtime, best_sol, pareto_solutions





