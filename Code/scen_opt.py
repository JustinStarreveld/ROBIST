# -*- coding: utf-8 -*-
"""
This file contains methods pertaining to existing scenario optimization methods

Created on Jan 25 2023

@author: Justin Starreveld: j.s.starreveld@uva.nl
@author: Guanyu Jin: g.jin@uva.nl
"""

# import external packages
import numpy as np
import scipy.stats
import math
import time
from sklearn.model_selection import train_test_split 
import warnings

def compute_ala2015_N_min(dim_x, risk_param_epsilon, conf_param_alpha):
    """Theorem 4 in Alamo, T., Tempo, R., Luque, A., & Ramirez, D. R. (2015). 
    Randomized methods for design of uncertain systems; Sample complexity and sequential algorithms
    
    Provides explicit bound on the sample size N needed to guarantee robustness
    (under conditions of convexity in x and feasibility and uniqueness)
    """
    N_min = 1/risk_param_epsilon * (math.exp(1) / (math.exp(1) - 1)) * (math.log(1/conf_param_alpha) + dim_x - 1)
    N_min = np.ceil(N_min).astype(int) 
    return N_min
                  
def determine_cal2005_N_min(dim_x, risk_param_epsilon, conf_param_alpha):
    def compute_cal2005_vio_bound(N, dim_x, risk_param_epsilon):
        bound = scipy.special.comb(N,dim_x,exact=False) * ((1 - risk_param_epsilon)**(N - dim_x))
        return bound
    
    # Do bisection search between 1 and ala2015_N_min to determine cal2005 N_min
    a = 1
    #f_a = compute_cam2008_vio_bound(a, dim_x, risk_param_epsilon)
    b = compute_ala2015_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    f_b = compute_cal2005_vio_bound(b, dim_x, risk_param_epsilon)
    if f_b == -1: # To catch overflow error
        return b
    while True:
        c = np.ceil((a+b)/2).astype(int)
        f_c = compute_cal2005_vio_bound(c, dim_x, risk_param_epsilon)   
        if abs(a-b) == 1:
            if f_c <= conf_param_alpha:
                return c
            else:
                return c-1
        if f_c > conf_param_alpha:
            a = c
        else:
            b = c
            
def determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha):
    # from decimal import Decimal
    def compute_cam2008_vio_bound(N, dim_x, risk_param_epsilon):
        with warnings.catch_warnings(): # to prevent "infeasible" warning
            warnings.simplefilter("ignore")
            try:
                bound = sum(scipy.special.comb(N,i,exact=False) * risk_param_epsilon**i * (1 - risk_param_epsilon)**(N - i) for i in range(dim_x))
            except OverflowError:
                print("Note: Overflow error in computing cam2008 bound, will return ala2015 bound instead")
                return -1
            if np.isnan(bound):
                return -1
            return bound
    
    # Do bisection search between 1 and ala2015_N_min to determine N_min
    a = 1
    #f_a = compute_cam2008_vio_bound(a, dim_x, risk_param_epsilon)
    b = compute_ala2015_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    f_b = compute_cam2008_vio_bound(b, dim_x, risk_param_epsilon)
    if f_b == -1: # To catch overflow error
        return b
    
    while True:
        c = math.ceil((a+b)/2)
        f_c = compute_cam2008_vio_bound(c, dim_x, risk_param_epsilon)   
        if abs(a-b) == 1:
            if f_c <= conf_param_alpha:
                return c
            else:
                return c-1
        if f_c > conf_param_alpha:
            a = c
        else:
            b = c
  
def test_cal2005_cam2008_functions():
    # To test if cal2005 and cam2008 functions are working properly by replicating Table 2 in Campi, M. C., & Garatti, S. (2008)
    dim_x = 10
    conf_param_alpha = 1e-5
    # cam2008_epsilon = [0.1, 0.05, 0.025, 0.01]#, 0.005, 0.0025, 0.001]
    cam2008_epsilon = [0.005, 0.0025, 0.001]
    
    for risk_param_epsilon in cam2008_epsilon:
        print("Epsilon: " + str(round(risk_param_epsilon, 4)), end = ', ')
        N_cal = determine_cal2005_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
        print("Cal: " + str(N_cal), end = ', ')
        N_campi = determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
        print("Campi: " + str(N_campi), end = ', ')
        N_ala2015 = compute_ala2015_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
        print("ala2015: " + str(N_ala2015))    
  
def solve_with_cal2005(solve_SCP, problem_instance, generate_unc_param_data, 
                             risk_param_epsilon, conf_param_alpha, dim_x, 
                             random_seed=0, **kwargs):
    start_time = time.time()
    N = determine_cal2005_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    data = generate_unc_param_data(random_seed, N, **kwargs)
    x, obj = solve_SCP(data, **problem_instance)
    runtime = time.time() - start_time
    return runtime, N, x, obj 

def solve_with_campi2008(solve_SCP, problem_instance, generate_unc_param_data,
                         risk_param_epsilon, conf_param_alpha, dim_x, 
                         random_seed=0, **kwargs):  
    start_time = time.time()
    N = determine_cam2008_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    data = generate_unc_param_data(random_seed, N, **kwargs)
    x, obj = solve_SCP(data, **problem_instance)
    runtime = time.time() - start_time
    return runtime, N, x, obj

def solve_with_cal2013(solve_SCP, problem_instance, dim_x, data, risk_param_epsilon, conf_param_alpha, q=-1):
    start_time = time.time()
    # 1) given N, determine maximum q such that rhs of eq 12 is no greater than N
    N = len(data)
    z_tol_cal = risk_param_epsilon
    n_cal = dim_x
    beta_cal = conf_param_alpha
    
    if q == -1:
        def eval_eq_12_cal2013(z_tol_cal, beta_cal, q, n_cal):
            return 2/z_tol_cal * math.log(1/beta_cal) + 4/z_tol_cal * (q+n_cal)
        
        # do bisection search to find maximum q
        a = 0
        b = N - n_cal - 1
        f_b = eval_eq_12_cal2013(z_tol_cal, beta_cal, b, n_cal)
        
        if f_b <= N:
            q = b
        else:
            while True:
                if b-a == 1:
                    if eval_eq_12_cal2013(z_tol_cal, beta_cal, b, n_cal) <= N:
                        q = b
                        break
                    else:
                        q = a
                        break
                
                c = math.ceil((a+b)/2)
                f_c = eval_eq_12_cal2013(z_tol_cal, beta_cal, c, n_cal)
                if f_c > N:
                    b = c
                else:
                    a = c
    
    # 2) iteratively, using Lagrange multiplier-based rule, discard q scenarios        
    # Start with all N scenarios and remove one-by-one
    original_setting = None
    if "get_dual_solution" in problem_instance.keys():
        original_setting = problem_instance['get_dual_solution']
        problem_instance.remove('get_dual_solution')
        
    num_removed = 0
    while num_removed < q:
        x, obj, duals = solve_SCP(data, **problem_instance, get_dual_solution=True)
        scen_i = np.argmax(duals)
        data = np.delete(data, scen_i, axis=0)
        num_removed += 1
        
    # return final solution
    x, obj, duals = solve_SCP(data, **problem_instance)
    
    if original_setting is not None:
        problem_instance['get_dual_solution'] = original_setting
    
    return x, obj, (time.time() - start_time), q

def determine_N_car2014(dim_x, risk_param_epsilon, conf_param_alpha):
    N_1_car2014 = 20 * (dim_x-1) # using the rule of thumb proposed in their paper
    try:
        B_eps = sum(math.comb(N_1_car2014, i)*(risk_param_epsilon**i)*((1-risk_param_epsilon)**(N_1_car2014 - i)) for i in range(dim_x+1))
        N_2_car2014 = math.ceil((math.log(conf_param_alpha) - math.log(B_eps)) / math.log(1-risk_param_epsilon))
    except OverflowError:
        # Equation (6) can be substituted by the handier formula:
        N_2_car2014 = math.ceil((1/risk_param_epsilon) * math.log(1/conf_param_alpha))
        
    return N_1_car2014, N_2_car2014
    

def solve_with_car2014(dim_x, risk_param_epsilon, conf_param_alpha, 
                       solve_SCP, unc_function, problem_instance, 
                       generate_data, generate_data_kwargs,
                       N_1=0, N_2=0, random_seed=0):
    start_time = time.time()
    if N_1 == 0 or N_2 == 0:
        N_1, N_2 = determine_N_car2014(dim_x, risk_param_epsilon, conf_param_alpha)

    # (2) sample N_1 and N_2 independent scenarios
    data = generate_data(random_seed, N_1+N_2, **generate_data_kwargs)
    data_1, data_2 = train_test_split(data, train_size=(N_1/(N_1+N_2)), random_state=random_seed)

    # (3) solve problem with N_1
    x_1, obj_1 = solve_SCP(data_1, **problem_instance)
    
    # (4) detuning step
    obj_f = max(obj_1, np.max(unc_function(x_1, data_2, **problem_instance)))
    runtime = time.time() - start_time
    return x_1, obj_f, runtime   

def determine_N_cal2016(dim_x, risk_param_epsilon, scale_eps_prime=0.7):
    """This method determines N by calibrating with respect to the 
    (asymptotic) bound on expected number of iterations = (1 − \beta_{eps_prime}(N))^−1
    we set N to the lowest possible s.t. this bound on iterations <= 10
    (this follows the approach taken in the numerical examples in Calafiore, G.C. (2016)).
    """
    start_time = time.time()
    eps_prime = scale_eps_prime*risk_param_epsilon
    N = dim_x
    while True:
        f_beta = scipy.stats.beta(dim_x, N+1-dim_x).cdf
        beta_eps_prime = 1 - f_beta(eps_prime)
        if beta_eps_prime == 1:
            N += 1
            continue
        ub_iter = 1 / (1 - beta_eps_prime)
        if ub_iter <= 10:
            return (time.time() - start_time), N
        N += 1

def determine_N_oracle_cal2016(dim_x, risk_param_epsilon, conf_param_alpha, N, scale_eps_prime=0.7):
    """This method is based on Equation (18) of Calafiore, G.C. (2016).
    - We assume that the number of samples (N) is given/fixed
    - In Section D.  it is stated that "We may suggest setting \epsilon^{\prime} 
    in the range [0.5, 0.9] \epsilon". Therefore we set the default value of scale_eps_prime=0.7
    """
    start_time = time.time()
    eps_prime = scale_eps_prime*risk_param_epsilon
    delta = risk_param_epsilon - eps_prime
    rhs = (risk_param_epsilon / delta) * math.log(1/conf_param_alpha) + dim_x - 1
    N_oracle = math.ceil((rhs - N*(delta/2 + eps_prime)) / delta)
    return (time.time() - start_time), N_oracle

def solve_with_cal2016(N, N_eval, dim_x, risk_param_epsilon, conf_param_alpha, 
                       solve_SCP, uncertain_constraint, problem_instance, 
                       generate_data, generate_data_kwargs, random_seed, 
                       numeric_precision=1e-6, verbose=False):
    # Collect info
    time_start = time.time()
    total_train_data_used = 0
    total_test_data_used = 0
    scale_eps_prime = 0.7 # "We suggest setting "\epsilon^{\prime} in the range [0.5, 0.9] \epsilon"
    cal_eps_prime = scale_eps_prime*risk_param_epsilon
    iter_k = 1
    while True:
        # Generate N i.i.d. samples
        S = generate_data(random_seed + iter_k, N, **generate_data_kwargs)
        total_train_data_used += N
        
        if verbose:
            print(f'iteration: {iter_k}, solving SCP with |S|={N}...')
            
        # Solve SCP with data
        start_time = time.time()
        x, obj = solve_SCP(S, **problem_instance)
        solve_time = round(time.time() - start_time)
        
        if verbose:
            print(f'iteration: {iter_k}, solved SCP in {solve_time} seconds')
    
        # Use "randomized oracle" to determine whether sufficient
        seed_k = random_seed + 99*iter_k
        S_eval = generate_data(seed_k, N_eval, **generate_data_kwargs)
        total_test_data_used += N_eval
        constr = uncertain_constraint(x, S_eval, **problem_instance)
        num_vio = sum(constr > -numeric_precision) 
        
        if num_vio <= cal_eps_prime * N_eval:
            flag = True
        else:
            flag = False
        
        if verbose:
            print(f'iteration: {iter_k}, using oracle with N_o={N_eval}, found {num_vio} violations, solution accepted? {flag}')
        
        if flag:
            total_time = time.time() - time_start
            return x, obj, iter_k, total_train_data_used, total_test_data_used, total_time
        else:
            iter_k += 1
            

def gar2022_determine_set_sizes(dim_x, risk_param_epsilon, conf_param_alpha, return_lbs_M=False):
    def gar2022_determine_lower_bounds_eq_5(max_j, gar_eps, gar_beta):
        lbs_M = list()
        for j in range(1, max_j+1):
            if j == 1:
                N = j
            else:
                N = lbs_M[j-1]
                
            while True:
                lhs = 0
                for i in range(j):
                    lhs += scipy.special.comb(N,i,exact=False) * (gar_eps)**i * (1 - gar_eps)**(N - i)
                if lhs <= gar_beta:
                    lbs_M.append(N)
                    if j == 1:
                        lbs_M.append(N) # M_0 = M_1
                    # print("computed M for j=",j,", where M_j=", N)
                    break
                else:
                    N = N + 1
        return lbs_M
        
    def gar2022_determine_set_sizes_eq_8(lbs_M, gar_d, gar_eps, gar_beta):
        set_sizes = []
        for j in range(gar_d+1):
            h_j = gar_beta / ((gar_d+1)*(lbs_M[j]+1))
            for m in range(j, lbs_M[j]+1):
                h_j += scipy.special.comb(m,j,exact=False) * (1-gar_eps)**(m-j)
            gar_alpha = min(gar_beta, h_j)
            rhs = (2/gar_eps)*(j*math.log((2/gar_eps)) +  math.log((1/gar_alpha))) + 1
            set_sizes.append(math.ceil(rhs))
        return set_sizes
    
    def gar2022_determine_set_sizes_bisection(gar_d, max_j, M_j, gar_eps, gar_beta, init_extreme_N):
        # bisection procedure as described in Section III. A. Computational aspects
        
        def gar2022_check_eq_7(gar_d, gar_eps, gar_beta, j, M_j):
            lhs = scipy.special.comb(M_j, j, exact=False) * (1-gar_eps)**(M_j-j)
            rhs = 0
            for m in range(j, M_j+1):
                rhs += scipy.special.comb(m, j, exact=False) * (1 - gar_eps)**(m - j)
            rhs = rhs * gar_beta / ((gar_d+1)*(M_j+1))

            if lhs <= rhs:
                return True
            else:
                return False
        
        set_sizes = []
        for j in range(max_j+1):
            # first check eq 7
            M_j = lbs_M[j]
            if gar2022_check_eq_7(gar_d, gar_eps, gar_beta, j, M_j):
                set_sizes.append(M_j)
            else:
                # bisection search between M_j and init_extreme_N
                a = M_j
                b = init_extreme_N
                while True:
                    c = np.ceil((a+b)/2).astype(int)
                    f_c = gar2022_check_eq_6(gar_d, gar_eps, gar_beta, j, M_j, c)
                    if abs(a-b) == 1:
                        if f_c == True:
                            set_sizes.append(c)
                            break
                        else:
                            set_sizes.append(c+1)
                            break
                    if f_c == True:
                        b = c
                    else:
                        a = c
    
        return set_sizes
    
    def gar2022_check_eq_6(gar_d, gar_eps, gar_beta, j, M_j, N):
        lhs = 0
        for m in range(j, M_j+1):
            lhs += scipy.special.comb(m, j, exact=False) * (1 - gar_eps)**(m - j)
        lhs = lhs * gar_beta / ((gar_d+1)*(M_j+1))
        rhs = scipy.special.comb(N, j, exact=False) * (1 - gar_eps)**(N - j)
        if lhs >= rhs:
            return True
        else:
            return False
    
    def gar2022_alg2_check_eq_6(gar_eps, lambda_k_k, k, M_k, N):
        lhs = 0
        for m in range(k, M_k+1):
            lhs += scipy.special.comb(m, k, exact=False) * (1 - gar_eps)**(m - k)
        lhs = lhs * lambda_k_k
        rhs = scipy.special.comb(N, k, exact=False) * (1 - gar_eps)**(N - k)
        if lhs >= rhs:
            return True
        else:
            return False
        
    
    set_size_time_start = time.time()
    
    # convert notation
    gar_d = dim_x
    gar_eps = risk_param_epsilon
    gar_beta = conf_param_alpha
    
    # some upper bound for N
    init_extreme_N = compute_ala2015_N_min(dim_x, risk_param_epsilon, conf_param_alpha)
    
    # if problem is large, we simplify the procedure by using the bisection procedure as 
    # described in Section III. A. Computational aspects only for j = 0,...,100
    if dim_x > 100:
        max_j = min(dim_x, 100)
        lbs_M = gar2022_determine_lower_bounds_eq_5(max_j, gar_eps, gar_beta)
        set_sizes = gar2022_determine_set_sizes_bisection(gar_d, max_j, lbs_M, gar_eps, gar_beta, init_extreme_N)
        time_elapsed = time.time() - set_size_time_start
        return time_elapsed, set_sizes
    

    # determine "lower bounds" M_j for j = 0,...,d
    lbs_M = gar2022_determine_lower_bounds_eq_5(gar_d, gar_eps, gar_beta)
    
    # Algorithm 2:
    init_lambda_d = gar_beta / (lbs_M[gar_d] + 1)
    # Now determine N^prime_d
    N = lbs_M[gar_d]
    while True:
        lhs = 0
        for m in range(gar_d, lbs_M[gar_d]+1):
            lhs += scipy.special.comb(m, gar_d, exact=False) * (1 - gar_eps)**(m - gar_d)
        lhs = lhs * init_lambda_d
        rhs = scipy.special.comb(N, gar_d, exact=False) * (1 - gar_eps)**(N - gar_d)
        if lhs >= rhs:
            N_prime_d = N
            break
        else:
            N = N + 1
    
    N_prime_vec = np.empty(gar_d+1, dtype=np.int)
    N_prime_vec[gar_d] =  N_prime_d
    
    # loop 1)
    for k in range(gar_d-1, -1, -1):
        # steps 1.1 and 1.2:
        lambda_k_prev = gar_beta / (lbs_M[gar_d]+1)
        for j in range(gar_d, k, -1):
            numer = scipy.special.comb(N_prime_vec[j], k, exact=False) * (1-gar_eps)**(N_prime_vec[j] - k)
            for m in range(lbs_M[j-1]+1, lbs_M[j]+1):
                numer = numer - lambda_k_prev * scipy.special.comb(m, k, exact=False) * (1-gar_eps)**(m - k)
            numer = max(numer, 0)
            
            denom = 0
            for m in range(k, lbs_M[j-1]+1):
                denom = denom + lambda_k_prev * scipy.special.comb(m, k, exact=False) * (1-gar_eps)**(m - k)
                
            mu_k_j = numer / denom
            if mu_k_j >= 1:
                print(f"Error in Alg2 of gar2022 with k={k} and j={j}: mu_k_j >= 1, will use eq(6) instead")
                set_sizes = gar2022_determine_set_sizes_bisection(gar_d, gar_d, lbs_M, gar_eps, gar_beta, init_extreme_N)
                time_elapsed = time.time() - set_size_time_start
                return time_elapsed, set_sizes
            lambda_k_j = (1-mu_k_j) * lambda_k_prev
            lambda_k_prev = lambda_k_j
        lambda_k_k = lambda_k_prev
        
        # step 1.3:
        # Bisection approach:
        M_k = lbs_M[k]
        a = M_k
        f_a = gar2022_alg2_check_eq_6(gar_eps, lambda_k_k, k, M_k, a)
        if f_a == True:
            N_prime_vec[k] = a
        else:
            b = init_extreme_N
            while True:
                c = np.ceil((a+b)/2).astype(int)
                f_c = gar2022_alg2_check_eq_6(gar_eps, lambda_k_k, k, M_k, c)
                if abs(a-b) == 1:
                    if f_c == True:
                        N_prime_vec[k] = c
                        break
                    else:
                        N_prime_vec[k] = c + 1
                        break
                if f_c == True:
                    b = c
                else:
                    a = c
            
    set_sizes = []
    set_sizes.append(int(N_prime_vec[0]))
    for k in range(1, gar_d + 1):
        if N_prime_vec[k] < set_sizes[k-1]:
            set_sizes.append(set_sizes[k-1])
        else:
            set_sizes.append(int(N_prime_vec[k]))
    
    time_elapsed = time.time() - set_size_time_start
    
    if return_lbs_M:
        return time_elapsed, set_sizes, lbs_M
    return time_elapsed, set_sizes

def solve_with_gar2022(dim_x, set_sizes, solve_SCP, uncertain_constraint, 
                        generate_data, random_seed, problem_instance, time_limit_solve=1*60*60,
                        numeric_precision=1e-6, **kwargs):
    time_main_solves = 0
    time_determine_supp = 0
    j = 0
    S = []
    while True:
        if j == 0:
            N_minus1 = 0
        else:
            N_minus1 = set_sizes[j-1]
        
        N_to_gen = int(set_sizes[j] - N_minus1)
        seed_j = random_seed + 99*j
        data_sample_j = generate_data(seed_j, N_to_gen, **kwargs)
        S = S + data_sample_j
        
        solve_start_time = time.time()
        x, obj = solve_SCP(S, **problem_instance)
        solve_time = time.time() - solve_start_time
        time_main_solves += solve_time
        
        # Determine number of support constraints
        supp_start_time = time.time()
        s_j = 0
        # Assume that support constraints can only be among active constraints
        constr = uncertain_constraint(x, S, **problem_instance)
        active = np.where(constr > -numeric_precision)[0]
        for i_scen in active:
            S_min = [s for i,s in enumerate(S) if i != i_scen]
            
            x_min, obj_min = solve_SCP(S_min, **problem_instance)
            if (x != x_min and obj != obj_min):
                s_j += 1
                if s_j > j:
                    break
            
        time_determine_supp += time.time() - supp_start_time
        
        if s_j <= j:
            return x, obj, j, s_j, set_sizes, time_main_solves, time_determine_supp
        else:
            j += 1
    




































