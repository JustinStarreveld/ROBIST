# Import packages
import numpy as np
import cvxpy as cp
import mosek
import scipy.stats
import math
import time

from robust_sampling import compute_lb

# Auxillary functions:
def compute_opt_given_data(alpha, beta, par, phi_div, phi_dot, data, time_limit_mosek):
    N = data.shape[0]
    M = 1000 #TODO: fix hardcode (M large enough such that constraint is made redundant)
    p_min, lb = determine_min_p(alpha, beta, par, phi_div, phi_dot, N)
    
    if p_min > 1:
        return None, None, None, None, p_min
    
    F = np.ceil(p_min * N)
    start_time = time.time()
    x, y, obj = opt_set(data, F, M, time_limit_mosek)
    runtime = time.time() - start_time
    sum_y = np.sum(y)
    return runtime, x, sum_y, obj, p_min

def opt_set(data, F, M, time_limit):
    N = data.shape[0]
    k = data.shape[1]
    x = cp.Variable(k, nonneg = True)
    y = cp.Variable(N, boolean = True)
    constraints = [cp.sum(x[0:(k-1)]) <= x[k-1]-1, x <= 10, data @ x <= 1 + (1-y)*M, cp.sum(y) >= F]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    return(x.value, y.value, prob.value)

def determine_min_p(alpha, beta, par, phi_div, phi_dot, N):
    # "fixed" settings for this procedure
    delta = 0.1
    stopping_criteria_epsilon = 0.0001
    r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, 1)
    p = np.array([beta, 1-beta])
    lb = compute_lb(p, r, par, phi_div)
    p_prev = p
    while True:
        if p[0] + delta > 1 - stopping_criteria_epsilon:
            delta = delta/10
        p = p + np.array([delta, -delta])
        lb = compute_lb(p, r, par, phi_div)
        if lb < beta:
            continue
        else:
            delta = delta / 10
            if delta < stopping_criteria_epsilon:
                break
            else:
                p = p_prev 
    return p[0], lb

def check_conv_comb(Z_arr):
    conv_comb_points = []
    if len(Z_arr) >= 3:
        for i in range(len(Z_arr)):
            z_i = Z_arr[i]
            Z_rest = np.append(Z_arr[:i], Z_arr[(i+1):], axis = 0)
            # solve optimization problem, if feasible, z_i is convex combination of points in Z_rest 
            # (https://en.wikipedia.org/wiki/Convex_combination)
            alpha = cp.Variable(len(Z_rest), nonneg = True)
            constraints = [alpha @ Z_rest == z_i, cp.sum(alpha) == 1]
            obj = cp.Maximize(alpha[0]) # no true objective function, only interested whether feasible solution exists
            prob = cp.Problem(obj,constraints)
            prob.solve(solver=cp.MOSEK)
            if prob.status != 'infeasible':
                conv_comb_points.append(i) 
    return conv_comb_points

def compute_calafiore_N_min(dim_x, beta, alpha):
    N_min = np.ceil(dim_x /((1-beta)*alpha)).astype(int) - 1
    return N_min

def compute_alamo_N_min(dim_x, beta, alpha):
    N_min = (math.exp(1) / (math.exp(1) - 1)) * (1 / (1-beta)) * (dim_x + math.log(1/alpha))
    N_min = np.ceil(N_min).astype(int) 
    return N_min
                   
def compute_calafiore_vio_bound(N, dim_x, beta):
    bound = math.comb(N, dim_x) * ((1 - (1-beta))**(N - dim_x))
    return bound

def compute_campi_vio_bound(N, dim_x, beta):
    bound = 0
    for i in range(dim_x):
        try:
            bound += math.comb(N, i) * ((1-beta)**i) * (1 - (1-beta))**(N - i)
        except OverflowError:
            print("Note: Overflow error in computing Campi bound, will return Alamo bound instead")
            return -1
    return bound

def determine_calafiore_N_min(dim_x, beta, alpha):
    # Do bisection search between 1 and calafiore N_min
    a = 1
    #f_a = compute_campi_vio_bound(a, dim_x, beta)
    b = compute_alamo_N_min(dim_x, beta, alpha)
    f_b = compute_calafiore_vio_bound(b, dim_x, beta)
    if f_b == -1: # To catch overflow error with computing bounds for large k
        return b
    
    while True:
        c = np.ceil((a+b)/2).astype(int)
        f_c = compute_calafiore_vio_bound(c, dim_x, beta)   
        if abs(a-b) == 1:
            if f_c <= alpha:
                return c
            else:
                return c-1
        if f_c > alpha:
            a = c
            #f_a = f_c
        else:
            b = c
            #f_b = f_c
            
def determine_campi_N_min(dim_x, beta, alpha):
    # Do bisection search between 1 and calafiore N_min
    a = 1
    #f_a = compute_campi_vio_bound(a, dim_x, beta)
    b = compute_alamo_N_min(dim_x, beta, alpha)
    f_b = compute_campi_vio_bound(b, dim_x, beta)
    if f_b == -1: # To catch overflow error with computing bounds for large k
        return b
    
    while True:
        c = np.ceil((a+b)/2).astype(int)
        f_c = compute_campi_vio_bound(c, dim_x, beta)   
        if abs(a-b) == 1:
            if f_c <= alpha:
                return c
            else:
                return c-1
        if f_c > alpha:
            a = c
            #f_a = f_c
        else:
            b = c
            #f_b = f_c
  
def test_calafiore_campi_functions():
    # EXTRA: Test if calafiore and campi functions are working properly by replicating Table 2 in Campi, M. C., & Garatti, S. (2008)
    dim_x = 10
    campi_epsilon = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
    li_beta = [(1-eps) for eps in campi_epsilon]
    alpha = 10**-5
    
    for beta in li_beta:
        print("Epsilon: " + str(round(1-beta, 4)), end = ', ')
        N_cal = determine_calafiore_N_min(dim_x, beta, alpha)
        print("Cal: " + str(N_cal), end = ', ')
        N_campi = determine_campi_N_min(dim_x, beta, alpha)
        print("Campi: " + str(N_campi), end = ', ')
        N_alamo = compute_alamo_N_min(dim_x, beta, alpha)
        print("Alamo: " + str(N_alamo))    
  
def solve_with_campi_N(solve_SCP, data, time_limit_solve):  
    start_time = time.time()
    [x, obj] = solve_SCP(data, time_limit_solve)
    runtime = time.time() - start_time
    return runtime, x, obj

def solve_with_calafiore2016(N, N_o, dim_x, beta, alpha, solve_SCP, uncertain_constraint, 
                            generate_data, random_seed, time_limit_solve,
                            numeric_precision):
    # Collect info
    time_start = time.time()
    total_train_data_used = 0
    total_test_data_used = 0
    
    # Using Calafiore2016 notation
    iter_k = 1
    
    # We suggest setting "\epsilon^{\prime} in the range [0.5, 0.9] \epsilon"
    epsilon_prime = 0.7(1-beta)
    
    while True:
        # Generate N i.i.d. samples
        S = generate_data(random_seed + iter_k, dim_x, N)
        total_train_data_used += N
        
        # Solve SCP with data
        x, obj = solve_SCP(dim_x, S, time_limit_solve)
    
        # Use "randomized oracle" to determine whether sufficient
        S_eval = generate_data(random_seed + 99*iter_k, dim_x, N_o)
        total_test_data_used += N_o
        constr = uncertain_constraint(S_eval, x)
        num_vio = sum(constr>(0+numeric_precision)) 
        
        if num_vio <= epsilon_prime * N_o:
            flag = True
        else:
            flag = False
            
        if flag:
            total_time = time.time() - time_start
            return x, obj, iter_k, total_train_data_used, total_test_data_used, total_time
        else:
            iter_k += 1
            

def solve_with_Garatti2022(dim_x, beta, alpha, solve_SCP, uncertain_constraint, 
                           generate_data, random_seed, time_limit_solve,
                           numeric_precision):
    
    time_main_solves = 0
    time_determine_supp = 0
    
    set_size_time_start = time.time()
    set_sizes_N = Garatti2022_determine_size_of_sets(dim_x, beta, alpha)
    time_determine_set_sizes = time.time() - set_size_time_start
    
    j = 0
    S = np.array([])
    while True:
        if j == 0:
            N_min_1 = 0
        else:
            N_min_1 = set_sizes_N[j-1]
        
        N_to_gen = int(set_sizes_N[j] - N_min_1)
        seed_j = random_seed + j
        data_sample_j = generate_data(seed_j, dim_x, N_to_gen)
        
        if len(S) == 0:
            S = data_sample_j
        else:
            S = np.append(S, data_sample_j, axis = 0)
        
        solve_start_time = time.time()
        [x, obj] = solve_SCP(dim_x, S, time_limit_solve)
        solve_time = time.time() - solve_start_time
        time_main_solves += solve_time
        
        # Determine number of support constraints
        supp_start_time = time.time()
        s_j = 0
        # Assume that support constraints can only be among active constraints
        constr = uncertain_constraint(S, x)
        active = np.where(constr > (0-numeric_precision))[0]
        for i_scen in active:
            S_min = np.copy(S)
            S_min = np.delete(S_min, i_scen, axis=0)
            
            [x_min, obj_min] = solve_SCP(dim_x, S_min, time_limit_solve)
            if (not np.array_equal(x, x_min) and obj != obj_min):
                s_j += 1
            
        time_determine_supp += time.time() - supp_start_time
        
        if s_j <= j:
            return x, obj, j, s_j, len(S), time_determine_set_sizes, time_main_solves, time_determine_supp
        else:
            j += 1
    
def Garatti2022_determine_size_of_sets(dim_x, beta, alpha):
    # Convert notation
    gar_d = dim_x
    gar_eps = 1 - beta
    gar_beta = alpha
    
    # Algorithm 2
    # First determine "lower bounds" M_j for j = 0,...,d
    lbs_M = list()
    for j in range(gar_d+1):
        N = j
        while True:
            lhs = 0
            for i in range(j):
                lhs += math.comb(N, i) * ((gar_eps)**i) * (1 - gar_eps)**(N - i)
            if lhs <= gar_beta:
                lbs_M.append(N)
                break
            else:
                N = N + 1
        
    init_lambda_d = gar_beta / (lbs_M[gar_d] + 1)
    # Now determine N_prime_d
    N = lbs_M[gar_d]
    while True:
        lhs = 0
        for m in range(gar_d, lbs_M[gar_d]+1):
            lhs += math.comb(m, gar_d) * (1 - gar_eps)**(m - gar_d)
        lhs = lhs * init_lambda_d
        rhs = math.comb(N, gar_d) * (1 - gar_eps)**(N - gar_d)
        if lhs >= rhs:
            N_prime_d = N
            break
        else:
            N = N + 1
    
    N_prime_vec = np.empty(gar_d+1, dtype=np.int)
    N_prime_vec[gar_d] =  N_prime_d
    
    for k in range(gar_d-1, -1, -1):
        # steps 1.1 and 1.2:
        lambda_k_prev = gar_beta / (lbs_M[gar_d] + 1)
        for j in range(gar_d, k, -1):
            
            numer = math.comb(N_prime_vec[j], k) * (1-gar_eps)**(N_prime_vec[j] - k)
            for m in range(lbs_M[j-1]+1, lbs_M[j] + 1):
                numer = numer - lambda_k_prev * (math.comb(m, k) * (1-gar_eps)**(m - k))
            numer = max(numer, 0)
            
            denom = 0
            for m in range(k, lbs_M[j-1] + 1):
                denom = denom + lambda_k_prev * (math.comb(m, k) * (1-gar_eps)**(m - k))
                
            mu_k_j = numer / denom
            if mu_k_j >= 1:
                print("Error: mu_k_j >= 1 in garatti2022 method")
                return None
            lambda_k_j = (1-mu_k_j) * lambda_k_prev
            lambda_k_prev = lambda_k_j
        lambda_k_k = lambda_k_prev
        
        # step 1.3:
        N = lbs_M[k]
        while True:
            lhs = 0
            for m in range(k, lbs_M[gar_d]+1):
                lhs += math.comb(m, k) * (1 - gar_eps)**(m - k)
            lhs = lhs * lambda_k_k
            rhs = math.comb(N, k) * (1 - gar_eps)**(N - k)
            if lhs >= rhs:
                N_prime_k = N
                N_prime_vec[k] =  N_prime_k
                break
            else:
                N = N + 1
        
    N_vec = np.zeros(gar_d+1)
    N_vec[0] = N_prime_vec[0]
    for k in range(1, gar_d + 1):
        if N_prime_vec[k] < N_vec[k-1]:
            N_vec[k] = N_vec[k-1]
        else:
            N_vec[k] = N_prime_vec[k]
    
    return N_vec

#### For CVaR, substitute slope = np.array([1/(1-beta),0]) and const = np.array([0,1])
#### Choose phi_conj from the phi-divergence file
def af_RC_exp_pmin(p,R,r,phi_conj,slope,const):  
    N = len(p)
    I = len(R[0])
    K = len(slope)
    theta = cp.Variable(1)
    lbda = cp.Variable((N,K), nonneg = True)
    a = cp.Variable(I)
    v = cp.Variable(K, nonneg = True)
    alpha = cp.Variable(1)
    beta = cp.Variable(1)
    gamma = cp.Variable(1,nonneg = True)
    t = cp.Variable(N)
    s = cp.Variable(N)
    w = cp.Variable(N)
    constraints = [a >= 0, cp.sum(a) == 1]
    for i in range(N):
        constraints.append(-(R @ a)[i] - cp.sum(lbda[i]) - beta <= 0)
        constraints.append(s[i] == -alpha + lbda[i]@slope)
        constraints.append(lbda[i] <= v)
        constraints = phi_conj(gamma,s[i],t[i],w[i],constraints)
    constraints.append(alpha + beta + gamma * r  + v@const + p@t <= -theta)
    obj = cp.Maximize(theta)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)
    return(a.value, prob.value)































