# Import packages
import numpy as np
import scipy.stats
import math
import time
from decimal import Decimal 
from sklearn.model_selection import train_test_split 

# # To compute general phi-divergence bounds:
# from phi_divergence import mod_chi2_cut
# def compute_phi_div_lowerbound(p_feas, N, conf_param_alpha, phi_div=mod_chi2_cut, phi_dot=2):
#     """
#     phi_div: function
#         Specifies the phi-divergence distance, default: phi_divergence.mod_chi2_cut()
#     phi_dot: int
#         Specifies the 2nd order derivative of phi-div function evaluated at 1, default: 2
#     """
#     import cvxpy as cp
#     dof = 1
#     r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-conf_param_alpha, dof)
#     p = np.array([p_feas, 1-p_feas])
#     q = cp.Variable(2, nonneg = True)
#     constraints = [cp.sum(q) == 1]
#     constraints = phi_div(p, q, r, None, constraints)
#     obj = cp.Minimize(q[0])
#     prob = cp.Problem(obj, constraints)
#     prob.solve(solver=cp.GUROBI)
#     return prob.value

def compute_mod_chi2_lowerbound(p_feas, N, conf_param_alpha):
    r = 1/N*scipy.stats.chi2.ppf(1-conf_param_alpha, 1)
    q_feas = p_feas - math.sqrt(p_feas*(1-p_feas)*r)
    return q_feas

def compute_alamo_N_min(dim_x, desired_prob_rhs, conf_param_alpha):
    N_min = (math.exp(1) / (math.exp(1) - 1)) * (1 / (1-desired_prob_rhs)) * (dim_x + math.log(1/conf_param_alpha))
    N_min = np.ceil(N_min).astype(int) 
    return N_min
                  
def determine_calafiore_N_min(dim_x, desired_prob_rhs, conf_param_alpha):
    def compute_calafiore_vio_bound(N, dim_x, desired_prob_rhs):
        bound = math.comb(N, dim_x) * ((1 - (1-desired_prob_rhs))**(N - dim_x))
        return bound
    
    # Do bisection search between 1 and alamo_N_min to determine calafiore N_min
    a = 1
    #f_a = compute_campi_vio_bound(a, dim_x, desired_prob_rhs)
    b = compute_alamo_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
    f_b = compute_calafiore_vio_bound(b, dim_x, desired_prob_rhs)
    if f_b == -1: # To catch overflow error with computing bounds for large dim_x / high desired_prob_rhs 
        return b
    while True:
        c = np.ceil((a+b)/2).astype(int)
        f_c = compute_calafiore_vio_bound(c, dim_x, desired_prob_rhs)   
        if abs(a-b) == 1:
            if f_c <= conf_param_alpha:
                return c
            else:
                return c-1
        if f_c > conf_param_alpha:
            a = c
        else:
            b = c
            
def determine_campi_N_min(dim_x, desired_prob_rhs, conf_param_alpha):
    def compute_campi_vio_bound(N, dim_x, desired_prob_rhs):
        try:
            bound = sum(math.comb(N, i) * ((1-desired_prob_rhs)**i) * (1 - (1-desired_prob_rhs))**(N - i) for i in range(dim_x+1))
        except OverflowError:
            print("Note: Overflow error in computing Campi bound, will return Alamo bound instead")
            return -1
        return bound
    
    # Do bisection search between 1 and alamo_N_min to determine calafiore N_min
    a = 1
    #f_a = compute_campi_vio_bound(a, dim_x, desired_prob_rhs)
    b = compute_alamo_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
    f_b = compute_campi_vio_bound(b, dim_x, desired_prob_rhs)
    if f_b == -1: # To catch overflow error with computing bounds for large k
        return b
    
    while True:
        c = math.ceil((a+b)/2)
        f_c = compute_campi_vio_bound(c, dim_x, desired_prob_rhs)   
        if abs(a-b) == 1:
            if f_c <= conf_param_alpha:
                return c
            else:
                return c-1
        if f_c > conf_param_alpha:
            a = c
        else:
            b = c
            
def determine_N_calafiore2016(dim_x, desired_prob_rhs, conf_param_alpha, scale_eps_prime, N_eval):
    start_time = time.time()
    # Convert to Cal notation
    cal_eps = 1-desired_prob_rhs
    cal_eps_prime = scale_eps_prime*cal_eps
    cal_delta = cal_eps - cal_eps_prime
    cal_beta = conf_param_alpha
    cal_n = dim_x
    N = cal_n
    while True:
        f_beta = scipy.stats.beta(cal_n, N + 1 - cal_n).cdf
        beta_eps_prime = 1 - f_beta(cal_eps_prime)
        ub_iter = 1 / max((1 - beta_eps_prime), 1e-10)
        if ub_iter > 10e6:
            N += 1
            continue
        lhs = N_eval * cal_delta + N*((cal_delta)/2 + cal_eps_prime)
        rhs = (cal_eps / cal_delta)*(math.log(1/cal_beta)) + cal_n - 1
        
        if lhs >= rhs:
            return N, (time.time() - start_time)
        else:
            N += 1
  
def test_calafiore_campi_functions():
    # EXTRA: Test if calafiore and campi functions are working properly by replicating Table 2 in Campi, M. C., & Garatti, S. (2008)
    dim_x = 10
    campi_epsilon = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
    li_desired_prob_rhs = [(1-eps) for eps in campi_epsilon]
    conf_param_alpha = 10**-5
    
    for desired_prob_rhs in li_desired_prob_rhs:
        print("Epsilon: " + str(round(1-desired_prob_rhs, 4)), end = ', ')
        N_cal = determine_calafiore_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
        print("Cal: " + str(N_cal), end = ', ')
        N_campi = determine_campi_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
        print("Campi: " + str(N_campi), end = ', ')
        N_alamo = compute_alamo_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
        print("Alamo: " + str(N_alamo))    
  
def solve_with_calafiore2005(solve_SCP, problem_instance, generate_unc_param_data, 
                             risk_param_epsilon, conf_param_alpha, dim_x, 
                             random_seed=0, **kwargs):
    start_time = time.time()
    N = determine_calafiore_N_min(dim_x, 1-risk_param_epsilon, conf_param_alpha)
    data = generate_unc_param_data(random_seed, N, **kwargs)
    x, obj = solve_SCP(data, **problem_instance)
    runtime = time.time() - start_time
    return runtime, N, x, obj 

def solve_with_campi2008(solve_SCP, problem_instance, generate_unc_param_data,
                         risk_param_epsilon, conf_param_alpha, dim_x, 
                         random_seed=0, **kwargs):  
    start_time = time.time()
    N = determine_campi_N_min(dim_x, 1-risk_param_epsilon, conf_param_alpha)
    data = generate_unc_param_data(random_seed, N, **kwargs)
    x, obj = solve_SCP(data, **problem_instance)
    runtime = time.time() - start_time
    return runtime, N, x, obj

def solve_with_care2014(solve_SCP, problem_instance, generate_unc_param_data, 
                        eval_unc_obj, conf_param_alpha, dim_x, N_1=0, N_2=0,
                        random_seed=0, **kwargs):
    start_time = time.time()
    if N_1 == 0:
        N_1 = 20*dim_x
    epsilon = 1 - eval_unc_obj['info']['desired_rhs']
    
    # (1) compute the smallest integer N_2 such that (6) holds
    if N_2 == 0:
        try:
            B_eps = sum(math.comb(N_1, i)*(epsilon**i)*((1-epsilon)**(N_1 - i)) for i in range(dim_x+1))
            N_2 = math.ceil((math.log(conf_param_alpha) - math.log(B_eps)) / math.log(1-epsilon))
        except OverflowError:
            # Equation (6) can be substituted by the handier formula:
            N_2 = math.ceil((1/epsilon) * math.log(1/conf_param_alpha))
        if N_2 <= 0:
            print("ERROR: N_2 (Care) <= 0. Likely that N_1 set too high for desired epsilon")

    # (2) sample N_1 and N_2 independent scenarios
    data = generate_unc_param_data(random_seed, N_1+N_2, **kwargs)
    data_1, data_2 = train_test_split(data, train_size=(N_1/(N_1+N_2)), random_state=random_seed)

    # (3) solve problem with N_1
    # start_time = time.time()
    x_1, obj_1 = solve_SCP(data_1, **problem_instance)
    # print("Runtime solve:", (time.time() - start_time))
    
    # (4) detuning step
    # start_time = time.time()
    unc_obj_func = eval_unc_obj['function']
    obj_f = max(obj_1, np.max(unc_obj_func(x_1, data_2, **problem_instance)))
    # print("Runtime detuning:", (time.time() - start_time))
    
    runtime = time.time() - start_time
    return x_1, obj_f, N_2, runtime   

def solve_with_calafiore2016(N, N_eval, scale_eps_prime, dim_x, desired_prob_rhs, conf_param_alpha, solve_SCP, uncertain_constraint, 
                            generate_data, random_seed, time_limit_solve,
                            numeric_precision):
    # Collect info
    time_start = time.time()
    total_train_data_used = 0
    total_test_data_used = 0
    
    # Using Calafiore2016 notation
    iter_k = 1
    
    # We suggest setting "\epsilon^{\prime} in the range [0.5, 0.9] \epsilon"
    cal_eps = 1-desired_prob_rhs
    cal_eps_prime = scale_eps_prime*cal_eps
    
    while True:
        # Generate N i.i.d. samples
        S = generate_data(random_seed + iter_k, dim_x, N)
        total_train_data_used += N
        
        # Solve SCP with data
        x, obj = solve_SCP(dim_x, S, time_limit_solve)
    
        # Use "randomized oracle" to determine whether sufficient
        S_eval = generate_data(random_seed + 99*iter_k, dim_x, N_eval)
        total_test_data_used += N_eval
        constr = uncertain_constraint(S_eval, x)
        num_vio = sum(constr>(0+numeric_precision)) 
        
        if num_vio <= cal_eps_prime * N_eval:
            flag = True
        else:
            flag = False
            
        if flag:
            total_time = time.time() - time_start
            return x, obj, iter_k, total_train_data_used, total_test_data_used, total_time
        else:
            iter_k += 1
            

def solve_with_garatti2022(dim_x, set_sizes, solve_SCP, uncertain_constraint, 
                           generate_data, random_seed, time_limit_solve,
                           numeric_precision):
    time_main_solves = 0
    time_determine_supp = 0
    j = 0
    S = np.array([])
    while True:
        if j == 0:
            N_min_1 = 0
        else:
            N_min_1 = set_sizes[j-1]
        
        N_to_gen = int(set_sizes[j] - N_min_1)
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
            return x, obj, j, s_j, set_sizes, time_main_solves, time_determine_supp
        else:
            j += 1
    
def garatti2022_determine_set_sizes(dim_x, desired_prob_rhs, conf_param_alpha):
    set_size_time_start = time.time()
    # Convert notation
    gar_d = dim_x
    gar_eps = 1 - desired_prob_rhs
    gar_beta = conf_param_alpha
    
    # First determine "lower bounds" M_j for j = 0,...,d
    lbs_M = list()
    for j in range(gar_d+1):
        if j == 0:
            N = j
        else:
            N = lbs_M[j-1]
            
        while True:
            lhs = 0
            for i in range(j):
                lhs += Decimal(math.comb(N, i)) * Decimal(((gar_eps)**i) * (1 - gar_eps)**(N - i))
            if lhs <= gar_beta:
                lbs_M.append(N)
                break
            else:
                N = N + 1
                
    # If problem is large, we simplify and use the explicit bound
    if dim_x > 100:
        time_elapsed = time.time() - set_size_time_start
        set_sizes = garatti2022_determine_set_sizes_eq_8(lbs_M, gar_d, gar_eps, gar_beta)
        return set_sizes, time_elapsed
    
    # Algorithm 2:
        
    init_lambda_d = gar_beta / (lbs_M[gar_d] + 1)
    # Now determine N_prime_d
    N = lbs_M[gar_d]
    while True:
        lhs = 0
        for m in range(gar_d, lbs_M[gar_d]+1):
            lhs += Decimal(math.comb(m, gar_d)) * Decimal((1 - gar_eps)**(m - gar_d))
        lhs = lhs * Decimal(init_lambda_d)
        rhs = Decimal(math.comb(N, gar_d)) * Decimal((1 - gar_eps)**(N - gar_d))
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
                
            mu_k_j = Decimal(numer) / Decimal(denom)
            if mu_k_j >= 1:
                print("Error: mu_k_j >= 1 in garatti2022 method")
                return None
            lambda_k_j = (1-mu_k_j) * lambda_k_prev
            lambda_k_prev = lambda_k_j
        lambda_k_k = lambda_k_prev
        
        # step 1.3:
            
        # Increment by 1 approach:
        # N = lbs_M[k]
        # while True:
        #     lhs = 0
        #     for m in range(k, lbs_M[k]+1):
        #         lhs += math.comb(m, k) * (1 - gar_eps)**(m - k)
        #     lhs = lhs * lambda_k_k
        #     rhs = math.comb(N, k) * (1 - gar_eps)**(N - k)
        #     if lhs >= rhs:
        #         N_prime_k = N
        #         N_prime_vec[k] =  N_prime_k
        #         break
        #     else:
        #         N = N + 1
        
        # Bisection approach:
        M_k = lbs_M[k]
        a = M_k
        f_a = garatti2022_check_eq_6(dim_x, gar_eps, lambda_k_k, k, j, M_k, a)
        if f_a == True:
            N_prime_vec[k] = a
        else:
            b = compute_alamo_N_min(dim_x, desired_prob_rhs, conf_param_alpha)
            while True:
                c = np.ceil((a+b)/2).astype(int)
                f_c = garatti2022_check_eq_6(dim_x, gar_eps, lambda_k_k, k, j, M_k, c)
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
            
    N_vec = np.zeros(gar_d+1)
    N_vec[0] = N_prime_vec[0]
    for k in range(1, gar_d + 1):
        if N_prime_vec[k] < N_vec[k-1]:
            N_vec[k] = N_vec[k-1]
        else:
            N_vec[k] = N_prime_vec[k]
    
    time_elapsed = time.time() - set_size_time_start
    return N_vec, time_elapsed

def garatti2022_check_eq_6(dim_x, gar_eps, lambda_k_k, k, j, M_k, N):
    lhs = 0
    for m in range(k, M_k+1):
        lhs += Decimal(math.comb(m, k)) * Decimal((1 - gar_eps)**(m - k))
    lhs = lhs * Decimal(lambda_k_k)
    rhs = Decimal(math.comb(N, k)) * Decimal((1 - gar_eps)**(N - k))
    if lhs >= rhs:
        return True
    else:
        return False

def garatti2022_determine_set_sizes_eq_8(lbs_M, gar_d, gar_eps, gar_beta):
    set_sizes = list()

    for j in range(gar_d+1):
        h_j = Decimal(gar_beta) / Decimal(((gar_d+1)*(lbs_M[j]+1)))
        for m in range(j, lbs_M[j]+1):
            h_j += Decimal(math.comb(m,j))*Decimal((1-gar_eps)**(m-j))
            
        gar_alpha = min(gar_beta, h_j)
            
        rhs = (2/gar_eps)*(j*math.log((2/gar_eps)) +  math.log((1/gar_alpha))) + 1
        set_sizes.append(math.ceil(rhs))
        
    return set_sizes


# OLD CODE: can be deleted at some point...

#TODO: rewrite this method
# def compute_opt_given_data(conf_param_alpha, desired_prob_rhs, phi_div, phi_dot, data):
    # N = data.shape[0]
    # M = 1000 #TODO: fix hardcode (M large enough such that constraint is made redundant)
    # p_min, lb = determine_min_p(N, conf_param_alpha, desired_prob_rhs, phi_div, phi_dot)
    
    # if p_min > 1:
    #     return None, None, None, None, p_min
    
    # F = np.ceil(p_min * N)
    # start_time = time.time()
    # x, y, obj = opt_set(data, F, M, None)
    # runtime = time.time() - start_time
    # sum_y = np.sum(y)
    # return runtime, x, sum_y, obj, p_min

# def opt_set(data, F, M, time_limit):
#     N = data.shape[0]
#     k = data.shape[1]
#     x = cp.Variable(k, nonneg = True)
#     y = cp.Variable(N, boolean = True)
#     constraints = [cp.sum(x[0:(k-1)]) <= x[k-1]-1, x <= 10, data @ x <= 1 + (1-y)*M, cp.sum(y) >= F]
#     obj = cp.Maximize(cp.sum(x))
#     prob = cp.Problem(obj,constraints)
#     prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
#     return(x.value, y.value, prob.value)

# Almost never occurs in higher dimensions
# def check_conv_comb(Z_arr):
#     conv_comb_points = []
#     if len(Z_arr) >= 3:
#         for i in range(len(Z_arr)):
#             z_i = Z_arr[i]
#             Z_rest = np.append(Z_arr[:i], Z_arr[(i+1):], axis = 0)
#             # solve optimization problem, if feasible, z_i is convex combination of points in Z_rest 
#             # (https://en.wikipedia.org/wiki/Convex_combination)
#             alpha = cp.Variable(len(Z_rest), nonneg = True)
#             constraints = [alpha @ Z_rest == z_i, cp.sum(alpha) == 1]
#             obj = cp.Maximize(alpha[0]) # no true objective function, only interested whether feasible solution exists
#             prob = cp.Problem(obj,constraints)
#             prob.solve(solver=cp.MOSEK)
#             if prob.status != 'infeasible':
#                 conv_comb_points.append(i) 
#     return conv_comb_points

# def compute_prob_add_sigmoid(bound_train):
#     slope = 15 # Hardcoded value for now...
#     if bound_train > 0: 
#         x = bound_train 
#     else:
#         x = - bound_train
#     return 1- (1 / (1 + math.exp(-slope*x)))   

# #### For CVaR, substitute slope = np.array([1/(1-beta),0]) and const = np.array([0,1])
# #### Choose phi_conj from the phi-divergence file
# def af_RC_exp_pmin(p,R,r,phi_conj,slope,const):  
#     N = len(p)
#     I = len(R[0])
#     K = len(slope)
#     theta = cp.Variable(1)
#     lbda = cp.Variable((N,K), nonneg = True)
#     a = cp.Variable(I)
#     v = cp.Variable(K, nonneg = True)
#     alpha = cp.Variable(1)
#     beta = cp.Variable(1)
#     gamma = cp.Variable(1,nonneg = True)
#     t = cp.Variable(N)
#     s = cp.Variable(N)
#     w = cp.Variable(N)
#     constraints = [a >= 0, cp.sum(a) == 1]
#     for i in range(N):
#         constraints.append(-(R @ a)[i] - cp.sum(lbda[i]) - beta <= 0)
#         constraints.append(s[i] == -alpha + lbda[i]@slope)
#         constraints.append(lbda[i] <= v)
#         constraints = phi_conj(gamma,s[i],t[i],w[i],constraints)
#     constraints.append(alpha + beta + gamma * r  + v@const + p@t <= -theta)
#     obj = cp.Maximize(theta)
#     prob = cp.Problem(obj,constraints)
#     prob.solve(solver=cp.SCS)
#     return(a.value, prob.value)

































