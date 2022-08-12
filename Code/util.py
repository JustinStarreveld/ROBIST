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
