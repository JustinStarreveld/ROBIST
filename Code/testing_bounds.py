import numpy as np
import scipy.stats
import math
import time

def compute_mod_chi2_lowerbound(p_feas, N, conf_param_alpha):
    r = 1/N*scipy.stats.chi2.ppf(1-conf_param_alpha, 1)
    q_feas = max(p_feas - math.sqrt(p_feas*(1-p_feas)*r),0)
    return q_feas

# from section 10.2 of Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009)
def compute_bental2009_lowerbound(L, N, conf_param_alpha):
    # from scipy.special import comb
    from fractions import Fraction
    from decimal import Decimal

    def choose(n,k):
        if k > n//2: k = n - k
        p = Fraction(1)
        for i in range(1,k+1):
            p *= Fraction(n - i + 1, i)
        return int(p)
    
    # use bisection search to find lowest q
    eps = 1e-5
    a = 0
    b = L/N
    
    comb_terms = []
    for k in range(L, N+1):
        # comb_terms.append(Decimal(choose(N, k)))
        comb_terms.append(Decimal(scipy.special.comb(N,k)))
    
    while True:
        if abs(a-b) < eps:
            return a
        
        c = (a+b)/2
        f_c = sum(comb_terms[k-L] * Decimal(c**k * (1-c)**(N-k)) for k in range(L, N+1))
        if f_c > conf_param_alpha:
            b = c
        else:
            a = c
    

p_settings = [0.8, 0.99]
N_settings = [10, 100, 1000, 10000]
# N_settings = [10000]
alpha_settings = [0.05, 1e-10]  #[0.05, 1e-10]    

np.random.seed(0)

for p in p_settings:
    for N in N_settings:
        
        # L = np.random.binomial(N, p)
        L = int(N * p)
        # print(L)
        
        for conf_param_alpha in alpha_settings:
            start_time = time.time()
            lb_chi2 = format(round(compute_mod_chi2_lowerbound((L/N), N, conf_param_alpha), 4), '.4f')
            time_chi2 = format(round(time.time() - start_time, 1), '.1f')

            start_time = time.time()
            lb_bental = format(round(compute_bental2009_lowerbound(L, N, conf_param_alpha),4), '.4f')
            time_bental = format(round(time.time() - start_time, 1), '.1f')
            
            print(p, N, conf_param_alpha, lb_bental, time_bental, lb_chi2, time_chi2)









