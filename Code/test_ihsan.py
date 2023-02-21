# import external packages
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import time
import math

# import internal packages
import phi_divergence as phi
from iter_gen_and_eval_alg import iter_gen_and_eval_alg
import util

def solve_with_ihsan2013(dim_x,risk_param_epsilon,conf_param_alpha,N,
                         Omega_init=0.1,step_size=0.01,random_seed=0):
    from phi_divergence import mod_chi2_cut
    import scipy.stats
    import itertools

    def generate_data(random_seed, N, **kwargs):
        np.random.seed(random_seed)
        dim_x = kwargs.get('dim_x',2)
        data = np.random.uniform(-1,1,size = (N,dim_x)) # generates N random scenarios    
        return data 

    def make_center(lb,ub,m):
        delta = (ub-lb)/m
        center = np.arange(lb+delta/2,ub,delta)
        return(center)          

    def get_freq(m,data,lb,ub):
        Freq = np.zeros(m)
        delta = (ub-lb)/m
        for i in range(len(data)):
            index = int(np.floor((data[i]-lb)/delta))
            Freq[index] = Freq[index] + 1
        return(Freq/len(data))
    
    def get_freq_v2(data,lb,ub,m,indices):
        dim_x = len(data[0])
        num_cells = len(indices)
        Freq = np.zeros(num_cells)
        delta = (ub-lb)/m
        
        for i in range(len(data)):
            ind = 0
            for j in range(dim_x):
                index_j = int(np.floor((data[i][j]-lb)/delta))
                ind += m**(dim_x - 1 - j) * index_j
            Freq[ind] += 1
        return(Freq/len(data))

    def solve_rc(Omega,a,b):
        d = len(a[0])
        x = cp.Variable(d, nonneg = True)
        z = cp.Variable(d)
        w = cp.Variable(d)
        constraints = [cp.norm(z,1)+Omega*cp.norm(w,2) + a[0] @ x <= b]
        for i in range(d):
            constraints.append(z[i] + w[i] == -a[i+1] @ x) 
        
        # add our additional constraints
        constraints.append(cp.sum(x[0:(d-1)]) <= x[d-1]-1)
        constraints.append(x<=10)
        
        obj = cp.Maximize(cp.sum(x))
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.MOSEK)
        return(x.value)

    def lower_bound(alpha,p,S,N,phi_dot=2):
        N_v = len(p)
        q = cp.Variable(N_v, nonneg = True)
        t = cp.Variable(N_v, nonneg = True)
        r = phi_dot/(2*N)*scipy.stats.chi2.ppf(1-alpha, N_v-1)
        constraints = [cp.sum(q) == 1]
        f_obj = 0
        for i in range(N_v):
            if S[i] == 1:
                f_obj = f_obj + q[i]
            z = cp.vstack([2*(q[i]-p[i]),(t[i]-q[i])])
            constraints.append(cp.norm(z,2) <= (t[i]+q[i]))
        constraints.append(cp.sum(t) <= r)
        obj = cp.Minimize(f_obj)
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.MOSEK)
        return(prob.value)
        
    def cpt_feas(cpt_arr,x,a,b,indices):
        d = len(cpt_arr)
        S = np.zeros(len(indices))
        for i in range(len(S)):
            const = a[0]
            for j in range(d):
                const = const + cpt_arr[j][indices[i][j]] * a[j+1] 
            if const.dot(x) <= b:
                S[i] = 1
        return(S)
    
    def lower_bound_ROBIST(data, x, conf_param_alpha, phi_div=mod_chi2_cut, phi_dot=2, numeric_precision=1e-6):
        N = len(data)
        constr_evals = (np.dot(data,x)) - 1
        N_vio = sum(constr_evals>(0+numeric_precision))
        p_vio = N_vio/N
        if p_vio == 0:
            return 1
        elif p_vio == 1:
            return 0
        return util.compute_cc_lb_chi2_analytic(1-p_vio, N, conf_param_alpha)
    
    def get_true_prob(x, dim_x):
        return(1/2+1/(2*x[dim_x-1]))
    
    # see notation from ihsan2013
    # a = [np.array([1,1]), np.array([1,0]), np.array([0,1])]
    # b = 10
    a = []
    for i in range(dim_x+1):
        if i == 0:
            a.append(np.array([0 for j in range(dim_x)]))
        else:
            temp = [0 for j in range(i-1)]
            temp.append(1)
            temp = temp + [0 for j in range(i, dim_x)]
            a.append(np.array(temp))
    b = 1
    
    data = generate_data(random_seed, N, dim_x=dim_x)
    
    cpt_arr = []
    lb = -1
    ub = 1
    m = 10 # assume that the support is always split into 10 equal intervals, even as dim_x increases
    
    # OLD CODE: Assumes independence
    # np.random.seed(random_seed) 
    # xi = np.random.uniform(size = (dim_x,N))*2-1
    # N = N**dim_x # assume data is indep and all combinations are taken
    # # to get all possible combinations of independent data:
    # data = np.array(np.meshgrid(*xi)).T.reshape(-1,dim_x)
    # indices = np.asarray(list((itertools.product(np.arange(m), repeat = dim_x))))
    # p = np.zeros(len(indices))
    # freq_ct = []
    # for i in range(dim_x):
    #     cpt_arr.append(make_center(lb,ub,m))
    #     freq_ct.append(get_freq(m, data.T[i],lb,ub))
    # for j in range(len(indices)):
    #     p[j] = 1
    #     for k in range(dim_x):
    #         p[j] = p[j] * freq_ct[k][indices[j][k]]
    
    for i in range(dim_x):
        cpt_arr.append(make_center(lb,ub,m))
        
    indices = np.asarray(list((itertools.product(np.arange(m), repeat = dim_x))))
    p = get_freq_v2(data,lb,ub,m,indices)
    
    start_time = time.time()
    Omega = Omega_init
    lowerbound = -np.inf
    while lowerbound < 1-risk_param_epsilon:
        x = solve_rc(Omega,a,b)
        S = cpt_feas(cpt_arr,x,a,b,indices)
        lowerbound = lower_bound(conf_param_alpha,p,S,N)
        
        # add our method for deriving lbs
        lb_robist = lower_bound_ROBIST(data, x, conf_param_alpha)
        
        obj = np.sum(x)
        print('Omega:', Omega)
        print('True Prob:',get_true_prob(x, dim_x))
        print('lowerbound Ihsan:',lowerbound)
        print('lowerbound ROBIST:',lb_robist)
        print('Objective:', obj)
        print()
        Omega = Omega + step_size

    runtime = time.time() - start_time

    return data, runtime, x, obj, lowerbound

dim_x = 3
N = 100**dim_x
risk_param_epsilon = 0.1
conf_param_alpha = 0.001

data, runtime, x, obj, lowerbound = solve_with_ihsan2013(dim_x,risk_param_epsilon,conf_param_alpha,N,Omega_init=0.5,step_size=0.025,random_seed=0)

def solve_toyproblem_true_prob(beta, dim_x):
    x = cp.Variable(dim_x, nonneg = True)
    constraints = [(1-2*beta)*x[dim_x-1] + 1 >= 0, cp.sum(x[0:(dim_x-1)]) <= x[dim_x-1]-1, x<=10]
    # constraints = [(1-2*beta)*x[dim_x-1] + 1 >= 0, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)

x_opt, obj_opt =  solve_toyproblem_true_prob(1-risk_param_epsilon, dim_x)

print("Optimal obj:", obj_opt)





