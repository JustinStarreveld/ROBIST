import numpy as np
import cvxpy as cp
import mosek
import time
import scipy.stats
#from sklearn.model_selection import train_test_split

import phi_divergence as phi
#import robust_sampling as rs
import itertools as itertools
#import dataio
#import util


def make_center(lb,up,m):
    delta = (up-lb)/m
    center = np.arange(lb+delta/2,up,delta)
    return(center)          

def get_freq(cpt,data,lb,up):
    Freq = np.zeros(len(cpt))
    delta = (up-lb)/len(cpt)
    for i in range(len(data)):
        index = int(np.floor((data[i]-lb)/delta))
        Freq[index] = Freq[index] + 1
    return(Freq/len(data))

def solve_rc(Omega,a,b):
    d = len(a[0])
    x = cp.Variable(d, nonneg = True)
    z = cp.Variable(d)
    w = cp.Variable(d)
    constraints = [cp.norm(z,1)+Omega*cp.norm(w,2) + a[0] @ x <= b]
    for i in range(d):
        constraints.append(z[i] + w[i] == -a[i+1] @ x) 
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.MOSEK)
    return(x.value)

def lower_bound(alpha,p,S,N):
    N_v = len(p)
    q = cp.Variable(N_v, nonneg = True)
    t = cp.Variable(N_v, nonneg = True)
    r = 1/N*scipy.stats.chi2.ppf(1-alpha, N_v-1)
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
    return(prob.value, q.value)
    
def cpt_feas(cpt_arr,x,a,b,indices):
    d = len(cpt_arr)
    S = np.zeros(len(indices))
    for i in range(len(S)):
        const = a[0]
        for j in range(d):
            const = const + cpt_arr[j][indices[i][j]] * a[j+1] 
        #print(const.dot(x))
        if const.dot(x) <= b:
            S[i] = 1
    return(S)
    
def Ishan(dim_x,alpha,beta,N,Omega,step_size,a,b, random_seed=0):
    np.random.seed(random_seed) 
    xi = np.random.uniform(size = (dim_x,N))*2-1
    # xi = np.random.uniform(size = (dim_x,N))*0-0.5
    cpt_arr = []
    lb = -1
    up = 1
    m = 10
    indices = np.asarray(list((itertools.product(np.arange(m), repeat = dim_x))))
    p = np.zeros(len(indices))
    freq_ct = []
    for i in range(dim_x):
        cpt_arr.append(make_center(lb,up,m))
        freq_ct.append(get_freq(cpt_arr[i], xi[i],-1,1))
        # if i == 0:
        #     freq_ct.append(np.array([0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05]))
        # else:
        #     freq_ct.append(np.array([0.025, 0.075, 0.2, 0.15, 0.05, 0.125, 0.175, 0.1, 0.075, 0.025]))
        
    #print(cpt_arr[0])
    #print(freq_ct)
    for j in range(len(indices)):
        p[j] = 1
        for k in range(dim_x):
            p[j] = p[j] * freq_ct[k][indices[j][k]]
    
    lowerbound = -np.inf
    while lowerbound < beta:
        x = solve_rc(Omega,a,b)
        #print(x)
        S = cpt_feas(cpt_arr,x,a,b,indices)
        lowerbound,q = lower_bound(alpha,p,S,(N**dim_x))
        print('Omega', Omega)
        print('lowerbound',lowerbound)
        print('objective', np.sum(x))
        print()
        Omega = Omega + step_size
        
random_seed = 1234
a = [np.array([1,1]), np.array([1,0]), np.array([0,1])]
b = 10
Omega = 0.7
step_size = 0.01
alpha = 0.001
beta = 0.95               #### lower bound too low? S is mostly 1
dim_x = 2
N = 100
Ishan(dim_x,alpha,beta,N,Omega,step_size,a,b,random_seed=random_seed)          
