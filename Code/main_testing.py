# Import packages
import numpy as np
import pandas as pd
import cvxpy as cp
import mosek
import time
import scipy
from sklearn.model_selection import train_test_split

import phi_divergence as phi
import robust_sampling as rs
import dataio
import util

# Problem specific functions:
def generate_data(random_seed, k, N):
    np.random.seed(random_seed)
    data = np.random.uniform(-1,1,size = (N,k)) # generates N random scenarios    
    return data 

def generate_data_with_nominal(random_seed, k, N):
    data_nominal = np.array([[0] * k])
    np.random.seed(random_seed)
    data = np.random.uniform(-1,1,size = (N-1,k)) # generate N-1 scenarios
    data = np.concatenate((data_nominal,data)) # add nominal case to training data
    return data

def solve_SCP(k, S, time_limit):
    #k = S.shape[1]
    x = cp.Variable(k, nonneg = True)
    if len(S) > 0:
        constraints = [(S @ x) - 1 <= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    else:
        constraints = [cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time: time_limit})
    except cp.error.SolverError:
        return (None, None)
    return (x.value, prob.value)

def uncertain_constraint(S, x):
    return (np.dot(S, x)) - 1

def get_true_prob(x, k):
    return(1/2+1/(2*x[k-1]))
    
def solve_toyproblem_true_prob(beta, k):
    x = cp.Variable(k, nonneg = True)
    constraints = [(1-2*beta)*x[k-1] + 1 >= 0, cp.sum(x[0:(k-1)]) <= x[k-1]-1, x<=10]
    obj = cp.Maximize(cp.sum(x))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(x.value, prob.value)

k = 1000
dim_x = k
beta = 0.95
alpha = 10e-6
time_limit_solve = 5*60
numeric_precision = 1e-6

import statistics as s

# Store output in lists
vec_obj = []
vec_true_prob = []
vec_mean_size_S = []
vec_max_size_S = []
vec_num_iter = []
vec_time = []
vec_data = []

set_sizes, time_determine_set_sizes = util.Garatti2022_determine_set_sizes(dim_x, beta, alpha)






