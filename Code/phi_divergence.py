#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cvxpy as cp
import seaborn as sns
import mosek
import matplotlib.pyplot as plt
import datetime as date
from datetime import datetime as dt
from dateutil.relativedelta import *
import scipy.stats
from scipy.stats import rankdata


# In[5]:


######## all phi-conjugates


def mod_chi2_conj(gamma,s,t,w,constraints):
    constraints.append(cp.norm(cp.vstack([w,t/2]))<=(t+2*gamma)/2)
    constraints.append(s/2+gamma<= w)
    return(constraints)

def kb_conj(gamma,s,t,w,constraints):
    constraints.append(w - gamma <= t)
    constraints.append(cp.kl_div(gamma,w)+gamma+s-w<= 0)
    return(constraints)


# In[3]:


################ all phi_cuts_constraints

def mod_chi2_cut(p,q,r,par,constraints):
    N = p.shape[0]
    phi_cons = 0
    for i in range(N):
        phi_cons = phi_cons + 1/p[i]*(q[i]-p[i])**2
    constraints.append(phi_cons<=r)
    return(constraints)

def kb_cut(p,q,r,par,constraints):
    N = p.shape[0]
    phi_cons = 0
    for i in range(N):
        phi_cons = phi_cons -cp.entr(q[i]) - q[i]*np.log(p[i])
    constraints.append(phi_cons<=r)
    return(constraints)


# In[6]:


########### all phi evaluations

def kb_eva(p,q):
    N = len(p)
    phi = 0
    for i in range(N):
        if q[i]<= 0:
            phi = phi + 0
        else:
            phi = phi + q[i]*np.log(q[i]/p[i])
    return(phi)

def mod_chi2_eva(p,q):
    N = len(p)
    phi = 0
    for i in range(N):
        if p[i]== 0:
            return (np.inf)
        phi = phi + (p[i]-q[i])**2/p[i]
    return(phi)

