"""
Basic functions for toy problem
"""   
# external imports
import numpy as np
import cvxpy as cp
import time

# problem specific functions:
def get_param_data(random_seed, **kwargs):
    np.random.seed(random_seed)
    dim_x = kwargs.get('dim_x')
    N = kwargs.get('N')
    c = np.random.randint(-10,-1,dim_x)
    bb = np.random.normal(100,0,N)
    b = 100
    a = np.random.randint(1,50,size=(1,N,dim_x,))  
    param_dict = {'c': c, 'a': a, 'b': b, 'bb': bb}
    return param_dict

def generate_unc_param_data(random_seed, **kwargs):
    np.random.seed(random_seed)
    dim_x = kwargs.get('dim_x')
    N = kwargs.get('N')
    a = np.random.randint(1,50,size=(N,dim_x))  
    return a

def solve_SCP(S, **kwargs):
    setup_time_start = time.time()
    dim_x = kwargs.get('dim_x')
    c = kwargs['c']
    b = kwargs['b']
    x = cp.Variable(dim_x, nonneg=True)
    constraints = []
    for s in range(len(S)):
        constraints.append(cp.sum(cp.multiply(S[s], x)) - b <= 0)
    constraints.append(x<=1)
    obj = cp.Minimize(cp.sum(cp.multiply(c, x))) # formulate as a minimization problem
    prob = cp.Problem(obj,constraints)
    time_limit = kwargs.get('time_limit', 5*60) - (time.time() - setup_time_start)
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=time_limit)
    primal_solution = x.value
    obj = prob.value
    if kwargs.get('get_dual_sol', False):
        dual_solution = []
        for s in range(len(S)):
            dual_solution.append(constraints[s].dual_value)
        return primal_solution, obj, dual_solution
    return primal_solution, obj

def unc_function(solution, data, **kwargs):
    return (np.dot(data,solution)) - kwargs['b']

def eval_OoS(solution, data, numeric_precision=1e-6, **kwargs):
    f_evals = unc_function(solution, data, **kwargs)
    N_vio = sum(f_evals>-numeric_precision)
    N = len(data)
    p_feas = 1 - N_vio/N
    return p_feas

def solve_SAA_with_MIO(risk_param_epsilon, data, problem_instance, time_limit=30*60, verbose=False):
    time_start = time.time()
    dim_x = problem_instance['dim_x']
    c = problem_instance['c']
    b = problem_instance['b']
    big_M = b*100
    N = len(data)

    x = cp.Variable(dim_x, nonneg=True)
    y = cp.Variable(N, boolean=True)
    constraints = []
    for s in range(N):
        constraints.append(cp.sum(cp.multiply(data[s], x)) - b <= y[s]*big_M)
    constraints.append(x<=1)
    constraints.append(cp.sum(1-y) >= (1 - risk_param_epsilon) * N)
    obj = cp.Minimize(cp.sum(cp.multiply(c, x))) # formulate as a minimization problem
    prob = cp.Problem(obj,constraints)
    time_limit_solve = time_limit - (time.time() - time_start)
    prob.solve(solver=cp.GUROBI, verbose=verbose, TimeLimit=time_limit_solve)
    sol = x.value
    obj = prob.value
    runtime = time.time() - time_start
    return runtime, obj, sol


# This part of the code is a replication of https://github.com/jnan97/ALSO-X/blob/main/ALSOX_code/Experiment%201/Linear%20CCP.py
import sys
from gurobipy import GRB, Model, quicksum, Env
import numpy as np
import math
import time
from numpy import linalg as LA
def solve_with_jiang2022(risk_param_epsilon, N, problem_instance, time_limit=2*60*60, verbose=False):
    start=time.time()
    a = problem_instance['a']
    c = problem_instance['c']
    bb = problem_instance['bb']

    epsilon = risk_param_epsilon
    random_size = N
    sample_size = 1
    x_random_size = problem_instance['dim_x']

    violation = math.floor(random_size*epsilon)

    def Model_cvar():
        #Model
        if not verbose: 
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            m = Model(env=env)
        else:
            print ("Begin to solve model 4")
            m = Model()
        
        m.setParam('TimeLimit', time_limit)

        # Create variables
        m.update()
        x = m.addVars(x_random_size,lb=0,ub=1,name="x")
        s = m.addVars(random_size,lb=-GRB.INFINITY,name="s")
        beta = m.addVar(lb=-GRB.INFINITY,ub=0,name="beta")
        m.update()
        
        # Set functions
        
        #obj = 0
        g_x_xi = [None] * len(s)
        z = [None] * len(s)
        f_x = 0
        g_x = 0
        g_x_k = [None] * sample_size
        
        # Set functions

        for k in range(sample_size):
            g_x_xi = [None] * random_size
            for i in range(random_size):
                g_x = 0
                
                for j in range(x_random_size):
                    g_x += a[k,i,j]*x[j]
                    m.update()
                g_x_xi[i] = g_x
            g_x_k[k]=g_x_xi
        # not necessary
        
        # Add objective    
        m.setObjective(quicksum(c[i]*x[i] for i in range(x_random_size)), GRB.MINIMIZE)
        m.update()
        # Add constraints
    
        m.addConstrs(g_x_k[j][i] <= bb[i] + s[i]  for j in range(sample_size) for i in range(random_size))
        m.addConstrs(s[i] >= beta for i in range(random_size))
        
        m.addConstr(s.sum()/random_size -beta*(1-epsilon) <= 0)
    
        # Solve the problem
        m.update()
            
        m.optimize()

        # Store solutions
        CVAR_s = m.getAttr('x', s)
        CVAR_x = m.getAttr('x', x)
        
        return m.objVal,CVAR_s,CVAR_x

    CVaR=Model_cvar()
    CVAR=CVaR[0]
    CVAR_s=CVaR[1]
    CVAR_x=CVaR[2]

    # quantitle bounds
    def Model_q():
        if not verbose: 
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            m = Model(env=env)
        else:
            print ("Begin to solve model quantitle relaxtion")
            m = Model()
        
        m.setParam('TimeLimit', time_limit)
        
        # Create variables
        x = m.addVars(x_random_size,lb=0,ub=1, name="x")
        m.update()
        
        # Set functions
        
        #obj = 0
        g_x_k = [None] * sample_size
        f_x = 0
        g_x = 0
        
        # Set functions
        for i in range(x_random_size):
        
            f_x += c[i]*x[i]
            m.update()
        
        for k in range(sample_size):
            g_x_xi = [None] * random_size
            for i in range(random_size):
            #   g_x_si[i] = g_x + si[i] 
                g_x = 0
                
                for j in range(x_random_size):
                    g_x += a[k,i,j]*x[j]
                    m.update()
                g_x_xi[i] = g_x
            g_x_k[k]=g_x_xi
        
        # Add objective    
        m.setObjective(quicksum(c[i]*x[i] for i in range(x_random_size)), GRB.MINIMIZE)
        m.update()
        # Add constraints
        

        con1=m.addConstrs(g_x_k[j][0] <= bb[0]  for j in range(sample_size))
        # Solve the problem
        m.update()
        m.params.OutputFlag=0    
        m.optimize()
        kkk = m.getAttr('x', x)
        result = [0]*random_size
        obj = 0
        for i in range(x_random_size):
            obj += c[i]*kkk[i]
        result[0]=obj
        
        
        for i in range(1,random_size):   
            m.remove(con1)
            con1=m.addConstrs( g_x_k[j][i]  <= bb[i] for j in range(sample_size) )
            m.update()
            m.params.OutputFlag=0    
            m.optimize()
            kkk = m.getAttr('x', x)
            obj = 0
            for j in range(x_random_size):
                obj += c[j]*kkk[j]
            result[i]=obj
        return result

    result=Model_q()
    sorted_array=np.sort(result)
    v_q = sorted_array[random_size-violation+1]

    # alsox
    if verbose: print ("Begin to solve alsox")

    # Set initial values
    delta_1=1e-2
    t1 =  v_q
    t2 =  CVAR
    delta_t = t2-t1
    t = (t1+t2)/2.0

    if not verbose: 
        env = Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = Model(env=env)
    else:
        m = Model()
    
    m.setParam('TimeLimit', time_limit)

    # Create variables
    m.update()
    s = m.addVars(random_size,lb=0,name="s")
    x = m.addVars(x_random_size,lb=0,ub=1,name="x")

    m.update()

    # Set functions
    obj = 0
    g_x_xi = [None] * len(s)
    g_x_k = [None] * sample_size
    f_x = 0

    o = [1] * random_size
    # Set functions
    for i in range(random_size):
        obj += s[i]*o[i]
        m.update()

    for i in range(x_random_size):   
        f_x += c[i]*x[i]
        m.update()
        
    for k in range(sample_size):
        g_x_xi = [None] * random_size
        for i in range(random_size):
            #   g_x_si[i] = g_x + si[i] 
            g_x = 0
                
            for j in range(x_random_size):
                g_x += a[k,i,j]*x[j]
                m.update()
            g_x_xi[i] = g_x
        g_x_k[k]=g_x_xi

    # Add objective    
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Add constraints
    m.addConstrs(g_x_k[k][i] <= bb[i] + s[i]  for k in range(sample_size) for i in range(random_size))
    con1=m.addConstr(f_x <= t)
    for i in range(random_size):
        s[i].start = CVAR_s[i]
    for i in range(x_random_size):
        x[i].start = CVAR_x[i] 
    # Solve the problem
    m.params.OutputFlag=0 
    m.optimize()

    # Store solutions
    ii = m.getAttr('x', s)
    kkk = m.getAttr('x', x)
    iteration = 0

    while delta_t >= delta_1: 
        
        support=0
        for j in ii:
            if ii[j]!=0:
                support +=1
        if support > violation:
            t1=con1.SARHSUp
        else:
            t2=con1.SARHSLow
        t=(t1+t2)/2.0
        delta_t=t2-t1
        con1.RHS = t
        for i in range(random_size):
            s[i].start = ii[i]
        for i in range(x_random_size):
            x[i].start = kkk[i] 
        m.params.OutputFlag=0 
        m.optimize()
        iteration+=1
        ii = m.getAttr('x', s)
        kkk = m.getAttr('x', x)
        con1.RHS = t2
        m.params.OutputFlag=0 
        m.optimize()
        ii = m.getAttr('x', s)
        kkk = m.getAttr('x', x)
        modeltime = time.time() - start

        if verbose: 
            print('alsox')
            print(f'The modeltime is {modeltime:.3f} s.')
            print(f'The number of iterations is {iteration:.0f}.')
            print(f'The objective s is {t2:.5f}.')

        return modeltime, iteration, t2, kkk.values()
    
def solve_with_jiang2022_plus(risk_param_epsilon, N, problem_instance, time_limit=2*60*60, verbose=False):
    start=time.time()
    a = problem_instance['a']
    c = problem_instance['c']
    bb = problem_instance['bb']

    epsilon = risk_param_epsilon
    random_size = N
    sample_size = 1
    x_random_size = problem_instance['dim_x']

    violation = math.floor(random_size*epsilon)

    def Model_cvar():
        if not verbose: 
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            m = Model(env=env)
        else:
            print ("Begin to solve model 4")
            m = Model()
        
        m.setParam('TimeLimit', time_limit)

        # Create variables
        m.update()
        x = m.addVars(x_random_size,lb=0,ub=1,name="x")
        s = m.addVars(random_size,lb=-GRB.INFINITY,name="s")
        beta = m.addVar(lb=-GRB.INFINITY,ub=0,name="beta")
        m.update()
        
        # Set functions
        
        #obj = 0
        g_x_xi = [None] * len(s)
        z = [None] * len(s)
        f_x = 0
        g_x = 0
        g_x_k = [None] * sample_size
        
        # Set functions

        for k in range(sample_size):
            g_x_xi = [None] * random_size
            for i in range(random_size):
                g_x = 0
                
                for j in range(x_random_size):
                    g_x += a[k,i,j]*x[j]
                    m.update()
                g_x_xi[i] = g_x
            g_x_k[k]=g_x_xi
        # not necessary
        
        # Add objective    
        m.setObjective(quicksum(c[i]*x[i] for i in range(x_random_size)), GRB.MINIMIZE)
        m.update()
        # Add constraints
    
        m.addConstrs(g_x_k[j][i] <= bb[i] + s[i]  for j in range(sample_size) for i in range(random_size))
        m.addConstrs(s[i] >= beta for i in range(random_size))
        
        m.addConstr(s.sum()/random_size -beta*(1-epsilon) <= 0)
    
        # Solve the problem
        m.update()
            
        m.optimize()

        # Store solutions
        CVAR_s = m.getAttr('x', s)
        CVAR_x = m.getAttr('x', x)
        
        return m.objVal,CVAR_s,CVAR_x

    CVaR=Model_cvar()
    CVAR=CVaR[0]
    CVAR_s=CVaR[1]
    CVAR_x=CVaR[2]

    # quantitle bounds
    def Model_q():
        if not verbose: 
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            m = Model(env=env)
        else:
            print ("Begin to solve model quantitle relaxtion")
            m = Model()
        
        m.setParam('TimeLimit', time_limit)
        m.update()
        
        # Create variables
        x = m.addVars(x_random_size,lb=0,ub=1, name="x")
        m.update()
        
        # Set functions
        
        #obj = 0
        g_x_k = [None] * sample_size
        f_x = 0
        g_x = 0
        
        # Set functions
        for i in range(x_random_size):
        
            f_x += c[i]*x[i]
            m.update()
        
        for k in range(sample_size):
            g_x_xi = [None] * random_size
            for i in range(random_size):
            #   g_x_si[i] = g_x + si[i] 
                g_x = 0
                
                for j in range(x_random_size):
                    g_x += a[k,i,j]*x[j]
                    m.update()
                g_x_xi[i] = g_x
            g_x_k[k]=g_x_xi
        
        # Add objective    
        m.setObjective(quicksum(c[i]*x[i] for i in range(x_random_size)), GRB.MINIMIZE)
        m.update()
        # Add constraints
        

        con1=m.addConstrs(g_x_k[j][0] <= bb[0]  for j in range(sample_size))
        # Solve the problem
        m.update()
        m.params.OutputFlag=0    
        m.optimize()
        kkk = m.getAttr('x', x)
        result = [0]*random_size
        obj = 0
        for i in range(x_random_size):
            obj += c[i]*kkk[i]
        result[0]=obj
        
        
        for i in range(1,random_size):   
            m.remove(con1)
            con1=m.addConstrs( g_x_k[j][i]  <= bb[i] for j in range(sample_size) )
            m.update()
            m.params.OutputFlag=0    
            m.optimize()
            kkk = m.getAttr('x', x)
            obj = 0
            for j in range(x_random_size):
                obj += c[j]*kkk[j]
            result[i]=obj
        return result

    result=Model_q()
    sorted_array=np.sort(result)
    v_q = sorted_array[random_size-violation+1]

    def AM(t,ii,kkk):
        o = [1] * random_size
        delta_d = 10000
        #delta_2 = 1e-3*CVAR
        delta_2 = 1e-3
        value = 1000
        
        if not verbose: 
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            m = Model(env=env)
        else:
            m = Model()
        
        m.setParam('TimeLimit', time_limit)

        # Create variables
        m.update()
        s = m.addVars(random_size,lb=0,name="s")
        x = m.addVars(x_random_size,ub=1,lb=0,name="x")
    
        m.update()
        
        # Set functions
        
        obj = 0
        g_x_xi = [None] * random_size
        g_x_k = [None] * sample_size
        f_x = 0
    # g_x = 0
        # Set functions
        for i in range(random_size):
            obj += s[i]*o[i]
            m.update()

        for i in range(x_random_size):   
            f_x += c[i]*x[i]
            m.update()
        
    #     for i in range(random_size):
    #      #   g_x_si[i] = g_x + si[i] 
    #         g_x = 0
    #         for j in range(x_random_size):
    #             g_x += a[i,j]*x[j]
    #             m.update()
    #         g_x_xi[i] = g_x 
            
        for k in range(sample_size):
            g_x_xi = [None] * random_size
            for i in range(random_size):
            #   g_x_si[i] = g_x + si[i] 
                g_x = 0
                
                for j in range(x_random_size):
                    g_x += a[k,i,j]*x[j]
                    m.update()
                g_x_xi[i] = g_x
            g_x_k[k]=g_x_xi
            
        # Add objective    
        m.setObjective(obj, GRB.MINIMIZE)
        # Add constraints
        m.addConstr(f_x <= t)
        m.addConstrs(g_x_k[k][i] <= bb[i]+s[i] for i in range(random_size) for k in range(sample_size))
        
        for i in range(random_size):
            s[i].start = CVAR_s[i]
        for i in range(x_random_size):
            x[i].start = CVAR_x[i] 
        # Solve the problem
        
        m.params.OutputFlag=0 
        m.optimize()
        
        # Store solutions
        i_1 = m.getAttr('x', s)
        k_1 = m.getAttr('x', x)
        
        iteration_ap = 0
        
        while delta_d >= delta_2 and iteration_ap <= 5 :
            
            iteration_ap += 1 
            value_1 = value
            o = [0] * random_size
            in_arr=np.array(i_1.values())
            out_arr = np.argsort(in_arr) 
            for i in range(int(random_size*(1-epsilon))):
                o[out_arr[i]]=1
                
            value = sum(i_1[i]*o[i] for i in range(random_size))
            
            if  value == 0:
                delta_d = 0
                break
            delta_d = abs( value - value_1 )
            m.NumObj=0
            obj = 0
            for i in range(random_size):
                obj += s[i]*o[i]
                m.update()
            m.setObjective(obj, GRB.MINIMIZE)
            for i in range(random_size):
                s[i].start = i_1[i]
            for i in range(x_random_size):
                x[i].start = k_1[i]  
            m.params.OutputFlag=0 
            m.optimize()

            i_1 = m.getAttr('x', s)
            k_1 = m.getAttr('x', x)
            
        s_l_0 = 0
        for i in range(random_size):
            if i_1[i] != 0:
                s_l_0 +=1

        return s_l_0

    # alsox+
    if verbose: print ("Begin to solve alsox + ")

    delta_1 = 1e-3
    t1 =  v_q
    t2 =  CVAR
    delta_t = t2-t1
    t = (t1+t2)/2.0

    if not verbose: 
        env = Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = Model(env=env)
    else:
        m = Model()
    
    m.setParam('TimeLimit', time_limit)

    # Create variables
    m.update()
    s = m.addVars(random_size,lb=0,name="s")
    x = m.addVars(x_random_size,lb=0,ub=1,name="x")

    m.update()

    # Set functions

    obj = 0
    g_x_xi = [None] * len(s)
    g_x_k = [None] * sample_size
    f_x = 0
    # g_x = 0


    o = [1] * random_size
    # Set functions
    for i in range(random_size):
        obj += s[i]*o[i]
        m.update()

    for i in range(x_random_size):   
        f_x += c[i]*x[i]
        m.update()

    # for i in range(random_size):
    #  #   g_x_si[i] = g_x + si[i] 
    #     g_x = 0
    #     for j in range(x_random_size):
    #         g_x += a[i,j]*x[j]
    #         m.update()
    #     g_x_xi[i] = g_x 
    for k in range(sample_size):
        g_x_xi = [None] * random_size
        for i in range(random_size):
            #   g_x_si[i] = g_x + si[i] 
            g_x = 0
                
            for j in range(x_random_size):
                g_x += a[k,i,j]*x[j]
                m.update()
            g_x_xi[i] = g_x
        g_x_k[k]=g_x_xi
            

    # Add objective    
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    m.addConstrs(g_x_k[k][i] <= bb[i] + s[i] for i in range(random_size) for k in range(sample_size))
    con1=m.addConstr(f_x <= t)
    for i in range(random_size):
        s[i].start = CVAR_s[i]
    for i in range(x_random_size):
        x[i].start = CVAR_x[i] 
    # Solve the problem
    m.params.OutputFlag=0 
    m.optimize()

    # Store solutions
    ii = m.getAttr('x', s)
    kkk = m.getAttr('x', x)
    iteration = 0


    while delta_t >= delta_1: 
        
        support=0
        for j in ii:
            if ii[j]!=0:
                support +=1
        if support <= violation:
            t2=con1.SARHSLow
        else:
            t3=con1.SARHSUp
            t4=con1.SARHSLow
            
            AP_checking=AM(t,ii,kkk)
            if AP_checking<=violation:
                t2=t4
            else:
                t1=t3
        t=(t1+t2)/2
        delta_t=t2-t1
        con1.RHS = t
        for i in range(random_size):
            s[i].start = ii[i]
        for i in range(x_random_size):
            x[i].start = kkk[i]  
        m.params.OutputFlag=0 
        m.optimize()
        iteration+=1
        ii = m.getAttr('x', s)
        kkk = m.getAttr('x', x)
    con1.RHS = t2
    m.params.OutputFlag=0 
    m.optimize()
    ii = m.getAttr('x', s)
    kkk = m.getAttr('x', x)
    modeltime = time.time() - start

    if verbose:
        print('alsox+')
        print(f'The modeltime is {modeltime:.3f}.')
        print(f'The number of iterations is {iteration:.0f}.')
        print(f'The objective s is {t2:.5f}.')

    return modeltime, iteration, t2, kkk.values()

    





