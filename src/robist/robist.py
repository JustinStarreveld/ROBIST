# -*- coding: utf-8 -*-
"""
Created on Jan 25 2023

@author: Justin Starreveld: j.s.starreveld@uva.nl
@author: Guanyu Jin: g.jin@uva.nl
"""
# import external packages
import numpy as np
import scipy
import time
import math

class Robist:
    """
    ROBIST: Robust Optimization by Iterative Scenario Sampling and Statistical Testing
    
    Iterative generation and evaluation algorithm for uncertain convex 
    optimization problems
      
    INPUT:
    solve_SCP: function
        Solves the sampled convex program (SCP) for some set of sampled scenarios. 
        Returns the optimal primal (and dual) solution, along with the objective of the solved problem.
    problem_instance: dictionary
        Provides additional parametric information needed for solving the particular problem instance.
    eval_unc_obj: dictionary
        If None, there is no uncertainty in the objective of the problem.
        Else, this input argument provides the uncertain objective function to be
        evaluated along with additional information.
    eval_unc_constr: list of dictonaries
        If None, there is no uncertainty in the constraints of the problem.
        Else, this input argument provides the uncertain constraints to be
        evaluated along with additional information.
    data_train: list containing numpy.ndarray's
        Provides the training data set.
    data_test: list containing numpy.ndarray's
        Provides the test data set.
    conf_param_alpha: float
        The desired confidence level for the statistical confidence interval
        used in the evaluation of solutions, default: 0.05
    add_strategy : string
        Specifies the scenario addition strategy to be used, default: 'random_vio'.
    remove_strategy : string
        Specifies the scenario removal strategy to be used, default: 'random_any'.
    use_dual_sol: boolean
        True if one wishes to use dual solution information to speed up the algorithm
        False if not, default: False   
    use_tabu: boolean
        True if one wishes to use tabu lists in the removal and addition of scenarios,
        False if not, default: False
    numeric_precision: float
        The numeric precision to be used to compensate for floating-point math operations
        default: 1e-6
    verbose: boolean
        Specifies whether additional information should be printed, default: False
        
    OUTPUT:
    best_solution: dict
        All information regarding best found solution during the search
    runtime: float
        Total runtime of algorithm (in seconds)
    num_iter: dict
        Number of add & remove iterations performed
    non_dominated_solutions: list of tuples containing (obj, [feas_cert_constr_1, feas_cert_constr_2, ...])
        If the problem has at least 1 uncertain constraint, list of tuples containing non-dominated solutions 
        found during search. Else, None
    S_history: list of lists
        List that tracks the scenario sets used at each iteration
    """  
    
    def __init__(self, solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                 data_train, data_test, conf_param_alpha=0.05,
                 add_strategy='random_vio', rem_strategy='random_any', use_dual_sol=False,
                 use_tabu=False, numeric_precision=1e-6, verbose=False):
        
        self.solve_SCP = solve_SCP
        self.problem_instance = problem_instance
        self.eval_unc_obj = eval_unc_obj
        self.eval_unc_constr = eval_unc_constr
        self.data_train = data_train
        self.data_test = data_test
        self.conf_param_alpha = conf_param_alpha
        self.add_strategy = add_strategy
        self.rem_strategy = rem_strategy
        self.use_dual_sol = use_dual_sol
        problem_instance['get_dual_sol'] = use_dual_sol
        self.use_tabu = use_tabu
        self.numeric_precision = numeric_precision
        self.verbose = verbose
        
        # In case obj is uncertain, determine N2_min if not provided
        if eval_unc_obj is not None:
            eval_unc_obj_info = eval_unc_obj['info']
            if eval_unc_obj_info['risk_measure'] == 'probability' and 'N2_min' not in eval_unc_obj_info.keys():
                N2 = len(data_test)
                desired_rhs = eval_unc_obj_info.get('desired_rhs')
                N2_min = self._determine_N_min(N2, 1-desired_rhs)
                self.eval_unc_obj['info']['N2_min'] = N2_min
        

    """
    init_S: list
        Specifies the set of scenarios with which to initialize S
        default: None
    stop_criteria: dict
        Specifies the stopping criteria to be used.
        Can specify: 'max_elapsed_time', 'max_num_iterations', 'obj_stop'
        default: 'max_elapsed_time': 5 minutes
    store_all_solutions: boolean
        Specifies whether all found solutions should be returned or only the best,
        default: False
    random_seed: int
        Specifies random seed to be set at start of algorithm run, default: 0
    """
    def run(self, init_S=None, stop_criteria={'max_elapsed_time': 5*60}, 
            store_all_solutions=False, random_seed=0):
        start_time = time.time()
        
        # create variables to store info
        best_solution = None
        num_iter = {'add':0, 'remove':0}
        non_dominated_solutions = []
        S_history = []
        if store_all_solutions:
            all_solutions = []
        
        desired_constr_cert_rhs = []
        if self.eval_unc_constr is not None:
            for unc_constr_i in self.eval_unc_constr:
                desired_constr_cert_rhs.append(unc_constr_i['info']['desired_rhs'])
        
        if self.eval_unc_obj is not None:
            desired_obj_cert_rhs = self.eval_unc_obj['info']['desired_rhs']
            desired_cert_rhs = [desired_obj_cert_rhs] + desired_constr_cert_rhs
        else:
            desired_cert_rhs = desired_constr_cert_rhs
        
        # initialize algorithm
        np.random.seed(random_seed) # set seed for random strategies
        skip_solve = False
        
        if init_S is not None:
            S = init_S
            S_indices = [len(self.data_train)+1+i for i in range(len(init_S))]
        else:  
            # randomly pick a scenario from training data
            init_scen = np.random.randint(0, len(self.data_train)-1)
            S = [self.data_train[init_scen]] # assume first index contains nominal data
            S_indices = [init_scen] # tracks the indices of the scenarios in Z
        
        count_iter = 0
        while True:
            
            # check if stopping criteria has been met
            elapsed_time = time.time() - start_time
            num_iterations = sum(v for k,v in num_iter.items())
            if self._stopping_cond(stop_criteria, elapsed_time=elapsed_time, 
                                   num_iterations=num_iterations, 
                                   sol_info=[best_solution, desired_constr_cert_rhs]):
                break
            count_iter += 1
            
            S_history.append(S_indices.copy())
            if skip_solve == False:   
                # generate solution by solving sampled convex program for some set of scenarios S
                if self.use_dual_sol:
                    x_i, obj_scp, duals_i = self.solve_SCP(S, **self.problem_instance)
                else:
                    x_i, obj_scp = self.solve_SCP(S, **self.problem_instance)
                obj_i = obj_scp
                
                # determine certificates for generated solution
                feas_certificates_train = []
                feas_certificates_test = []
                evals_train = []
                if self.eval_unc_obj is not None: 
                    # get feasibility certificate on train data
                    obj_feas_cert, evals = self._compute_obj_feas_certificate(x_i, obj_i, self.data_train)
                    feas_certificates_train.append(obj_feas_cert)
                    evals_train.append(evals)
                    
                    # replace obj with obj certificate from test data
                    obj_cert = self._compute_obj_certificate(x_i, self.data_test)
                    obj_i = obj_cert
                
                # loop through unc constraints and compute feasibility certificates
                if self.eval_unc_constr is not None:
                    for unc_constr_i in self.eval_unc_constr:
                        # get feasibility certificate on train data
                        feas_cert_constr_i, evals = self._compute_constr_feas_certificate(x_i, self.data_train, unc_constr_i)
                        feas_certificates_train.append(feas_cert_constr_i)
                        evals_train.append(evals)
                        
                        # get feasibility certificate on test data
                        feas_cert_constr_i, evals = self._compute_constr_feas_certificate(x_i, self.data_test, unc_constr_i)
                        feas_certificates_test.append(feas_cert_constr_i)
            
                # check whether best solution can be replaced
                best_solution, update_best_yn = self._update_best_solution(best_solution, x_i, obj_i, feas_certificates_test, desired_constr_cert_rhs)
                
                # update pareto frontier          
                if self.eval_unc_constr is not None:
                    non_dominated_solutions = self._update_non_dominated_solutions(non_dominated_solutions, x_i, obj_i, feas_certificates_test)
            
            # optional: print info each iteration
            if self.verbose:
                print("-----------------")
                print("iter     : " + f'{round(count_iter,0):.0f}')
                if self.verbose and update_best_yn and skip_solve==False:
                    print("New best solution found!")
                print("S_ind    : [", *S_indices, "]")
                # print("S_vals   : [", *S, "]")
                # if self.use_dual_sol:
                #     print("duals    : [", *duals_i, "]")
                print("size_S   : " + f'{round(len(S),0):.0f}')
                print("obj_S    : " + f'{round(obj_scp,3):.3f}')
                if self.eval_unc_obj is not None:
                    print("obj_test : " + f'{round(obj_cert,3):.3f}')
                if len(feas_certificates_test) > 0:
                    for i in range(len(feas_certificates_test)):
                        print("train_cert_con_"+str(i)+" : " + f'{round(feas_certificates_train[i],3):.3f}')
                        print("test_cert_con_"+str(i)+" : " + f'{round(feas_certificates_test[i],3):.3f}')
            
            if store_all_solutions:
                all_solutions.append({'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test})
            
            # given the training feasibility certificates, we now determine the next action
            if self.use_tabu == True: 
                tabu_add = self._get_tabu_add(S_indices, S_history)
                tabu_remove = self._get_tabu_remove(S_indices, S_history)
            else:
                tabu_add = set()
                tabu_remove = set()
            
            # determine whether to add or remove scenarios using only the training data evaluations        
            possible_add_ind = self._get_possible_additions(evals_train, tabu_add)
            possible_rem_ind = self._get_possible_removals(S_indices, tabu_remove)
            
            add_or_remove = self._determine_action(feas_certificates_train, desired_cert_rhs, 
                                                   len(possible_add_ind), len(possible_rem_ind))
            
            if add_or_remove is None:
                break # signifies that no more actions are possible
            elif add_or_remove == True:
                if len(evals_train) > 1:
                    #TODO: add code to handle multiple uncertain functions
                    ...
                else:
                    S, S_indices = self._add_scenario(S, S_indices, possible_add_ind) 
                skip_solve = False
                num_iter['add'] += 1
            elif add_or_remove == False:                
                if len(evals_train) > 1:
                    #TODO: add code to handle multiple uncertain functions
                    ...
                else:
                    S, S_indices, i_removal = self._remove_scenario(S, S_indices, possible_rem_ind)
                    if self.use_dual_sol:
                        # check if removed scenario has dual==0
                        if duals_i[i_removal] > 0-self.numeric_precision and duals_i[i_removal] < 0+self.numeric_precision:
                            skip_solve = True
                            del duals_i[i_removal]
                        else:
                            skip_solve = False
            
                num_iter['remove'] += 1
        
            if self.verbose:
                if add_or_remove:
                    print("Decided to add a scenario to S")
                else:
                    print("Decided to remove a scenario from S")
            
        runtime = time.time() - start_time
        
        if store_all_solutions:
            return best_solution, runtime, num_iter, non_dominated_solutions, S_history, all_solutions
        
        return best_solution, runtime, num_iter, non_dominated_solutions, S_history
        
    def _compute_constr_feas_certificate(self, x, data, unc_constr_i):
        eval_constr_func = unc_constr_i['function']
        eval_constr_info = unc_constr_i['info']
        
        if eval_constr_info.get('risk_measure') == 'probability':
            N = len(data)
            constr_evals = eval_constr_func(x, data, **self.problem_instance)
            N_vio = sum(constr_evals>(0+self.numeric_precision))
            p_vio = N_vio/N
            return self._compute_phi_div_bound(p_vio, N), constr_evals
        elif eval_constr_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None, None
        else:
            print("ERROR: do not recognize risk measure")
            return None, None   
       
    def _compute_obj_feas_certificate(self, x, obj, data):
        eval_obj_func = self.eval_unc_obj['function']
        eval_obj_info = self.eval_unc_obj['info']
        
        if eval_obj_info.get('risk_measure') == 'probability':
            N = len(data)
            obj_evals = eval_obj_func(x, data, **self.problem_instance)
            constr = obj_evals - obj
            N_vio = sum(constr>(0+self.numeric_precision))
            p_vio = N_vio/N
            return self._compute_phi_div_bound(p_vio, N), constr
        elif eval_obj_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None, None
        else:
            print("ERROR: do not recognize risk measure")
            return None, None
        
    def _compute_obj_certificate(self, x, data):
        """
        Returns the "objective certificate" for a given solution x. This is the
        best obj claim that can be made given the data and risk tolerance.
        """
        eval_obj_func = self.eval_unc_obj['function']
        eval_obj_info = self.eval_unc_obj['info']
        
        if eval_obj_info.get('risk_measure') == 'probability':
            # evaluate function and sort in ascending order
            obj_evals = eval_obj_func(x, data, **self.problem_instance)
            obj_evals_sorted = np.sort(obj_evals)
            
            # grab N2_min-th largest eval
            N2_min = eval_obj_info.get('N2_min') 
            theta_min = obj_evals_sorted[N2_min - 1]
            return theta_min
        elif eval_obj_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None
        else:
            print("ERROR: do not recognize risk measure")
            return None
            
    def _compute_phi_div_bound(self, p_vio, N):
        if p_vio == 0:
            return 1
        elif p_vio == 1:
            return 0
        return self._compute_mod_chi2_lowerbound(1-p_vio, N, self.conf_param_alpha)
    
    def _compute_mod_chi2_lowerbound(self, p_feas, N, conf_param_alpha):
        r = 1/N*scipy.stats.chi2.ppf(1-conf_param_alpha, 1)
        q_feas = max(p_feas - math.sqrt(p_feas*(1-p_feas)*r),0)
        return q_feas
    
    def _update_best_solution(self, best_solution, x_i, obj_i, feas_certificates_test, desired_constr_cert_rhs):
        update_best_yn = False
        if best_solution is None:
            best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
            update_best_yn = True
        else:
            # first check if feasible, if both, then compare obj
            # if both infeasible, then take sol with smallest largest gap
            if len(feas_certificates_test) > 0:
                best_is_feas = all(best_solution['feas'][i] >= val for i,val in enumerate(desired_constr_cert_rhs))
                x_i_is_feas = all(feas_certificates_test[i] >= val for i,val in enumerate(desired_constr_cert_rhs))
            else:
                best_is_feas = True
                x_i_is_feas = True
            
            if (best_is_feas==False and x_i_is_feas):
                best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                update_best_yn = True
            elif (best_is_feas and x_i_is_feas):
                if (obj_i < best_solution['obj']):
                    best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                    update_best_yn = True
            elif (best_is_feas==False and x_i_is_feas==False):  
                max_gap_x_i = np.max(np.subtract(desired_constr_cert_rhs, feas_certificates_test))
                max_gap_best = np.max(np.subtract(desired_constr_cert_rhs, best_solution['feas']))
                if (max_gap_x_i < max_gap_best):
                    best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                    update_best_yn = True
    
        return best_solution, update_best_yn
    
    def _update_non_dominated_solutions(self, non_dominated_solutions, x_i, obj_i, feas_certificates_test):
        x_i_pareto_eff_yn = True
        indices_to_be_removed = []
        for j,(obj, li_feas_cert) in enumerate(non_dominated_solutions):
            if (obj <= obj_i and 
                all(li_feas_cert[i] >= feas_certificates_test[i] for i in range(len(li_feas_cert)))):
                x_i_pareto_eff_yn = False
            elif ((obj >= obj_i and all(li_feas_cert[i] <= feas_certificates_test[i] for i in range(len(li_feas_cert))))
                    and (obj > obj_i or any(li_feas_cert[i] < feas_certificates_test[i] for i in range(len(li_feas_cert))))):
                indices_to_be_removed.append(j)
                  
        for index in sorted(indices_to_be_removed, reverse=True):
            del non_dominated_solutions[index]
        if x_i_pareto_eff_yn:
            non_dominated_solutions.append((obj_i, feas_certificates_test))
    
        return non_dominated_solutions
    
    def _get_tabu_add(self, S_current, S_past):
        tabu_add = set()
        for S in S_past:
            if len(S) == len(S_current) + 1:
                if all(i in S for i in S_current):
                    tabu_add.add([i for i in S if i not in S_current])
        return tabu_add

    def _get_tabu_remove(self, S_current, S_past):
        tabu_remove = set()
        for S in S_past:
            if len(S) == len(S_current) - 1:
                if all(i in S_current for i in S):
                    tabu_remove.add([i for i in S_current if i not in S])
        return tabu_remove

    def _get_possible_additions(self, evals_train, tabu_add):
        possible_add_ind = set()
        for i in range(len(evals_train)):
            vio_i = np.argwhere(evals_train[i]>(0+self.numeric_precision))
            if len(vio_i) > 1:
                possible_add_ind.update(vio_i.squeeze())
            
        # remove tabu indices
        possible_add_ind = possible_add_ind - tabu_add
        return possible_add_ind

    def _get_possible_removals(self, S_indices, tabu_remove):
        if len(S_indices) == 1:
            return set()
        possible_rem_ind = set(S_indices) - tabu_remove
        return possible_rem_ind

    def _determine_action(self, feas_certificates_train, desired_cert_rhs, num_possible_additions, num_possible_removals):    
        # Determines whether it will be an add (True) or remove (False) or break (None) 
        if num_possible_additions == 0 and num_possible_removals == 0:
            return None
        elif num_possible_additions == 0:
            return False
        elif num_possible_removals == 0:
            return True
        
        threshold = self._compute_prob_add(feas_certificates_train, desired_cert_rhs)
        draw = np.random.uniform()
        if draw < threshold:
            return True
        else:
            return False
    
    def _compute_prob_add(self, feas_certificates_train, desired_cert_rhs, method='deterministic_w_x%'):
        """
        Returns the probability that an addition should take occur.
        This probability is dependent on feasibility cerficates derived from training data
        and can be derived in different ways, this is specified by the "method" argument.
        """
        if method == 'deterministic':
            if all(feas_certificates_train[i] >= desired_cert_rhs[i] for i in range(len(desired_cert_rhs))):
                return 0
            else:
                return 1
        elif method == 'deterministic_w_x%':
            x = 0.01 #default
            if all(feas_certificates_train[i] >= desired_cert_rhs[i] for i in range(len(desired_cert_rhs))):
                return x
            else:
                return 1-x
        else:
            print('Error: do not recognize method in "compute_prob_add" function')
            return 1

    def _add_scenario(self, S, S_indices, possible_add_ind):
        if self.add_strategy == 'random_vio':
            ind = np.random.choice(list(possible_add_ind))
            possible_add_ind.remove(ind)
        else:
            print("Error: do not recognize addition strategy")
            return None
        S_indices.append(ind)
        scen_to_add = self.data_train[ind]
        if len(S) > 0:
            S.append(scen_to_add)
        else:
            S = [scen_to_add]
        return S, S_indices

    def _remove_scenario(self, S, S_indices, possible_rem_ind):
        if self.rem_strategy == 'random_any':
            scen_to_remove = np.random.choice(list(possible_rem_ind))
            while scen_to_remove > len(self.data_train):
                scen_to_remove = np.random.choice(list(possible_rem_ind))
            i_removal = S_indices.index(scen_to_remove)
            del S[i_removal]
            # S = [e for i,e in enumerate(S) if i != i_removal]
            del S_indices[i_removal]
        else:
            print("Error: do not recognize removal strategy")
            return None
        
        return S, S_indices, i_removal
    
    def _stopping_cond(self, stop_criteria, **kwargs):
        """
        Returns true if a stopping condition is met, else False.
        """
        if 'obj_stop' in stop_criteria:
            best_sol, desired_constr_cert_rhs = kwargs['sol_info']
            if best_sol is None:
                return False
            if len(desired_constr_cert_rhs) > 0:
                best_is_feas = all(best_sol['feas'][i] >= val for i,val in enumerate(desired_constr_cert_rhs))
                if best_is_feas and best_sol['obj'] <= stop_criteria['obj_stop']:
                    return True
            elif best_sol['obj'] <= stop_criteria['obj_stop']:
                return True
        if (kwargs.get('elapsed_time',0) >= stop_criteria.get('max_elapsed_time', 10e12) 
            or kwargs.get('num_iterations',0) >= stop_criteria.get('max_num_iterations', 10e12)):
            return True
        else:
            return False   

    def _determine_N_min(self, N, risk_param_epsilon):
        p_feas_min = self._determine_p_feas_min(N, risk_param_epsilon)
        N_min = math.ceil(p_feas_min * N)
        return N_min

    def _determine_p_feas_min(self, N, risk_param_epsilon):
        # golden section search in interval (desired_prob_rhs, 1)
        gr = (math.sqrt(5) + 1) / 2
        tol = 1e-5
        desired_prob_rhs = 1-risk_param_epsilon
        a = desired_prob_rhs
        b = 1 - 1/N
        
        if self._compute_phi_div_bound(1-b, N) < desired_prob_rhs:
            return 1
        
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        while abs(b - a) > tol:
            f_c = abs(self._compute_phi_div_bound(1-c, N) - desired_prob_rhs)
            f_d = abs(self._compute_phi_div_bound(1-d, N) - desired_prob_rhs)
            
            if f_c < f_d: 
                b = d
            else:
                a = c
    
            # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr
    
        return (b + a) / 2
