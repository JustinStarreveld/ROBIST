# -*- coding: utf-8 -*-
"""
Created on Jan 25 2023

@author: Justin Starreveld: j.s.starreveld@uva.nl
@author: Guanyu Jin: g.jin@uva.nl
"""
# import external packages
import numpy as np
import time
import math

# import internal packages
import util

class iter_gen_and_eval_alg:
    """
    iterative generation and evaluation algorithm for uncertain convex 
    optimization problems
      
    INPUT:
    solve_SCP: function
        Solves the sampled convex program (SCP) with a set of sampled scenarios. 
        Returns the optimal solution and objective of the solved problem.
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
    use_tabu: boolean
        True if one wishes to use tabu lists in the removal and addition of scenarios,
        False if not, default: False
    numeric_precision: float
        The numeric precision to be used to compensate for floating-point math operations
        default: 1e-6
    random_seed: int
        Specifies random seed to be set at start of algorithm run, default: 0
    verbose: boolean
        Specifies whether additional information should be printed, default: False
        
    OUTPUT:
    best_sol: dict
        All information regarding best found solution during the search
    runtime: float
        Total runtime of algorithm (in seconds)
    num_iter: dict
        Number of add & remove iterations
    pareto_frontier: list of tuples containing (obj, [feas_cert_constr_1, feas_cert_constr_2, ...])
        If the problem has at least 1 uncertain constraint, list of tuples containing pareto efficient 
        frontier found during search. Else, None
    S_history: list of lists
        List that tracks the indices of scenario sets used at each iteration
    """  
    
    def __init__(self, solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                 data_train, data_test, conf_param_alpha=0.05,
                 add_strategy='random_vio', rem_strategy='random_any',
                 use_tabu=False, numeric_precision=1e-6, random_seed=0, 
                 verbose=False):
        
        self.solve_SCP = solve_SCP
        self.problem_instance = problem_instance
        self.eval_unc_obj = eval_unc_obj
        self.eval_unc_constr = eval_unc_constr
        self.data_train = data_train
        self.data_test = data_test
        self.conf_param_alpha = conf_param_alpha
        self.add_strategy = add_strategy
        self.rem_strategy = rem_strategy
        self.use_tabu = use_tabu
        self.numeric_precision = numeric_precision
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Set N2_min if not provided
        if eval_unc_obj is not None:
            eval_unc_obj_info = eval_unc_obj['info']
            if eval_unc_obj_info['risk_measure'] == 'probability' and 'N2_min' not in eval_unc_obj_info.keys():
                N2 = len(data_test)
                desired_rhs = eval_unc_obj_info.get('desired_rhs')
                N2_min = self._determine_N_min(N2, desired_rhs)
                self.eval_unc_obj['info']['N2_min'] = N2_min
        

    """
    stop_criteria: dict
        Specifies the stopping criteria to be used.
        Can specify: 'max_elapsed_time', 'max_num_iterations',
        default: 'max_elapsed_time': 5 minutes
    store_all_solutions: boolean
        Specifies whether all found solutions should be returned or only the best,
        default: False
    """
    def run(self, stop_criteria={'max_elapsed_time': 5*60}, store_all_solutions=False):
        # store important info
        best_sol = None
        num_iter = {'add':0, 'remove':0}
        pareto_frontier = []
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
        np.random.seed(self.random_seed) # set seed for random strategies
        S_values = np.array([self.data_train[0]]) # assume first index contains nominal data
        S_indices = [0] # tracks the indices of the scenarios in Z
        
        start_time = time.time()
        count_iter = 0
        while True:
            # check if stopping criteria has been met
            elapsed_time = time.time() - start_time
            num_iterations = sum(v for k,v in num_iter.items())
            if self._stopping_cond(stop_criteria, elapsed_time=elapsed_time, num_iterations=num_iterations):
                break
            
            count_iter += 1
            
            x_i, obj_scp, duals = self.solve_SCP(S_values, **self.problem_instance)
            obj_i = obj_scp
            S_history.append(S_indices.copy())
            
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
            
            # loop through unc constraints and compute bounds
            if self.eval_unc_constr is not None:
                for unc_constr_i in self.eval_unc_constr:
                    # get feasibility certificate on train data
                    feas_cert_constr_i, evals = self._compute_constr_feas_certificate(x_i, self.data_train, unc_constr_i)
                    feas_certificates_train.append(feas_cert_constr_i)
                    evals_train.append(evals)
                    
                    # get feasibility certificate on test data
                    feas_cert_constr_i, evals = self._compute_constr_feas_certificate(x_i, self.data_test, unc_constr_i)
                    feas_certificates_test.append(feas_cert_constr_i)
            
            if self.verbose:
                print("-----------------")
                print("iter     : " + f'{round(count_iter,0):.0f}')
                print("S_ind    : [", *S_indices, "]")
                print("S_vals   : [", *S_values, "]")
                print("duals    : [", *duals, "]")
                print("size_S   : " + f'{round(len(S_values),0):.0f}')
                print("obj_S    : " + f'{round(obj_scp,3):.3f}')
                if self.eval_unc_obj is not None:
                    print("obj_test : " + f'{round(obj_cert,3):.3f}')
                if len(feas_certificates_test) > 0:
                    for i in range(len(feas_certificates_test)):
                        print("cert_con_"+str(i)+" : " + f'{round(feas_certificates_test[i],3):.3f}')
                        print("train_cert_con_"+str(i)+" : " + f'{round(feas_certificates_train[i],3):.3f}')
            
            if store_all_solutions:
                all_solutions.append({'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test})
            
            # check whether best solution can be replaced
            best_sol, update_best_yn = self._update_best_sol(best_sol, x_i, obj_i, feas_certificates_test, desired_constr_cert_rhs)
            
            if self.verbose and update_best_yn:
                print("New best solution found!")
            
            # update pareto frontier          
            if self.eval_unc_constr is not None:
                pareto_frontier = self._update_pareto_frontier(pareto_frontier, x_i, obj_i, feas_certificates_test)
            
            # given the training feasibility certificates, we now determine action
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
                    S_values, S_indices = self._add_scenarios(S_values, S_indices, possible_add_ind) 
                num_iter['add'] += 1
            elif add_or_remove == False:                
                if len(evals_train) > 1:
                    #TODO: add code to handle multiple uncertain functions
                    ...
                else:
                    S_values, S_indices = self._remove_scenarios(S_values, S_indices, possible_rem_ind, duals)
                num_iter['remove'] += 1
        
            if self.verbose:
                if add_or_remove:
                    print("Decided to add a scenario to S")
                else:
                    print("Decided to remove a scenario from S")
            
        runtime = time.time() - start_time
        
        if store_all_solutions:
            return best_sol, runtime, num_iter, pareto_frontier, S_history, all_solutions
        
        return best_sol, runtime, num_iter, pareto_frontier, S_history
        
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
        return util.compute_mod_chi2_lowerbound(1-p_vio, N, self.conf_param_alpha)
    
    def _update_best_sol(self, best_sol, x_i, obj_i, feas_certificates_test, desired_constr_cert_rhs):
        update_best_yn = False
        if best_sol is None:
            best_sol = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
            update_best_yn = True
        else:
            # first check if feasible, if both, than compare obj
            # if both infeasible, then take sol with smallest largest gap
            best_is_feas = all(best_sol['feas'][i] >= val for i,val in enumerate(desired_constr_cert_rhs))
            x_i_is_feas = all(feas_certificates_test[i] >= val for i,val in enumerate(desired_constr_cert_rhs))
            
            if (best_is_feas==False and x_i_is_feas):
                best_sol = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                update_best_yn = True
            elif (best_is_feas and x_i_is_feas):
                if (obj_i < best_sol['obj']):
                    best_sol = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                    update_best_yn = True
            elif (best_is_feas==False and x_i_is_feas==False):  
                max_gap_x_i = np.max(np.subtract(desired_constr_cert_rhs, feas_certificates_test))
                max_gap_best = np.max(np.subtract(desired_constr_cert_rhs, best_sol['feas']))
                if (max_gap_x_i < max_gap_best):
                    best_sol = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates_test}
                    update_best_yn = True
    
        return best_sol, update_best_yn
    
    def _update_pareto_frontier(self, pareto_frontier, x_i, obj_i, feas_certificates_test):
        x_i_pareto_eff_yn = True
        indices_to_be_removed = []
        for j,(obj, li_feas_cert) in enumerate(pareto_frontier):
            if (obj <= obj_i and 
                all(li_feas_cert[i] >= feas_certificates_test[i] for i in range(len(li_feas_cert)))):
                x_i_pareto_eff_yn = False
            elif ((obj >= obj_i and all(li_feas_cert[i] <= feas_certificates_test[i] for i in range(len(li_feas_cert))))
                    and (obj > obj_i or any(li_feas_cert[i] < feas_certificates_test[i] for i in range(len(li_feas_cert))))):
                indices_to_be_removed.append(j)
                  
        for index in sorted(indices_to_be_removed, reverse=True):
            del pareto_frontier[index]
        if x_i_pareto_eff_yn:
            pareto_frontier.append((obj_i, feas_certificates_test))
    
        return pareto_frontier
    
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

    def _get_possible_additions_old(self, evals_train, tabu_add, numeric_precision):
        evals_train_add = []
        max_num_possible_additions = 0
        for i in range(len(evals_train)):
            evals_train_add_i = evals_train[i].copy()
            if len(tabu_add) > 0:
                evals_train_add_i = np.delete(evals_train_add_i, list(tabu_add))
            
            evals_train_add.append(evals_train_add_i)
            
            num_vio = sum(evals_train_add_i>(0+numeric_precision))
            if num_vio > max_num_possible_additions:
                max_num_possible_additions = num_vio
            
        return evals_train_add, max_num_possible_additions

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
        possible_rem_ind = set(S_indices) - tabu_remove
        # binding_scenarios = set()
        # for i in range(len(evals_train)):
        #     binding_i = np.argwhere((evals_train[i]>(0-self.numeric_precision)) & (evals_train[i]<(0+self.numeric_precision)))
        #     if len(binding_i) > 1:
        #         binding_scenarios.update(binding_i.squeeze())
        # scen_is_binding_yn = {i: (True if i in binding_scenarios else False) for i in possible_rem_ind}
        # return possible_rem_ind, scen_is_binding_yn
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
        elif method == 'sigmoid':
            max_prob_add = 0
            for i,cert in enumerate(feas_certificates_train):
                prob_add = util.compute_prob_add_sigmoid(desired_cert_rhs[i] - cert)
                if prob_add > max_prob_add:
                    max_prob_add = prob_add
            return max_prob_add
        else:
            print('Error: do not recognize method in "compute_prob_add" function')
            return 1

    def _add_scenarios(self, S_values, S_indices, possible_add_ind):
        if self.add_strategy == 'random_vio':
            ind = np.random.choice(list(possible_add_ind))
            possible_add_ind.remove(ind)
        else:
            print("Error: do not recognize addition strategy")
            return None
        S_indices.append(ind)
        scen_to_add = [self.data_train[ind]]
        if len(S_values) > 0:
            S_values = np.append(S_values, scen_to_add, axis = 0)
        else:
            S_values = np.array(scen_to_add, dtype=object)
        return S_values, S_indices

    def _remove_scenarios(self, S_values, S_indices, possible_rem_ind, duals):
        if self.rem_strategy == 'random_any':
            li_scen_to_remove = []
            while True:
                ind = np.random.choice(list(possible_rem_ind))
                i_ind = S_indices.index(ind)
                possible_rem_ind.remove(ind)
                li_scen_to_remove.append(i_ind)
                if duals[i_ind] > (0+self.numeric_precision):
                    break
            
            S_values = np.array([e for i,e in enumerate(S_values) if i not in li_scen_to_remove])
            for i_ind in sorted(li_scen_to_remove, reverse=True):
                del S_indices[i_ind]
        else:
            print("Error: do not recognize removal strategy")
            return None
        
        return S_values, S_indices
    
    def _stopping_cond(self, stop_criteria, **kwargs):
        """
        Returns true if a stopping condition is met, else False.
        """
        if (kwargs.get('elapsed_time',0) >= stop_criteria.get('max_elapsed_time', 10e12) 
            or kwargs.get('num_iterations',0) >= stop_criteria.get('max_num_iterations', 10e12)):
            return True
        else:
            return False   

    def _determine_N_min(self, N, desired_prob_rhs):
        p_feas_min = self._determine_p_feas_min(N, desired_prob_rhs)
        N_min = math.ceil(p_feas_min * N)
        return N_min

    def _determine_p_feas_min(self, N, desired_prob_rhs):
        # golden section search in interval (desired_prob_rhs, 1)
        gr = (math.sqrt(5) + 1) / 2
        tol = 1e-5
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

    def _determine_p_feas_min_old(self, N, desired_prob_rhs):
        # "fixed" settings for this procedure
        delta = 0.1
        stopping_criteria_epsilon = 0.0001
        
        p_vio = 1-desired_prob_rhs        
        lb = self._compute_phi_div_bound(p_vio, N)
        p_prev = p_vio
        while True:
            if p_vio - delta < stopping_criteria_epsilon:
                delta = delta/10
            p_vio = p_vio - delta
            lb = self._compute_phi_div_bound(p_vio, N)
            if lb < desired_prob_rhs:
                continue
            else:
                delta = delta / 10
                if delta < stopping_criteria_epsilon:
                    break
                else:
                    p_vio = p_prev 
        return 1-p_vio, lb
