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
        Returns the optimal primal (and dual if use_dual_sol==true) solution, 
        along with the objective of the solved problem.
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
    data_train: list 
        Provides the training data set.
    data_valid: list 
        Provides the validation data set.
        If set to None, this signals that ROBIST should use the training
        data to select the most promising solution.
    data_test: list 
        Provides the testing data set.
        If set to None, this signals that ROBIST should use the validation
        data to derive collectively valid probabilistic guarantees.
    conf_param_alpha: float
        The statistical confidence level for the probability guarantee
        used in the evaluation of solutions.
        If set to None, this signals that the user does not care about the 
        probabilistic guarantees, instead the empirical values are used.
        default: None
    add_strategy : string
        Specifies the scenario addition strategy to be used, default: 'random_vio'.
    remove_strategy : string
        Specifies the scenario removal strategy to be used, default: 'random_any'.
    use_dual_sol: boolean
        True if one wishes to use dual solution information to potentially speed up the algorithm
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
    best_solution: dictionary
        All information regarding best found solution during the search
    runtime: float
        Total runtime of algorithm (in seconds)
    num_iter: dictionary
        Number of add & remove iterations performed
    non_dominated_solutions: list of tuples containing (obj, [feas_cert_constr_1, feas_cert_constr_2, ...])
        If the problem has at least 1 uncertain constraint, list of tuples containing non-dominated solutions 
        found during search. Else, None
    S_history: list
        List that tracks the scenario sets used at each iteration
    """  
    
    def __init__(self, solve_SCP, problem_instance, eval_unc_obj, eval_unc_constr, 
                 data_train, data_valid, data_test=None, conf_param_alpha=None,
                 add_strategy='random_vio', rem_strategy='random_any', use_dual_sol=False,
                 use_tabu=False, numeric_precision=1e-6, verbose=False):
        
        self.solve_SCP = solve_SCP
        self.problem_instance = problem_instance
        self.eval_unc_obj = eval_unc_obj
        self.eval_unc_constr = eval_unc_constr
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.conf_param_alpha = conf_param_alpha
        self.add_strategy = add_strategy
        self.rem_strategy = rem_strategy
        self.use_dual_sol = use_dual_sol
        problem_instance['get_dual_sol'] = use_dual_sol
        self.use_tabu = use_tabu
        self.numeric_precision = numeric_precision
        self.verbose = verbose
        
        # In case obj is uncertain, determine N2_min (if not already provided)
        if eval_unc_obj is not None:
            eval_unc_obj_info = eval_unc_obj['info']
            if eval_unc_obj_info['risk_measure'] == 'probability':
                if data_valid is None and 'N1_min' not in eval_unc_obj_info.keys():
                    N1 = len(data_train)
                    desired_rhs = eval_unc_obj_info.get('desired_rhs')
                    if self.conf_param_alpha is None:
                        N1_min = math.ceil(N1 * (1-desired_rhs))
                    else:
                        N1_min = self._determine_N_min(N1, 1-desired_rhs)
                    self.eval_unc_obj['info']['N1_min'] = N1_min

                if data_valid is not None and 'N2_min' not in eval_unc_obj_info.keys():
                    N2 = len(data_valid)
                    desired_rhs = eval_unc_obj_info.get('desired_rhs')
                    if self.conf_param_alpha is None:
                        N2_min = math.ceil(N2 * (1-desired_rhs))
                    else:
                        N2_min = self._determine_N_min(N2, 1-desired_rhs)
                    self.eval_unc_obj['info']['N2_min'] = N2_min
                
                
                if data_test is not None and 'N3_min' not in eval_unc_obj_info.keys():
                    N3 = len(data_test)
                    desired_rhs = eval_unc_obj_info.get('desired_rhs')
                    if self.conf_param_alpha is None:
                        N3_min = math.ceil(N3 * (1-desired_rhs))
                    else:
                        N3_min = self._determine_N_min(N3, 1-desired_rhs)
                    self.eval_unc_obj['info']['N3_min'] = N3_min


    """
    init_S: list
        Specifies the set of scenarios with which to initialize the sampled set of scenarios (S)
        default: None (in which case we randomly pick a scenario from training data)
    stop_criteria: dict
        Specifies the stopping criteria to be used
        Can specify: 'max_elapsed_time', 'max_num_iterations'
        default: 'max_elapsed_time': 300 seconds
    return_all_solutions: boolean
        Specifies whether all found solutions should be returned 
        default: False
    random_seed: int
        Specifies random seed to be set at start of algorithm run
        default: 0
    prob_opposite_action: float
        Specifies the probability with which the add/remove decision is flipped 
        default: 0.01
    """
    def run(self, init_S=None, stop_criteria={'max_elapsed_time': 5*60}, 
            return_all_solutions=False, random_seed=0, prob_opposite_action=0.01):
        # create variables to store info
        start_time = time.time()
        best_solution = None
        num_iter = {'add':0, 'remove':0}
        non_dominated_solutions = []
        S_history = []
        all_solutions = []
        desired_cert_rhs = []
        if self.eval_unc_obj is not None:
            desired_cert_rhs.append(self.eval_unc_obj['info']['desired_rhs'])
        if self.eval_unc_constr is not None:
            for unc_constr_i in self.eval_unc_constr:
                desired_cert_rhs.append(unc_constr_i['info']['desired_rhs'])
        
        # initialize algorithm
        if init_S is not None:
            # if we are given an initial set of sampled scenarios, we assign dummy indices 
            S = init_S
            S_indices = [len(self.data_train)+i for i in range(len(init_S))]
            for s in init_S:
                np.append(self.data_train, s) 
        else:  
            # randomly pick a scenario from training data set
            init_scen = np.random.randint(0, len(self.data_train))
            S = [self.data_train[init_scen]]
            S_indices = [init_scen] # tracks the indices of the scenarios in Z
        np.random.seed(random_seed) # use numpy.random as PRNG for randomization
        skip_iter = False # if use_dual_sol==True, may be able to skip an iteration in certain cases 
        
        if self.verbose: 
                print("--------------------------------------------------------------------")
                print(f"Initialization complete, will run with T_max={stop_criteria.get('max_elapsed_time', 'default')} and i_max={stop_criteria.get('max_num_iterations', 'None')}")
                print("--------------------------------------------------------------------")
                print()
                print("--------------------------------------------------------------------")
                print("Starting generation phase")
                print("--------------------------------------------------------------------")

        while True:
            # check if stopping criteria has been met
            elapsed_time = time.time() - start_time
            num_iterations = sum(num_iter.values())
            if self._stopping_cond(stop_criteria, 
                                   elapsed_time=elapsed_time, 
                                   num_iterations=num_iterations):
                break
               
            if skip_iter == False:
                # generate solution by solving sampled convex program for some set of scenarios S
                S_history.append(S_indices.copy())
                if self.use_dual_sol:
                    x_i, obj_i, duals_i = self.solve_SCP(S, **self.problem_instance)
                else:
                    x_i, obj_i = self.solve_SCP(S, **self.problem_instance)
                
                # evaluate generated solution on training data
                certificates_train = []
                evals_train = []
                if self.eval_unc_obj is not None: 
                    # get feasibility certificate on train data
                    feas_cert_obj, evals = self._compute_obj_feas_certificate(x_i, obj_i, self.data_train)
                    certificates_train.append(feas_cert_obj)
                    evals_train.append(evals)
                
                # loop through uncertain constraints and compute feasibility certificates
                if self.eval_unc_constr is not None:
                    for unc_constr_i in self.eval_unc_constr:
                        # get feasibility certificate on train data
                        feas_cert_constr_i, evals = self._compute_constr_feas_certificate(x_i, self.data_train, unc_constr_i)
                        certificates_train.append(feas_cert_constr_i)
                        evals_train.append(evals)
            
                # store information
                all_solutions.append({'sol': x_i, 'obj': obj_i, 'cert_train': certificates_train})

                if self.verbose: 
                    print("-----------------")
                    print("iter     : " + f'{round(num_iterations,0):.0f}')
                    print("S_ind    : [", *S_indices, "]")
                    print("obj_S    : " + f'{round(obj_i,3):.3f}')
                    for i in range(len(certificates_train)):
                        print("train_cert_"+str(i)+" : " + f'{round(certificates_train[i],3):.3f}')
            else:
                skip_iter = False
            
            # given the training certificates, we now determine the next action
            if self.use_tabu == True: 
                tabu_add = self._get_tabu_add(S_indices, S_history)
                tabu_remove = self._get_tabu_remove(S_indices, S_history)
            else:
                tabu_add = set()
                tabu_remove = set()
            
            # determine whether to add or remove scenarios using the training data evaluations        
            possible_add_ind = self._get_possible_additions(evals_train, tabu_add, S)
            possible_rem_ind = self._get_possible_removals(S_indices, tabu_remove)
            add_or_remove = self._determine_action(certificates_train, desired_cert_rhs, 
                                                   len(possible_add_ind), len(possible_rem_ind),
                                                   prob_opposite_action)
            
            if add_or_remove is None: # signifies that no more actions are possible
                if self.verbose:
                    print("No additions or removals possible")
                break

            if add_or_remove == True:
                if len(evals_train) > 1:
                    #TODO: add code to handle multiple uncertain functions
                    ...
                else:
                    S, S_indices = self._add_scenario(S, S_indices, possible_add_ind) 
                num_iter['add'] += 1
            else:                
                if len(evals_train) > 1:
                    #TODO: add code to handle multiple uncertain functions
                    ...
                else:
                    S, S_indices, i_removal = self._remove_scenario(S, S_indices, possible_rem_ind)
                    if self.use_dual_sol:
                        # check if removed scenario has dual==0
                        if duals_i[i_removal] > 0-self.numeric_precision and duals_i[i_removal] < 0+self.numeric_precision:
                            skip_iter = True
                            del duals_i[i_removal]
            
                num_iter['remove'] += 1

            if self.verbose:
                if add_or_remove:
                    print("Decided to add a scenario to S")
                else:
                    print("Decided to remove a scenario from S")

        if self.verbose: 
            print()
            print("--------------------------------------------------------------------")
            print("Starting evaluation phase")
            print("--------------------------------------------------------------------")

        # use evaluations on validation (or training) data to select final solution
        if self.data_test is not None:
            for info_i in all_solutions:
                x_i = info_i['sol']
                obj_i = info_i['obj']
                
                # evaluate generated solutions on validation (or training) data
                if self.eval_unc_obj is not None: 
                    if self.data_valid is not None:
                        obj_cert = self._compute_obj_certificate(x_i, self.data_valid, N_min=self.eval_unc_obj['info']['N2_min'])
                    else:
                        obj_cert = self._compute_obj_certificate(x_i, self.data_train, N_min=self.eval_unc_obj['info']['N1_min'])
                    obj_i = obj_cert # replace obj with obj certificate
                
                feas_certificates_valid = []
                # compute feasibility certificates
                if self.eval_unc_constr is not None:
                    if self.data_valid is not None:
                        for unc_constr_i in self.eval_unc_constr:
                                feas_cert_constr_i, _ = self._compute_constr_feas_certificate(x_i, self.data_valid, unc_constr_i)
                                feas_certificates_valid.append(feas_cert_constr_i)
                    else:
                        if self.eval_unc_obj is not None:
                            feas_certificates_valid = info_i['cert_train'][1:]
                        else:
                            feas_certificates_valid = info_i['cert_train']

                info_i['cert_valid'] = feas_certificates_valid

                # check whether best solution can be updated
                best_solution, update_best_yn = self._update_best_solution(best_solution, x_i, obj_i, feas_certificates_valid, desired_cert_rhs)
                
                # check whether non-dominated solutions can be updated          
                if self.eval_unc_constr is not None:
                    non_dominated_solutions = self._update_non_dominated_solutions(non_dominated_solutions, obj_i, feas_certificates_valid)

                if self.verbose: 
                    if self.eval_unc_obj is not None:
                        print("obj_valid : " + f'{round(obj_cert,3):.3f}')
                    if len(feas_certificates_valid) > 0:
                        for i in range(len(feas_certificates_valid)):
                            print("valid_cert_con_"+str(i)+" : " + f'{round(feas_certificates_valid[i],3):.3f}')
                    # if update_best_yn:
                    #     print("New best solution found!")
        else:
            # if using collective prob guarantee, may want to only evaluate solutions for which train certificate >= 1-epsilon
            # solutions_to_eval = []
            # for info_i in all_solutions:
            #     if all(info_i['cert_train'][i] >= rhs for i,rhs in enumerate(desired_cert_rhs)):
            #         solutions_to_eval.append(info_i)
            solutions_to_eval = all_solutions
            if self.eval_unc_obj is not None: 
                obj_cert = self._compute_obj_certificate_collective(solutions_to_eval, self.data_valid)

                for j, sol_info_j in enumerate(solutions_to_eval):
                    sol_info_j['obj'] = obj_cert[j]  
             
            # loop through uncertain constraints and compute feasibility certificates
            if self.eval_unc_constr is not None:
                for sol_info_j in solutions_to_eval:
                    sol_info_j['feas_certificates_valid'] = []

                for i, unc_constr_i in enumerate(self.eval_unc_constr):
                    # get feasibility certificate on validation data
                    feas_cert_constr_i = self._compute_constr_feas_certificate_collective(solutions_to_eval, self.data_valid, unc_constr_i)
                    for j, sol_info_j in enumerate(solutions_to_eval):
                        sol_info_j['feas_certificates_valid'].append(feas_cert_constr_i[j])

            # find best solution
            best_solution = self._get_best_solution(solutions_to_eval, desired_cert_rhs)
            
            # find non-dominated solutions          
            if self.eval_unc_constr is not None:
                non_dominated_solutions = self._get_non_dominated_solutions(solutions_to_eval)
        
        if self.data_test is not None:
            # now that we have determined the "best" solution, we evalute it OoS with the test data
            obj_best = best_solution['obj']
            if self.eval_unc_obj is not None: 
                # replace obj with obj certificate from test data
                obj_cert_best = self._compute_obj_certificate(best_solution['sol'], self.data_test, N_min=self.eval_unc_obj['info']['N3_min'])
                obj_best = obj_cert_best
            
            feas_certificates_test = []
            # loop through uncertain constraints and compute feasibility certificates
            if self.eval_unc_constr is not None:
                for unc_constr_i in self.eval_unc_constr:
                    # get feasibility certificate on test data
                    feas_cert_constr_best, _ = self._compute_constr_feas_certificate(best_solution['sol'], self.data_test, unc_constr_i)
                    feas_certificates_test.append(feas_cert_constr_best)

            best_solution['obj'] = obj_best
            best_solution['feas'] = feas_certificates_test
        else:
            obj_cert_best = best_solution['obj']
            feas_certificates_test = best_solution['feas']


        if self.verbose:   
            print()
            print("--------------------------------------------------------------------")
            print("Done. Best solution info:")
            print("--------------------------------------------------------------------")
            if self.eval_unc_obj is not None:
                print("obj_test : " + f'{round(obj_cert_best,3):.3f}')
            if len(feas_certificates_test) > 0:
                for i in range(len(feas_certificates_test)):
                    print("test_cert_con_"+str(i)+" : " + f'{round(feas_certificates_test[i],3):.3f}')

        runtime = time.time() - start_time
        if return_all_solutions:
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

            if self.conf_param_alpha is None:
                return p_vio, constr_evals
            else:
                return self._compute_phi_div_bound(p_vio, N), constr_evals
        elif eval_constr_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None, None
        else:
            print("ERROR: do not recognize risk measure")
            return None, None   
       
    def _compute_constr_feas_certificate_collective(self, solutions_to_eval, data, unc_constr_i):
        eval_constr_func = unc_constr_i['function']
        eval_constr_info = unc_constr_i['info']
        
        if eval_constr_info.get('risk_measure') == 'probability':
            N = len(data)
            data_points = []
            for sol_info_i in solutions_to_eval:
                x = sol_info_i['sol']
                constr_evals = eval_constr_func(x, data, **self.problem_instance)
                data_points_i = np.array([1 if val <= (0+self.numeric_precision) else 0 for val in constr_evals]) # 1 if feas, else 0
                data_points.append(data_points_i)
            
            if self.conf_param_alpha is None:
                p_vio = [1 - np.mean(i) for i in data_points]
                return p_vio
            else:
                return self._compute_phi_div_bound_radiusAdjusted(data_points, N, len(solutions_to_eval))
        elif eval_constr_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None
        else:
            print("ERROR: do not recognize risk measure")
            return None   

    def _compute_obj_feas_certificate(self, x, obj, data):
        eval_obj_func = self.eval_unc_obj['function']
        eval_obj_info = self.eval_unc_obj['info']
        
        if eval_obj_info.get('risk_measure') == 'probability':
            N = len(data)
            obj_evals = eval_obj_func(x, data, **self.problem_instance)
            constr = obj_evals - obj
            N_vio = sum(constr>(0+self.numeric_precision))
            p_vio = N_vio/N
            if self.conf_param_alpha is None:
                return p_vio, constr
            else:
                return self._compute_phi_div_bound(p_vio, N), constr
        elif eval_obj_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None, None
        else:
            print("ERROR: do not recognize risk measure")
            return None, None
        
    def _compute_obj_certificate(self, x, data, N_min=None):
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
            
            # grab N_min-th largest eval
            theta_min = obj_evals_sorted[N_min - 1]
            return theta_min
        elif eval_obj_info.get('risk_measure') == 'expectation':
            #TODO: write code for expectation case
            return None
        else:
            print("ERROR: do not recognize risk measure")
            return None
        
    def _compute_obj_certificate_collective(self, solutions_to_eval, data):
        """
        Returns the "objective certificate" for a collection of solutions. This is the
        best obj claim that can be made given the data and risk tolerance.
        """
        eval_obj_func = self.eval_unc_obj['function']
        eval_obj_info = self.eval_unc_obj['info']
        if eval_obj_info.get('risk_measure') == 'probability':
            #TODO: finish function
            return ...
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
    
    def _compute_phi_div_bound_radiusAdjusted(self, data_points, N, m, K=10000):
        p_hat = [np.mean(data_points[i]) for i in range(m)]
        sigma_hat = [np.std(data_points[i]) for i in range(m)]

        # compute estimated cov matrix
        Sigma_hat = np.corrcoef(np.stack(data_points, axis=0))

        # draw K samples from estimated multivariate normal distribution
        if m > 1:
            mu = [0 for i in range(m)]
            sample_Y = np.random.multivariate_normal(mu, Sigma_hat, K, check_valid='ignore') # to avoid error when the estimated covariance is not positive-semidefinite
        else:
            sample_Y = np.random.normal(0, Sigma_hat, K)

        # evaluate (1-\alpha) quantile for max within each sample
        sample_Y_max = [np.max(sample_Y[i]) for i in range(K)]
        q_alpha_hat = np.quantile(sample_Y_max, 1-self.conf_param_alpha) 

        # set radius equal to this quantile, and compute feasibility certificates (probability guarantees)
        sqrt_radius = q_alpha_hat
        sqrt_N = math.sqrt(N)
        constant_term = sqrt_radius / sqrt_N
        gamma = [p_hat[i] - constant_term * math.sqrt(sigma_hat[i]) for i in range(m)]

        return gamma
    
    def _update_best_solution(self, best_solution, x_i, obj_i, feas_certificates, desired_cert_rhs):
        update_best_yn = False
        if best_solution is None:
            best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates}
            update_best_yn = True
        else:
            # first check if feasible, if both, then compare obj
            # if both infeasible, then take sol with smallest largest gap
            if len(feas_certificates) > 0:
                if len(desired_cert_rhs) == len(feas_certificates) + 1:
                    desired_constr_cert_rhs = desired_cert_rhs[1:] # remove obj cert rhs
                else:
                    desired_constr_cert_rhs = desired_cert_rhs
                best_is_feas = all(best_solution['feas'][i] >= val for i,val in enumerate(desired_constr_cert_rhs))
                x_i_is_feas = all(feas_certificates[i] >= val for i,val in enumerate(desired_constr_cert_rhs))
            else:
                best_is_feas = True
                x_i_is_feas = True
            
            if (best_is_feas==False and x_i_is_feas):
                best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates}
                update_best_yn = True
            elif (best_is_feas and x_i_is_feas):
                if (obj_i < best_solution['obj']):
                    best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates}
                    update_best_yn = True
            elif (best_is_feas==False and x_i_is_feas==False):  
                max_gap_x_i = np.max(np.subtract(desired_constr_cert_rhs, feas_certificates))
                max_gap_best = np.max(np.subtract(desired_constr_cert_rhs, best_solution['feas']))
                if (max_gap_x_i < max_gap_best):
                    best_solution = {'sol': x_i, 'obj': obj_i, 'feas': feas_certificates}
                    update_best_yn = True
    
        return best_solution, update_best_yn
    
    def _get_best_solution(self, solutions_to_eval, desired_cert_rhs):
        best_solution = None
        for sol_info in solutions_to_eval:
            best_solution, _ = self._update_best_solution(best_solution, sol_info['sol'], sol_info['obj'], sol_info['feas_certificates_valid'], desired_cert_rhs)
        return best_solution

    def _update_non_dominated_solutions(self, non_dominated_solutions, obj_i, feas_certificates):
        x_i_pareto_eff_yn = True
        indices_to_be_removed = []
        for j,(obj, li_feas_cert) in enumerate(non_dominated_solutions):
            if (obj <= obj_i and 
                all(li_feas_cert[i] >= feas_certificates[i] for i in range(len(li_feas_cert)))):
                x_i_pareto_eff_yn = False
            elif ((obj >= obj_i and all(li_feas_cert[i] <= feas_certificates[i] for i in range(len(li_feas_cert))))
                    and (obj > obj_i or any(li_feas_cert[i] < feas_certificates[i] for i in range(len(li_feas_cert))))):
                indices_to_be_removed.append(j)
                  
        for index in sorted(indices_to_be_removed, reverse=True):
            del non_dominated_solutions[index]
        if x_i_pareto_eff_yn:
            non_dominated_solutions.append((obj_i, feas_certificates))
    
        return non_dominated_solutions
    
    def _get_non_dominated_solutions(self, solutions_to_eval):
        non_dominated_solutions = []
        for sol_info in solutions_to_eval:
            non_dominated_solutions = self._update_non_dominated_solutions(non_dominated_solutions, sol_info['obj'], sol_info['feas_certificates_valid'])
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

    def _get_possible_additions(self, evals_train, tabu_add, S_current):
        possible_add_ind = set()
        if self.add_strategy == 'random_vio':
            for i in range(len(evals_train)):
                vio_i = np.argwhere(evals_train[i]>(0+self.numeric_precision))
                if len(vio_i) > 1:
                    possible_add_ind.update(vio_i.squeeze())
        elif self.add_strategy == 'random_any':
            scenarios_not_in_S = [i for i in range(len(self.data_train)) if i not in S_current]
            possible_add_ind.update(scenarios_not_in_S)
        elif self.add_strategy == 'largest_vio':
            for i in range(len(evals_train)):
                if sum(evals_train[i]>(0+self.numeric_precision)) > 0:
                    largest_vio_i = np.argmax(evals_train[i])
                    if np.issubdtype(largest_vio_i, np.integer):
                        possible_add_ind.add(largest_vio_i)
                    else:
                        possible_add_ind.update(largest_vio_i.squeeze())
        else:
            print("Error: do not recognize addition strategy")
            return None
            
        # remove tabu indices
        possible_add_ind = possible_add_ind - tabu_add
        return possible_add_ind

    def _get_possible_removals(self, S_indices, tabu_remove):
        if len(S_indices) == 1:
            return set()
        possible_rem_ind = set(S_indices) - tabu_remove
        return possible_rem_ind

    def _determine_action(self, feas_certificates_train, desired_cert_rhs, num_possible_additions, num_possible_removals, prob_opposite_action):    
        # Determines whether it will be an add (True) or remove (False) or break (None) 
        if num_possible_additions == 0 and num_possible_removals == 0:
            return None
        elif num_possible_additions == 0:
            return False
        elif num_possible_removals == 0:
            return True
        random_draw = np.random.uniform()
        satisfies_rhs = all(feas_certificates_train[i] >= rhs for i,rhs in enumerate(desired_cert_rhs))
        if (satisfies_rhs and random_draw > prob_opposite_action): 
            return False
        elif (not satisfies_rhs and random_draw < prob_opposite_action): 
            return False
        return True

    def _add_scenario(self, S, S_indices, possible_add_ind):
        ind = np.random.choice(list(possible_add_ind))
        possible_add_ind.remove(ind)
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
        if (kwargs.get('elapsed_time',0) >= stop_criteria.get('max_elapsed_time', 10e12) 
            or kwargs.get('num_iterations',0) >= stop_criteria.get('max_num_iterations', 10e12)):
            return True
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
