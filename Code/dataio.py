# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib settings:
plt.rcParams['figure.figsize'] = [9, 7]
plt.rcParams['figure.dpi'] = 100 # can be increased for better quality

def print_solution_info(sol_info):
    print('obj: ' + str(sol_info['obj']))
    print('lb_train: ' + str(sol_info['lb_train']))
    print('lb_test: ' + str(sol_info['lb_test']))
    print('time_found: ' + str(sol_info['time']))
    print('scenario_set: ' + str(sol_info['scenario_set']))

def plot_iter(num_iter, data, Z_arr, x, obj, p, lb, prob_true, save_plot, plot_type, show_legend,
              N, alpha, beta):
    plt.plot(data[:,0],data[:,1],'ok',markersize=1, label = 'All scenarios')
    
    if Z_arr is not None:
        plt.plot(Z_arr[:,0],Z_arr[:,1], color='blue', marker='+', linestyle='',
                 markersize=10, label = 'Chosen scenarios')

    # Add constraint to plot, given solution x
    constraint_x = np.linspace(-1, 1, 1000)
    constraint_y = (1 - x[0]*constraint_x) / x[1]
    plt.plot(constraint_x, constraint_y, '--r', label = r'$\xi_{1}x_{1}^{*}+\xi_{2}x_{2}^{*}\leq 1$' ,alpha=1)

    plt.title(r'Iter '+str(num_iter)+': $\mathbf{x}$ = (' + str(round(x[0],2)) + ', ' 
              + str(round(x[1],2)) + '), Obj = ' + str(round(obj,2)) 
              + ', $p$ = '+ str(round(p,2))
             + ', $LB$ = '+ str(round(lb,2))
             + ', True Prob. = '+ str(round(prob_true,2)))
    
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'output/ToyProblem/Scenarios_wConstraint_iter='+str(num_iter)+'_N=' + str(N) + '_alpha=' + str(alpha) + "_beta="+ str(beta)
        plt.savefig(plot_name + '.' + plot_type)
    
    plt.show()
    
def plot_solution(name, data, Z_arr, x, obj, lb, save_plot, plot_type, show_legend,
                  N, alpha, beta):
    if data.shape[1] > 2:
        print("ERROR: Cannot print larger than 2 dim")
        return
    
    plt.plot(data[:,0],data[:,1],'ok',markersize=1, label = 'All scenarios')
    
    if Z_arr is not None:
        plt.plot(Z_arr[:,0],Z_arr[:,1], color='blue', marker='+', linestyle='',
                 markersize=10, label = 'Chosen scenarios')

    # Add constraint to plot, given solution x
    constraint_x = np.linspace(-1, 1, 1000)
    constraint_y = (1 - x[0]*constraint_x) / x[1]
    plt.plot(constraint_x, constraint_y, '--r', label = r'$\xi_{1}x_{1}^{*}+\xi_{2}x_{2}^{*}\leq 1$' ,alpha=1)

    plt.title(name +': $\mathbf{x}^{*}$ = (' + f'{round(x[0],3):.3f}'
              + ', ' + f'{round(x[1],3):.3f}'
              + '), Obj = ' + f'{round(obj,3):.3f}'
              + ', LB = '+ f'{round(lb,3):.3f}')
    
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'Figures/ToyModel/Scenarios_wConstraint_'+name+'_N=' + str(N) + '_alpha=' + str(alpha) + "_beta="+ str(beta)
        plt.savefig(plot_name + '.' + plot_type)
    
    plt.show()
    
def plot_pareto_curve(pareto_solutions, beta, save_plot, name, plot_type, show_legend):
    # first we convert the list of tuples to a numpy array to get data in proper format
    array = np.array([*pareto_solutions])
    sorted_array = array[np.argsort(array[:, 0])]
    x = sorted_array[:,0] # contains lb
    y = sorted_array[:,1] # contains obj
        
    plt.plot(x, y, "-o")
    plt.axvline(beta, ls = '--')
    
    plt.xlabel("$\phi$-divergence LB")
    plt.ylabel("objective value");
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'Figures/ToyModel/ParetoCurve_'+name
        plt.savefig(plot_name + '.' + plot_type)
    
    plt.show()
    
def plot_obj_over_time(solutions, best_sol, save_plot, name, plot_type, show_legend):
    # first we convert the list of tuples to a numpy array to get data in proper format
    df = pd.DataFrame(solutions)
    x = df.loc[:,'time']
    y = df.loc[:,'obj']
        
    plt.plot(x, y, "-o")
    plt.axvline(best_sol['time'], ls = '--')
    
    plt.xlabel("time (s)")
    plt.ylabel("obj");
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'Figures/ToyModel/ObjOverTime_'+name
        plt.savefig(plot_name + '.' + plot_type)
    
    plt.show()

def plot_size_set_over_time(solutions, best_sol, save_plot, name, plot_type, show_legend):
    # first we convert the list of tuples to a numpy array to get data in proper format
    df = pd.DataFrame(solutions)
    df['size_S'] = df['scenario_set'].apply(lambda x: len(x))
    x = df.loc[:,'time']   
    y = df.loc[:,'size_S']
        
    plt.plot(x, y, "-o")
    plt.axvline(best_sol['time'], ls = '--')
    
    plt.xlabel("time (s)")
    plt.ylabel(r"|$\mathcal{S}$|")
    #plt.ylabel("size scenario set")
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'Figures/ToyModel/SetSizeOverTime_'+name
        plt.savefig(plot_name + '.' + plot_type)
    
    plt.show()

def plot_hist(values, x_label, y_label, title, num_bins, alpha):
    plt.hist(values, num_bins, density=False, alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    N = len(values)
    mu = values.mean()
    sigma = values.std()
    # Should manually adjust x and y to determine position of text
    plt.text(1.2, 11, r'$N={}, \mu={},\ \sigma={}$'.format(N, round(mu,3), round(sigma,3)))
    
    plt.grid(True)
    plt.show()
    
def plot_portfolio_holdings(solutions):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [5, 3]
    plt.rcParams['figure.dpi'] = 800 # can be increased for better quality
    
    # Follow bertsimas et al. by plotting the 10 & 90% quantiles with average (over 100 runs)
    # get quantiles and mean for each asset seperately
    q_10 = []
    means = []
    q_90 = []
    for i in solutions.columns:
        means.append(np.mean(solutions.loc[:,i]))
        q_10.append(np.percentile(solutions.loc[:,i], 10))
        q_90.append(np.percentile(solutions.loc[:,i], 90))
    
    x = ["x" + str(i) for i in solutions.columns]

    asymmetric_error = [np.array(means)-np.array(q_10), np.array(q_90)-np.array(means)]

    plt.errorbar(x, means, yerr=asymmetric_error,
                marker='o', markersize=2,
                linestyle='dotted')

    plt.xticks(x) # to ensure that all assets are shown on x-axis
    #plt.xlabel("Assets")
    plt.ylabel("Holding (%)")
    plt.tight_layout()
    plt.show()
            
    
def write_output_to_latex(num_settings, headers, data):
    textabular = f"{'l'*num_settings}|{'r'*(len(headers)-num_settings)}"
    texheader = " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    for label in data:
        if num_settings == 1:
            texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"
        elif num_settings == 2:
            texdata += f"{label[0]} & {label[1]} & {' & '.join(map(str,data[label]))} \\\\\n"
        elif num_settings == 3:
            texdata += f"{label[0]} & {label[1]} & {label[2]} & {' & '.join(map(str,data[label]))} \\\\\n"
        elif num_settings == 4:
            texdata += f"{label[0]} & {label[1]} & {label[2]} & {label[3]} & {' & '.join(map(str,data[label]))} \\\\\n"
        elif num_settings == 5:
            texdata += f"{label[0]} & {label[1]} & {label[2]} & {label[3]} & {label[4]} & {' & '.join(map(str,data[label]))} \\\\\n"
        else:
            print("ERROR: provided none OR more than 5 settings")

    print("\\begin{table}[H]")
    print("\\centering")
    print("\\resizebox{\\linewidth}{!}{\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}}")
    print("\\caption{}")
    print("\\label{}")
    print("\\end{table}")
    
def write_output_to_txt_file(output_data):
    with open(r'results_new.txt','w+') as f:
         f.write(str(output_data))

def read_output_from_txt_file(filepath):
    dic = ''
    with open(filepath,'r') as f:
             for i in f.readlines():
                dic=i #string
    output_data = eval(dic)
    return output_data












