# Import packages
import numpy as np
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

def plot_iter(name, num_iter, data, Z_arr, x, obj, lb, save_plot, plot_type, show_legend,
              N, alpha, beta):
    plt.plot(data[:,0],data[:,1],'ok',markersize=1, label = 'All scenarios')
    
    if Z_arr is not None:
        plt.plot(Z_arr[:,0],Z_arr[:,1], color='blue', marker='+', linestyle='',
                 markersize=10, label = 'Chosen scenarios')

    # Add constraint to plot, given solution x
    constraint_x = np.linspace(-1, 1, 1000)
    constraint_y = (1 - x[0]*constraint_x) / x[1]
    plt.plot(constraint_x, constraint_y, '--r', label = r'$\xi_{1}x_{1}^{*}+\xi_{2}x_{2}^{*}\leq 1$' ,alpha=1)

    plt.title('Iteration '+str(num_iter)+': Solution = (' + str(round(x[0],3)) + ', ' 
              + str(round(x[1],3)) + '), Objective value = ' + str(round(obj,3)) 
              + ', Lower bound = '+ str(round(lb,3)))
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.01, 0.6), loc='upper left')
    
    plt.tight_layout()
    
    if save_plot:
        plot_name = 'Figures/ToyModel/Scenarios_wConstraint_iter='+str(num_iter)+'_N=' + str(N) + '_alpha=' + str(alpha) + "_beta="+ str(beta)
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
    
def plot_pareto_curve(pareto_solutions, beta, best_obj, save_plot, plot_type, show_legend):
    # first we convert the list of tuples to a numpy array to get data in proper format
    array = np.array([*pareto_solutions])
    sorted_array = array[np.argsort(array[:, 0])]
    x = sorted_array[:,0] # contains lb
    y = sorted_array[:,1] # contains obj
    x = 1 - x
    
    plt.plot(x, y, "-o")
    plt.vlines(1-beta, 0, np.max(y), linestyles ="dotted")
    
    plt.xlabel("violation probability")
    plt.ylabel("objective value");
    
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
        else:
            print("ERROR: provided none OR more than 3 settings")

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












