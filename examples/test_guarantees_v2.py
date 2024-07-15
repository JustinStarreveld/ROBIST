# In this script we test whether the 1-alpha probability guarantee is empirically accurate
# We also examine to what extent multiple hypothesis testing is an issue

import math
import numpy as np
import scipy
import warnings 
warnings.simplefilter('ignore')
from pyprojroot import here

def make_table(m_values, N_values, alpha, num_reps, K, seed, output_data):
    table = "\\begin{table}[]\n"
    table += "\\centering\n"
    table += "\\resizebox{0.95\\textwidth}{!}{%\n"
    table += "\\begin{tabular}{@{}llrrrrrrrr@{}}\n"
    table += "\\toprule\n"
    table += " &  & \\multicolumn{2}{c}{alpha collective} &  & \\multicolumn{2}{c}{alpha individual} &  & \\multicolumn{2}{c}{MAE /w true prob} \\\\ \\cmidrule(lr){3-4} \\cmidrule(lr){6-7} \\cmidrule(l){9-10} \n"
    table += "m & N & \\multicolumn{1}{r}{old} & \\multicolumn{1}{r}{new} &  & \\multicolumn{1}{r}{old} & \\multicolumn{1}{r}{new} &  & \\multicolumn{1}{r}{old} & \\multicolumn{1}{r}{new} \\\\ \\midrule\n"

    m_prev = 0
    for m in m_values:
        for N in N_values:
            empirical_alpha_collective_old = round(output_data[(m, N)]["empirical_alpha_collective_old"], 3)
            empirical_alpha_collective = round(output_data[(m, N)]["empirical_alpha_collective"], 3)
            empirical_alpha_individual_old = round(output_data[(m, N)]["empirical_alpha_individual_old"], 3)
            empirical_alpha_individual = round(output_data[(m, N)]["empirical_alpha_individual"], 3)
            empirical_MAE_old = round(output_data[(m, N)]["empirical_MAE_old"], 3)
            empirical_MAE = round(output_data[(m, N)]["empirical_MAE"], 3)
            
            if m != m_prev:
                if m != m_values[0]:
                    table +=  "&  &  &  &  &  &  &  &  &  \\\\\n"
                table += f"{m} & {N} & {empirical_alpha_collective_old:.3f} & {empirical_alpha_collective:.3f} &  & {empirical_alpha_individual_old:.3f} & {empirical_alpha_individual:.3f} &  & {empirical_MAE_old:.3f} & {empirical_MAE:.3f} \\\\\n"
            else:
                table += f"    & {N} & {empirical_alpha_collective_old:.3f} & {empirical_alpha_collective:.3f} &  & {empirical_alpha_individual_old:.3f} & {empirical_alpha_individual:.3f} &  & {empirical_MAE_old:.3f} & {empirical_MAE:.3f} \\\\\n"
            m_prev = m

    table += "\\bottomrule\n"
    table += "\\end{tabular}%\n"
    table += "}\n"
    table += f"\\caption{{Comparison between old and new probability guarantee in terms of empirically observed alpha and mean absolute deviation. Settings: alpha={alpha}, numReps={num_reps}, numSamples={K} and seed={seed}.}}\n"
    table += "\\label{tab:compare_guarantee_method}\n"
    table += "\\end{table}\n"

    return table

output_folder = here("examples/output/StatisticalTesting/")
output_folder.mkdir(parents=True, exist_ok=True)
output_data = {}

m_values = [10, 100, 1000]
N_values = [100, 1000]
alpha = 0.10
num_reps = 1000
K = 10000
seed = 1234
rng = np.random.default_rng(seed)

for m in m_values:
    for N in N_values:
        output_file_name = f'test_guarantees_m={m}_N={N}_alpha={alpha}_numReps={num_reps}_numSamples={K}_seed={seed}'

        freq_invalid = []
        freq_invalid_old = []
        mean_abs_error = []
        mean_abs_error_old = []
        for rep in range(num_reps):
            p_true = 0.25 + rng.random(m) / 2 # assume that the true prob(xi feasible) is binomial with random p and independent of other xi

            # generate data and get basic info
            data_points = [rng.binomial(n=1, p=p, size=N) for p in p_true]
            p_hat = [np.mean(data_points[i]) for i in range(m)]
            sigma_hat = [np.std(data_points[i]) for i in range(m)]

            # compute estimated cov matrix
            Sigma_hat = np.corrcoef(np.stack(data_points, axis=0))

            # draw K samples from estimated multivariate normal distribution
            if m > 1:
                mu = [0 for i in range(m)]
                sample_Y = rng.multivariate_normal(mu, Sigma_hat, K, check_valid='ignore') # to avoid error when the estimated covariance is not positive-semidefinitec
            else:
                sample_Y = rng.normal(0, Sigma_hat, K)

            # evaluate (1-\alpha) quantile for max within each sample
            sample_Y_max = [np.max(sample_Y[i]) for i in range(K)]
            q_alpha_hat = np.quantile(sample_Y_max, 1-alpha) 

            # set radius equal to this quantile, and compute feasibility certificates (probability guarantees)
            sqrt_radius = q_alpha_hat
            sqrt_N = math.sqrt(N)
            constant_term = sqrt_radius / sqrt_N
            gamma = [p_hat[i] - constant_term * math.sqrt(sigma_hat[i]) for i in range(m)]

            # record for how many solutions this gamma is a valid lower bound on p
            num_invalid = sum(gamma[i] > p_true[i] for i in range(m))
            freq_invalid.append(num_invalid)

            # we also compute the old lowerbound for comparison
            r_old = 1/N*scipy.stats.chi2.ppf(1-alpha, 1)
            gamma_old = [max(p_hat[i] - math.sqrt(p_hat[i]*(1-p_hat[i])*r_old), 0) for i in range(m)]
            num_invalid_old = sum(gamma_old[i] > p_true[i] for i in range(m))
            freq_invalid_old.append(num_invalid_old)

            # finally compute the mean absolute error of the certificates
            mean_abs_error.append(np.mean(np.abs(np.subtract(gamma,p_true))))
            mean_abs_error_old.append(np.mean(np.abs(np.subtract(gamma_old,p_true))))

        # compute empirical frequency that an individual lb is invalid and frequency that >1 in a collection of lbs is invalid
        empirical_alpha_individual = sum(freq_invalid[rep] for rep in range(num_reps)) / (m*num_reps)
        empirical_alpha_collective = sum(freq_invalid[rep] > 0 for rep in range(num_reps)) / num_reps
        empirical_MAE = sum(mean_abs_error[rep] for rep in range(num_reps)) / num_reps

        empirical_alpha_individual_old = sum(freq_invalid_old[rep] for rep in range(num_reps)) / (m*num_reps)
        empirical_alpha_collective_old = sum(freq_invalid_old[rep] > 0 for rep in range(num_reps)) / num_reps
        empirical_MAE_old = sum(mean_abs_error_old[rep] for rep in range(num_reps)) / num_reps

        output_data[(m, N)] = {
            "empirical_alpha_individual": empirical_alpha_individual,
            "empirical_alpha_collective": empirical_alpha_collective,
            "empirical_MAE": empirical_MAE,
            "empirical_alpha_individual_old": empirical_alpha_individual_old,
            "empirical_alpha_collective_old": empirical_alpha_collective_old,
            "empirical_MAE_old": empirical_MAE_old
        }

        with open(output_folder / f'results_{output_file_name}_new.txt', 'w+') as f:
            f.write(str(output_data[(m, N)]))


table = make_table(m_values, N_values, alpha, num_reps, K, seed, output_data)
print(table)