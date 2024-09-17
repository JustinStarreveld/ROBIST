# ROBIST: Robust Optimization by Iterative Scenario Sampling and Statistical Testing

[![License: MIT][license-badge]][license]
[![DOI][Zenodo-badge]][Zenodo-url]

This repository provides the code for applying _ROBIST_, a simple, yet effective, data-driven algorithm for optimization under parametric uncertainty. 
The methodology and numerical experiments are described in detail in an accompanying paper, available online [here](https://optimization-online.org/?p=24671). 
The method was developed by Justin Starreveld, Guanyu Jin, Dick den Hertog and Roger Laeven.

## Code

The code is written in `Python`, version 3.10.9. The dependency packages are listed in `pyproject.toml`. The ROBIST algorithm is implemented in `src/robist/robist.py`.

## Installation

The recommended way to install `robist` in your (virtual) environment is with
Python's `pip`:
```
pip install robist
```
To be able to run the numerical experiments, make sure to clone the repository and install with the examples-requirements:
```
git clone https://github.com/JustinStarreveld/ROBIST.git
cd robist
pip install ".[examples]"
```

## Numerical Experiments

1) Toy Problem (abbreviated as tp). 
We compare ROBIST with the methods of Calafiore & Campi (2005) and Yanıkoglu & den Hertog (2013) in  [`examples/tp_experiments.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/tp_experiments.py). 

2) Altered Toy Problem (abbreviated as tp2).
We analyze the performance of ROBIST in more detail in [`examples/tp2_analysis.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/tp2_analysis.py), [`examples/tp2_extra_experiments_1.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/tp2_extra_experiments_1.py) and [`examples/tp2_extra_experiments_2.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/tp2_extra_experiments_2.py).

3) Linear CCP Problem (abbreviated as jiang2022).
We compare ROBIST with the SAA-based methods of Ahmed et al. (2017) and Jiang and Xie (2022) in [`examples/jiang2022_experiments.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/jiang2022_experiments.py).

4) Weighted Distribution Problem (abbreviated as wdp). 
We compare ROBIST with the scenario optimization methods of Calafiore & Campi (2005), Caré et al. (2014), Calafiore (2016) and Garatti et al. (2022) in [`examples/wdp_experiments.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/wdp_experiments.py).

5) Portfolio Management Problem (abbreviated as pm). 
We compare ROBIST with the data-driven robust optimization approach proposed by Bertsimas et al. (2018) and the scenario optimization approach of Calafiore (2013) in [`examples/pm_experiments.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/pm_experiments.py).

6) Two-Stage Lot-Sizing Problem (abbreviated as ls). 
We compare ROBIST with the method of Vayanos et al. (2012) in [`examples/ls_experiments.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/ls_experiments.py).

For more information on these experiments and the results we refer to [preprint-paper].

## Illustrative Example

We demonstrate ROBIST using the illustrative example as described in Section 2.2 of the paper.  

Here we apply the algorithm to the following toy problem from Yanıkoglu & den Hertog (2013):

$$
\begin{align\*}
    \max_{x_1,x_2 \leq 1}~&x_1 + x_2 \\
    \text{s.t.}~&z_1x_1+ z_2x_2 \leq 1,
\end{align\*}
$$

where $z_1$ and $z_2$ are uncertain parameters, both uniformly distributed with support $[-1,1]$.

Suppose we have access to a data set of $N=300$ realizations of $(\tilde{z}_1, \tilde{z}_2)$ and would like the solution to be feasible with probability of at least 90%. 
We illustrate the application of ROBIST for this toy problem using the following figures. 

First, we randomly split the data set into three equal-sized sets $\mathcal{D}^{train}\_{N_1}$, $\mathcal{D}^{\text{valid}}\_{N_2}$ and $\mathcal{D}^{\text{test}}\_{N_3}$, each containing $100$ scenarios.

![Data](https://github.com/JustinStarreveld/ROBIST/raw/main/docs/illustrative_figures/Illustrate_data_split_N=300.png)

We initialize the algorithm by optimizing for the expected/nominal scenario, i.e., $\bar{\mathbf{z}} = (z_1, z_2) = (0,0)$. This provides an initial solution: $\mathbf{x}\_{0} = (x_1, x_2) = (1,1)$ with an objective value of 2.
The next step is to use the training data $\mathcal{D}^{\text{train}}\_{N_1}$ to evaluate the robustness of $\mathbf{x}\_{0}$. This evaluation is illustrated in the following figure.

  ![At iteration 0](https://github.com/JustinStarreveld/ROBIST/raw/main/docs/illustrative_figures/Illustrate_wConstraint_iter=0_N=100_alpha=0.01.png)
  
We find that $\mathbf{x}\_{0}$ does not meet the desired level of robustness, thus the algorithm will randomly pick one of the 13 currently violated scenarios (indicated by red stars) and add the scenario to our set of sampled scenarios to be optimized over.
Suppose scenario $\hat{\mathbf{z}}^{11} = (0.96, 0.60)$ is chosen and we retrieve solution: $\mathbf{x}\_{1} = (0.4,1)$ with an objective value of 1.4.
Again, we can evaluate the robustness of our newly generated solution $\mathbf{x}\_{1}$ using the scenarios in $\mathcal{D}^{\text{train}}\_{N_1}$. This is depicted in the figure below.
  
  ![At iteration 1](https://github.com/JustinStarreveld/ROBIST/blob/main/docs/illustrative_figures/Illustrate_wConstraint_iter=1_N=100_alpha=0.01.png)
  
We find that $\mathbf{x}\_{1}$ exceeds our desired level of robustness, thus the algorithm will remove a randomly picked scenario from our set of sampled scenarios in the following iteration. 
The algorithm continues adding or removing scenarios and evaluating the resulting solutions on $\mathcal{D}^{\text{train}}\_{N_1}$ in this manner until either the time limit or iteration limit is reached. 

Once the stopping criteria is reached, we use the validation data $\mathcal{D}^{\text{valid}}\_{N_2}$ to properly evaluate each solution $\mathbf{x}\_{i}$ and obtain valid "feasibility certificates". 
These evaluations can then be used to construct a trade-off curve and aid in choosing a final solution (the orange line in the figure below depicts such a trade-off curve, where each orange square
represents a non-dominated solution with respect to the validation data). Recall that we would like our solution to be feasible with probability $\geq 90\%$, thus we select the solution with the highest 
objective value while requiring that the proxy certificate derived from the validation data is greater than or equal to 0.90. Finally, we utilize $\mathcal{D}^{\text{test}}\_{N_3}$ to provide
a valid feasibility certificate for this solution. In this example, the algorithm returns a solution with objective value 1.37 and feasibility certificate 0.93.  
  
  ![Trade-off curve](https://github.com/JustinStarreveld/ROBIST/raw/main/docs/illustrative_figures/TradeOffCurves_N=300_alpha=0.01_epsilon=0.1_iMax=100.png)
  
The script used to create the figures in this illustrative example is [`examples/tp_illustrative_plots.py`](https://github.com/JustinStarreveld/ROBIST/blob/main/examples/tp_illustrative_plots.py).

## Contact Information
Our code is not flawless. In case you have any questions or suggestions, please reach us at j.s.starreveld@uva.nl. 

## Citation
Was our software useful to you? You can cite us using:

```
@misc{ROBIST,
  doi = {10.5281/zenodo.10143595},
  year = {2023},
  author = {Justin Starreveld, Guanyu Jin, Dick den Hertog and Roger JA Laeven},
  title = {ROBIST: Robust Optimization by Iterative Scenario Sampling and Statistical Testing},
  url = {https://github.com/JustinStarreveld/ROBIST}
}
```

[license]:              		https://opensource.org/license/mit/
[license-badge]:        		https://img.shields.io/badge/license-MIT-blue
[Zenodo-url]:           		https://doi.org/10.5281/zenodo.10143595
[Zenodo-badge]: 				https://zenodo.org/badge/494070848.svg
[preprint-paper]:          		https://optimization-online.org/?p=24671
