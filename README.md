![Python](https://img.shields.io/badge/python-3.10+-blue)

# HOIPM4SDO
Implementation of a Higher-order Interior Point Method (IPM) for Semidefinite Optimization (SDO)

## 🎯 Overview

This repository implements higher-order interior-point methods (HOIPMs)
for semidefinite optimization (SDO), with a focus on improving convergence
near non-strictly complementary solutions. The algorithm is exhaustively explained in the paper 
'On the complexity of semi-definite optimization: a super-linearly convergent interior point method',
available at 'https://arxiv.org/abs/XXXX.XXXXX'.

## Features

- Higher-order IPM based on the AHO direction and full Newton system.
- Support for feasible starts: init_prep() function centers the given initial solution on the central path with gap equal to 1.
- Includes some pathological instances of SDO for testing.
- Logging of duality gap and convergence behavior.
- Code available for comparison against standard solvers, including MOSEK and SDPA in Python and SeDuMi and SDPT3 in MATLAB.

## 🚀 Installation
### 1. Clone the Repository
```bash
git clone https://github.com/username/HOIPM4SDO.git
cd HOIPM4SDO
```
### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
###  3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 📦 Dependencies

The project requires the following packages:
- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `pandas` - Data analysis and CSV handling
- `matplotlib` - Plotting and visualization
- `cvxpy` - Modeling language and solver interface

## Testing
- You may choose to work with the `runner.py` or the Jupyter notebook `experiments.ipynb`.
- Both the runner and the notebook navigate you through choosing a problem to solve and additional arguments to input.
- The function 'init_prep()' is used for centering the initial solution and fixing the initial gap.
- In both runner and notebook, you can choose the parameters ($\rho,p$) for the algorithm to use.
- Once the algorithm executes, the output contains the iteration information, including the gap and the current status of the algorithm.
- Upon termination, convergence and ratio history tables display, and a figure will be generated and saved in the experiments folder.
- To include other solvers (MOSEK, etc.), you need to run the code for using those solvers, provided in the 'solvers' folder. 
- Note that for MOSEK, you need a license, and SeDuMi and SDPT3 should be executed in MATLAB.
- For MOSEK and SDPA, there are helper functions to extract the gaps from their files. Please refer to 'extract_mu()' and 
'extract_mu_spda()' functions in 'solvers/cvxpy_solve.py', for your reference. 
