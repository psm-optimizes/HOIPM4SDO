![Python](https://img.shields.io/badge/python-3.10+-blue)

# HOIPM4SDO
Implementation of Higher-order Interior Point Method (IPM) of Semidefinite Optimization (SDO)

## Motivation

This repository implements higher-order interior-point methods (HOIPMs)
for semidefinite optimization (SDO), with a focus on improving convergence
near non-strictly complementary solutions.

## Features

- Higher-order IPM based on the AHO direction and full Newton system
- Support for feasible starts: init_prep() function centers the given inital solution on the central path with gap equal to 1 
- Includes some pathological instances of SDO for testing
- Logging of duality gap and convergence behavior
- Code available for comparison against standard solvers including MOSEK and SDPA in Python and SeDuMi and SDPT3 in MATLAB

## Installation
Clone the repository:
```bash
git clone https://github.com/username/HOIPM4SDO.git
cd HOIPM4SDO
```

## Testing
- To test the implementation, make sure you have the required Python packages, included in requirements.txt installed.
- You may choose to work with the runner.py or the jupyter notebook experiments.ipynb.
- Both the runner and the notebook navigate you through choosing a problem to solve and additional arguments to input.
- You can choose the parameters ($\rho,p$) for the algorithm to choose.
- Once the algorithm executes, the output contain the iteration informations includeing gap, current status of algorithm.
- Upon termination, convergence and ratio history tables display, and a figure will be generated and saved at experiments folder.
