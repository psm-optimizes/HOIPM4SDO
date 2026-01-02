# HOIPM4SDO
Implementation of Higher-order Interior Point Method (IPM) for Pathological Instances of Semidefinite Optimization

## Motivation

This repository implements higher-order interior-point methods (HOIPMs)
for semidefinite optimization (SDO), with a focus on improving convergence
near non-strictly complementary solutions.

## Features

- Higher-order IPM based on the AHO direction and full Newton system
- Support for infeasible and feasible starts
- Comparison against standard IPMs
- Logging of duality gap and convergence behavior

## Installation

Clone the repository:
```bash
git clone https://github.com/username/HOIPM4SDO.git
```

## Testing
To test the implementation, make sure you have the required Python packages, included in requirements.txt installed.

![Python](https://img.shields.io/badge/python-3.10+-blue)