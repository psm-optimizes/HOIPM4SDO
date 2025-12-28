from utilities import *
from init_sol_prep import init_prep
from HOIPM import set_ipm_params, HOIPM_FNS, output_process
from problems.elliptope import elliptope
from problems.EdK import EdK
from problems.problem_generator_source import GenSDODims, instance_loader, SDOGen
from problems.PDG import *
from problems.Hauenstein_ssc import Hauen_ssc
import numpy as np
from pathlib import Path


problem = input("Enter the problem of choice (Options: 'elliptope', 'EdK', 'GenSDO', 'PDG', 'HSSC'): ")
print(f"Selected problem: {problem}")

# Call problem data
if problem == 'elliptope':
    n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt = elliptope()

elif problem == 'EdK':
    # Initial Solutions available for n = 3,4,5
    n = int(input("Enter the dimension of EdK problem (available dimensions 3,4, and 5):"))
    n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt = EdK(n)

elif problem == 'GenSDO':
    instance = input("Enter the instance:") 
    GenProblemsLib = Path("problems/GeneratedProblems").resolve()
    mat_files = [f.name for f in GenProblemsLib.glob("*.mat")]
 
    if instance+'.mat' in mat_files: 
        n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt = instance_loader(instance)       
    else:
        n, n_B, n_N, m = map(int, input("Enter n, n_B, n_N, m (exact order separated with space):").split())
        gen_sdo_dims = GenSDODims(n=n, n_B=n_B, n_N=n_N, m=m)
        Xopt,Sopt = -np.eye(n+1), -np.eye(n+1)
        while np.linalg.eigh(Xopt+Sopt)[0][0] < 0:
            (Q,A,b,C,Xopt,yopt,Sopt,X_init,y_init,S_init)= SDOGen(gen_sdo_dims)
    
elif problem == 'PDG':
   instance = input("Enter the name of instance:")
   nSelf, mSelf, A, b, C, X_init, y_init, S_init = PDG_HSDE(instance)
   n = nSelf - 1
   m = mSelf - 1 
   print(f"Uploaded file: {instance}")

elif problem == 'HSSC':
    n, m, A, b, C, X_init, y_init, S_init = Hauen_ssc()

# Prepare the initial solution
X0, y0, S0 = init_prep(n, m, A,b,C, X_init, y_init, S_init)


# Algorithm parameters
HOIPMParams = set_ipm_params(beta=0.5, cent_tol=1e-1, 
                             crawl_step=0.9999, 
                             alpha=1, 
                             precision=1e-10, 
                             norm_thrsh=1e-16, cent_thrsh=1e-12, feas_tol=1e-10, 
                             reduction=1-14, 
                             gamma=1.05)

params = [] # Pairs of (ρ, p)
for i in [1,2]:
  for j in [i, 2*i, 3*i]:
    params.append((i, j))

# Output Dictionaries
muRecs  = {}
Records = {}
orders_used = {}
Drv_norm = {}

# Running the Algorithm
for rho, p in params: 
  print(f'###################### (ρ, p)={rho,p} ######################')
  (XX,yy,SS,mu_reached,muRecs[(rho,p)],Records[(rho,p)], orders_used[(rho,p)], Drv_norm[(rho,p)]) = HOIPM_FNS(A,b,C,m+1,n+1,X0,y0,S0,
                                                                                                              rho,p,
                                                                                                              HOIPMParams);
  if np.linalg.eigh(XX+SS)[0][0] < 0:
    break
  
if 'Xopt' in locals() and 'Sopt' in locals():
    output_process(problem, params, XX, yy, SS, muRecs, Records, orders_used,
                   n, m, A, b, C, Xopt=Xopt, Sopt=Sopt)
else:
    output_process(problem, params, XX, yy, SS, muRecs, Records, orders_used,
                   n, m, A, b, C)
