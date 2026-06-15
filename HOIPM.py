import numpy as np
import scipy as sp
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from datetime import datetime
from utilities import symkron, symet, svec, smat, is_pos_def, draw_table
from dataclasses import dataclass

@dataclass
class IPMParams:
    beta: float
    cent_tol: float
    crawl_step: float
    alpha: float
    precision: float
    norm_thrsh: float
    cent_thrsh:float
    feas_tol: float
    reduction: float
    gamma: float

def set_ipm_params(beta=0.5, cent_tol=1e-1, crawl_step=0.9999, alpha=1, precision=1e-10, norm_thrsh=1e-16, cent_thrsh=1e-12, feas_tol=1e-10, reduction=1-14, gamma=1.05):
    return IPMParams(
        beta=beta,
        cent_tol= cent_tol,
        crawl_step= crawl_step,
        alpha= alpha,
        precision= precision,
        norm_thrsh= norm_thrsh,
        cent_thrsh= cent_thrsh,
        feas_tol= feas_tol,
        reduction= reduction,
        gamma= gamma
    )    

def HOIPM_FNS(A,b,C,m,n,X0,y0,S0,rho,p,HOIPMParams):
  #-------------------------------------------------------------------
  # Initialization
  #-------------------------------------------------------------------

  beta       = HOIPMParams.beta
  cent_tol   = HOIPMParams.cent_tol
  crawl_step = HOIPMParams.crawl_step
  alpha      = HOIPMParams.alpha
  precision  = HOIPMParams.precision
  norm_thrsh = HOIPMParams.norm_thrsh
  cent_thrsh = HOIPMParams.cent_thrsh
  feas_tol   = HOIPMParams.feas_tol  
  reduction  = HOIPMParams.reduction
  gamma      = HOIPMParams.gamma
  

  mu_recsFNS   = []
  X_dist       = []
  S_dist       = []
  iterRecord   = [['Iter', 'Tr(XS)/n', 'Ratio (mu)']]
  n2           = n*n
  ns           = int(0.5*n*(n+1))
  const        = 1
  iter_counter = 0

  drv_norm     = []

  AA = np.zeros((m,ns))
  for i in range(m):
    t = svec(A[i,:,:])
    for j in range(len(t)):
      AA[i,j]=t[j]

  X_drv = np.zeros((p+1,n,n))
  y_drv = np.zeros((p+1,m))
  S_drv = np.zeros((p+1,n,n))

  X = X0
  S = S0
  y = y0

  order_used = []
  #-----------------------------------------------------------------------
  # Higher-order Newton Steps
  #-----------------------------------------------------------------------
  while (True):
    drv_agg   = 0
    print(f'########## Iteration {iter_counter} ##########')

    X_drv[0,] = X
    y_drv[0,] = y
    S_drv[0,] = S
    #-----------------------------------------------------------------------
    # Calculate complementarity
    #-----------------------------------------------------------------------
    mu      = ((np.trace(np.matmul(X,S))) / (n))**(1/rho)
    comp    = (np.trace(np.matmul(X,S))) / (n)

    print('Current Complementarity Gap:', format((np.trace(np.matmul(X,S))) / (n), ".2E"))

    comp_last = comp
    mu_recsFNS  = np.append(mu_recsFNS, (np.trace(np.matmul(X,S)))/(n))

    iterRow     = []
    iterRow.append(iter_counter)
    iterRow.append(format(np.trace(np.matmul(X,S))/n, ".2E"))

    if iter_counter == 0:
      iterRow.append(None)
    else:
      iterRow.append(format((mu_recsFNS[iter_counter])/((mu_recsFNS[iter_counter-1])**gamma),".2E"))

    iterRecord.append(iterRow)
    #-----------------------------------------------------------------------
    # Acquiring Higher Order Information
    #-----------------------------------------------------------------------
    const = 1
    W = (np.matmul(X,S)+np.matmul(S,X))/(2*(mu**rho))

    for j in range(1,p+1):
        #-------------------------------------------------------------------
        # Construct the system
        #-------------------------------------------------------------------
        P   = np.eye(n) # AHO
        EE  = symkron(P,np.matmul(np.linalg.inv(P).T,S))
        FF  = symkron(np.matmul(P,X),np.linalg.inv(P).T)
        M   = np.block([[np.zeros((ns,ns)), AA.T,             np.eye(ns)      ],
                        [AA,                np.zeros((m,m)),  np.zeros((m,ns))],
                        [EE,                np.zeros((ns,m)), FF              ]])

        if (mu > 1):
            if (j <= rho):
                XS = np.zeros((n,n))
                const = const*(rho-j+1)
                for k in range(1,j):
                    XS = XS + math.comb(j,k)*(np.matmul(X_drv[k,], S_drv[j-k,]) + np.matmul(S_drv[k,], X_drv[j-k,]))
                RC = (1/math.factorial(j))*svec((const*((mu)**(rho-j))*W) - 0.5*XS)

            else:
                XS = np.zeros((n,n))
                for k in range(1,j):
                    XS = XS + math.comb(j,k)*(np.matmul(X_drv[k,], S_drv[j-k,]) + np.matmul(S_drv[k,], X_drv[j-k,]))
                RC = (1/math.factorial(j))*svec(-0.5*XS)

        else:
            if (j <= rho):
                XS = np.zeros((n,n))
                const = const*(rho-j+1)
                for k in range(1,j):
                    XS = XS + math.comb(j,k)*(np.matmul(X_drv[k,], S_drv[j-k,]) + np.matmul(S_drv[k,], X_drv[j-k,]))
                RC = (1/math.factorial(j))*(mu**(j))*svec((const*((mu)**(rho-j))*W) - 0.5*XS)

            else:
                XS = np.zeros((n,n))
                for k in range(1,j):
                    XS = XS + math.comb(j,k)*(np.matmul(X_drv[k,], S_drv[j-k,]) + np.matmul(S_drv[k,], X_drv[j-k,]))
                RC = (1/math.factorial(j))*(mu**(j))*svec(-0.5*XS)

        r   = np.block([np.zeros(ns+m), RC])
        #-------------------------------------------------------------------
        # Linear system solver solution
        #-------------------------------------------------------------------
        z       = sp.linalg.solve(M, r)

        drv_agg = drv_agg + np.linalg.norm(z)

        if np.linalg.norm(z) < norm_thrsh:
          prev_order = j-1
          order_used.append(prev_order)
          for order in range(j, p+1):
            X_drv[order,] = np.zeros((n,n))
            y_drv[order,] = np.zeros(m)
            S_drv[order,] = np.zeros((n,n))
          break
        #-------------------------------------------------------------------
        # Calculate delta_x and delta_s
        #-------------------------------------------------------------------
        vdx       = z[:ns]
        vdy       = z[ns:ns+m]
        vds       = z[ns+m:]

        if (mu > 1):
          X_drv[j,] = (math.factorial(j))*smat(vdx)
          y_drv[j,] = (math.factorial(j))*vdy
          S_drv[j,] = (math.factorial(j))*smat(vds)
        else:
          X_drv[j,] = (1/(mu**(j)))*(math.factorial(j))*smat(vdx)
          y_drv[j,] = (1/(mu**(j)))*(math.factorial(j))*vdy
          S_drv[j,] = (1/(mu**(j)))*(math.factorial(j))*smat(vds)
    #-------------------------------------------------------------------
    # Higher Order Approximation
    #-------------------------------------------------------------------
    reduce_param  = crawl_step

    X_approx = np.zeros((n,n)) + X
    y_approx = np.zeros(m)     + y
    S_approx = np.zeros((n,n)) + S


    for j in range(1,p+1):
      X_approx = X_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*X_drv[j,]
      y_approx = y_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*y_drv[j,]
      S_approx = S_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*S_drv[j,]

    if np.linalg.norm(((np.matmul(X_approx,S_approx) + np.matmul(S_approx,X_approx))/(2*((reduce_param*mu)**rho))) - np.eye(n)) <= 2*beta:
      print('FORWARD-TRACKING')
      crawl_steps = 1
      reduce_param  = crawl_step

      while np.linalg.norm(((np.matmul(X_approx,S_approx) + np.matmul(S_approx,X_approx))/(2*((reduce_param*mu)**rho))) - np.eye(n)) <= 2*beta and 2*((reduce_param*mu)**rho) > 0:
        X_record = deepcopy(X_approx)
        y_record = deepcopy(y_approx)
        S_record = deepcopy(S_approx)

        reduce_param = crawl_step*reduce_param 
        crawl_steps = crawl_steps + 1

        X_approx = np.zeros((n,n)) + X
        y_approx = np.zeros(m)     + y
        S_approx = np.zeros((n,n)) + S

        for j in range(1,p+1):
          X_approx = X_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*X_drv[j,]
          y_approx = y_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*y_drv[j,]
          S_approx = S_approx + (((reduce_param*mu - mu)**j)/math.factorial(j))*S_drv[j,]

        if np.linalg.norm(((np.matmul(X_approx,S_approx) + np.matmul(S_approx,X_approx))/(2*((reduce_param*mu)**rho))) - np.eye(n)) > 2*beta or 2*((reduce_param*mu)**rho) < 0:

          X_approx = X_record
          y_approx = y_record
          S_approx = S_record

          break

        if np.trace(np.matmul(X_approx,S_approx))/n < 0:
          print('OVERSHOOT')
          X_approx = X_record
          y_approx = y_record
          S_approx = S_record

          break

        if (np.linalg.eigh(X_approx)[0][0] < 0) or (np.linalg.eigh(S_approx)[0][0]  < 0):
          X_approx = X_record
          y_approx = y_record
          S_approx = S_record

          break

    else:
      print('BACK-TRACK: One Step Back')
      X_approx = np.zeros((n,n)) + X
      y_approx = np.zeros(m)     + y
      S_approx = np.zeros((n,n)) + S
      break

    #--------------------------------------------------------------------
    # Recentering Step
    #--------------------------------------------------------------------
    #-------------------------------------------------------------------
    # Calculate the intermediate terms
    #-------------------------------------------------------------------
    if ((np.trace(np.matmul(X_approx,S_approx)))/(n)) > cent_thrsh:
      print(f'Centering in progress at gap {format(((np.trace(np.matmul(X_approx,S_approx))) / (n)), ".2E")}!')
      iter_cent = 0
      while (True):
        mu_cent     = ((np.trace(np.matmul(X_approx,S_approx))) / (n))

        P   = np.eye(n) # AHO
        EE  = symkron(P,np.matmul(np.linalg.inv(P).T,S_approx))
        FF  = symkron(np.matmul(P,X_approx),np.linalg.inv(P).T)
        M   = np.block([[np.zeros((ns,ns)), AA.T,             np.eye(ns)      ],
                        [AA,                np.zeros((m,m)),  np.zeros((m,ns))],
                        [EE,                np.zeros((ns,m)), FF              ]])

        XS  = symet(np.matmul(X_approx,S_approx),"AHO")
        RC  = svec((mu_cent)*np.eye(n) - XS)
        r   = np.block([np.zeros(ns+m),RC])
        #-------------------------------------------------------------------
        # Linear system solver solution
        #-------------------------------------------------------------------
        z       = np.linalg.solve(M, r)
        #-------------------------------------------------------------------
        # Calculate delta_x and delta_s
        #-------------------------------------------------------------------
        vdx     = z[:ns]
        vdy     = z[ns:ns+m]
        vds     = z[ns+m:]

        delta_X = smat(vdx)
        delta_y = vdy
        delta_S = smat(vds)
        #-------------------------------------------------------------------
        # Updating the Central Solution
        #-------------------------------------------------------------------
        X_approx  = X_approx + alpha*delta_X
        y_approx  = y_approx + alpha*delta_y
        S_approx  = S_approx + alpha*delta_S

        iter_cent = iter_cent + 1
        if np.linalg.norm((np.matmul(X_approx,S_approx) + np.matmul(S_approx,X_approx))/(2*(np.trace(np.matmul(X_approx,S_approx))/n)) - np.eye(n)) <= cent_tol:
          X = X_approx
          y = y_approx
          S = S_approx
          print('# of iteration centering steps:', iter_cent)
          break
    else:
      print(f'Skipping centering at gap {format(((np.trace(np.matmul(X_approx,S_approx))) / (n)), ".2E")}! Centring precision < machine precision!')
      X = X_approx
      y = y_approx
      S = S_approx
      # print('Iteration Final Centrality:', format(np.linalg.norm(((np.matmul(X,S) + np.matmul(S,X))/(2*(np.trace(np.matmul(X,S))/n))) - np.eye(n)), ".2E"))
      # print('Iteration Final Centrality:', format(np.linalg.norm(((np.matmul(X,S) + np.matmul(S,X))) - (2*(np.trace(np.matmul(X,S))/n))*np.eye(n)), ".2E"))
    # print('Gap upon centering:', format(np.trace(np.matmul(X,S))/n, ".2E"))

    drv_norm.append(drv_agg/p)

    #-------------------------------------------------------------------
    # Checking Infeasibility in Inner Iterations
    #-------------------------------------------------------------------
    # Primal
    for i in range(m):
      if np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X))) >= feas_tol:
        print('Primal Infeasibility with error of', np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X))))
    # Dual
    c_temp = np.zeros((n,n))
    for i in range(m):
      c_temp = c_temp + y[i]*A[i,]
    if np.linalg.norm(S - (C - c_temp)) >= feas_tol:
      print('Dual Infeasibility with error of', np.linalg.norm(S - (C - c_temp)))

    #-------------------------------------------------------------------
    # Termination Condition
    #-------------------------------------------------------------------
    iter_counter = iter_counter + 1
    if np.linalg.norm(((np.trace(np.matmul(X,S)))/(n)) - comp_last) < reduction:
      print('#################### Forced Termination ####################')
      print('Very small improvement after', iter_counter, 'iterations')
      break;
    
    if (0 < (np.trace(np.matmul(X,S))) / (n)  <= precision):
      # print('Final Solution Centrality:',np.linalg.norm(((np.matmul(X,S) + np.matmul(S,X))/(2*(np.trace(np.matmul(X,S))/n))) - np.eye(n)))
      # print('Final Solution Comp.:     ', np.trace(np.matmul(X,S))/n)
      #
      print('################## Successful Termination ##################')
      print(f'Terminated after {iter_counter} iterations at gap {format((np.trace(np.matmul(X,S))/n), ".2E")}')
      break

    X_memory = deepcopy(X_drv[0,])
    S_memory = deepcopy(S_drv[0,])

  iterRow     = []
  iterRow.append(iter_counter)
  iterRow.append(format(np.trace(np.matmul(X,S))/n, ".2E"))
  iterRow.append(format(((np.trace(np.matmul(X,S)))/(n))/((mu_recsFNS[iter_counter-1])**gamma),".2E"))

  iterRecord.append(iterRow)

  mu_recsFNS  = np.append(mu_recsFNS, (np.trace(np.matmul(X,S))) / (n))
  #-------------------------------------------------------------------
  # Checking Feasibility Conditions
  #-------------------------------------------------------------------
  ## Primal feasibility
  for i in range(m):
    if np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X))) >= feas_tol:
      print('Primal Infeasibility with error of', np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X))))

  ## Dual feasibility
  c_temp = np.zeros((n,n))
  for i in range(m):
    c_temp = c_temp + y[i]*A[i,]

  if np.linalg.norm(S - (C - c_temp)) >= feas_tol:
    print('Dual Infeasibility with error of', np.linalg.norm(S - (C - c_temp)))

  return (X,y,S,(np.trace(np.matmul(X,S))) / (n), mu_recsFNS, iterRecord, order_used, drv_norm)


def output_process(problem,params,XX,yy,SS,muRecs,Records,orders_used,n,m,A,b,C,Xopt=None,Sopt=None):
  
  if len(params) == 1:
    # Eigenvalues, distances, and feasibility checks
    print('Eig(XX+SS):', np.linalg.eigh(XX+SS)[0])

    if problem not in ['PDG', 'HSSC']:
      if Xopt is not None and Sopt is not None:
        print('\n')
        print('Primal Optimal Solution Error:',format(np.linalg.norm(Xopt-XX),".2E"))
        print('\n')
        print('Dual Optimal Solution Error:',format(np.linalg.norm(Sopt-SS),".2E"))
        print('\n')

    for i in range(m+1):
      print('Primal Infeasibility with error of', format(np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],XX))),".2E"))
    print('\n')

    c_temp = np.zeros((n+1,n+1))
    for i in range(m+1):
      c_temp = c_temp + yy[i]*A[i,]

    print('Dual Infeasibility with error of  ', format(np.linalg.norm(SS - (C - c_temp)),".2E"))
    print('\n')


  # Convergence history and ratios tables
  for rho, p in params:
    table_str = draw_table(Records[(rho,p)])
    print(f'######### (ρ,p)={rho,p} #########')
    print(table_str)


  # Plot the convergence history
  plt.figure(figsize=(12, 8))
  for (rho, p) in params:
    plt.plot(np.arange(len(muRecs[(rho,p)])),muRecs[(rho,p)], label=f'rho = {rho}, p = {max(orders_used[(rho,p)]) if orders_used[(rho,p)] else p}', marker='+')


  # Convergence history of solvers for convenience
  if problem in ['elliptope', 'EdK', 'PDG', 'GenSDO']:
    if problem == 'elliptope':
      mosek  = [1.0e+00, 2.1e-01, 3.3e-02, 4.6e-03, 1.1e-03, 3.1e-04, 8.0e-05, 1.8e-05, 4.0e-06, 9.8e-07, 2.6e-07, 6.9e-08, 1.6e-08, 3.6e-09, 8.2e-10, 2.1e-10, 5.4e-11]
      sedumi = [0.346, 0.0677, 0.0179, 0.00479, 0.00127, 0.000214, 4.65e-05, 9.38e-06, 3.32e-06, 7.59e-07, 1.86e-07, 5.41e-08, 1.53e-08, 4.52e-09, 1.4e-09, 3.7e-10, 1.01e-10, 2.99e-11]
      sdpa   = [1.0, 3.0, 0.8, 0.17, 0.035, 0.0062, 0.0011, 0.00018, 3.5e-05, 7.1e-06, 1.5e-06, 3.3e-07, 7.6e-08, 1.8e-08, 5.4e-09, 1.6e-09, 4.7e-10, 1.4e-10]
      sdpt3  = [1.00e+00, 4.33e+00, 9.67e-02, 1.50e-03, 8.67e-05, 2.60e-05, 2.83e-06, 1.23e-06, 1.13e-07, 4.67e-08, 6.67e-09, 1.37e-09, 2.63e-10] 
      sdpt3h = [1.00e+00, 4.67e-01, 1.00e+00, 8.00e-02, 1.87e-01, 2.03e-02, 1.40e-02, 1.70e-03, 1.03e-03, 1.70e-04, 7.67e-05, 1.60e-05, 6.00e-06, 1.47e-06, 4.67e-07, 7.00e-07, 3.33e-07, 1.63e-07, 8.00e-08, 3.67e-08, 1.83e-08, 9.00e-09, 4.33e-09, 2.10e-09, 1.03e-09, 5.33e-10, 2.50e-10, 1.23e-10, 6.33e-11, 4.67e-11]

      plt.plot(np.arange(len(mosek)), mosek, label='Mosek', color='purple', marker='^')
      plt.plot(np.arange(len(sedumi)), sedumi, label='SeDuMi', color='red', marker='^')
      plt.plot(np.arange(len(sdpa)), sdpa, label='SDPA', color='cyan', marker='*')
      plt.plot(np.arange(len(sdpt3)), sdpt3, label='SDPT3', color='green', marker='x')
      plt.plot(np.arange(len(sdpt3h)), sdpt3h, label='SDPT3-HSD', color='pink', marker='.')

    elif problem == 'EdK':
      if XX.shape[0] == 3:
        mosek  = [1.0e+00, 9.1e-02, 1.7e-02, 4.0e-03, 1.2e-03, 3.9e-04, 1.0e-04, 1.9e-05, 3.6e-06, 8.5e-07, 2.6e-07, 7.6e-08, 1.8e-08, 3.4e-09, 6.9e-10, 1.8e-10]
        sedumi = [1.19, 0.094, 0.027, 0.00761, 0.00212, 0.000587, 0.000162, 4.47e-05, 1.23e-05, 3.38e-06, 9.3e-07, 2.55e-07, 7.01e-08, 1.93e-08, 5.28e-09, 1.45e-09, 3.98e-10, 1.09e-10, 3e-11]
        sdpa   = [1.0, 0.24, 0.051, 0.0087, 0.0015, 0.00025, 4.1e-05, 7.7e-06, 1.6e-06, 3.2e-07, 7e-08, 1.6e-08, 3.7e-09, 1.1e-09, 3.3e-10, 9.5e-11, 2.7e-11]
        sdpt3  = [1.00e+00, 8.67e-02, 1.00e-02, 5.00e-03, 3.67e-04, 1.10e-04, 1.77e-05, 2.37e-06, 8.67e-07, 1.23e-07, 2.33e-08, 4.33e-09, 7.67e-10, 1.40e-10, 2.43e-11] 
        sdpt3h = [1.00e+00, 8.33e-02, 1.37e-02, 3.17e-03, 1.03e-03, 3.07e-04, 9.00e-05, 2.70e-05, 8.00e-06, 2.40e-06, 7.00e-07, 2.13e-07, 6.33e-08, 1.03e-07, 5.33e-08, 2.60e-08]
      elif XX.shape[0] == 4:
        mosek  = [1.0e+00, 1.3e-01, 2.7e-02, 7.0e-03, 2.2e-03, 7.0e-04, 1.5e-04, 3.5e-05, 8.3e-06, 2.4e-06, 6.0e-07, 1.6e-07, 3.5e-08, 8.4e-09, 2.1e-09, 6.1e-10, 1.4e-10]
        sedumi = [1.08, 0.222, 0.0538, 0.0137, 0.00378, 0.00103, 0.00029, 8.95e-05, 3.13e-05, 1.27e-05, 6.29e-06, 1.72e-06, 2.49e-07, 6.41e-08, 1.37e-08, 3.65e-09, 1.7e-09, 4.13e-10, 1e-10, 2.61e-11]
        sdpa   = [1.0, 0.24, 0.055, 0.01, 0.0023, 0.00054, 0.00013, 3.3e-05, 8.4e-06, 2.2e-06, 5.6e-07, 1.5e-07, 3.9e-08, 1e-08, 3.2e-09, 9.7e-10, 2.9e-10, 8.5e-11, 2.4e-11]
        sdpt3  = [1.00e+00, 1.17e-01, 1.92e-02, 1.03e-02, 1.08e-03, 4.25e-04, 5.75e-05, 2.75e-05, 2.75e-06, 1.85e-06, 5.00e-07, 3.00e-07, 3.50e-08, 9.75e-09, 2.18e-09, 5.00e-10, 7.00e-11, 1.65e-11] 
        sdpt3h = [1.00e+00, 1.28e-01, 1.43e-02, 5.50e-03, 8.25e-04, 4.25e-04, 5.75e-05, 3.50e-05, 4.75e-06, 2.75e-06, 3.75e-07, 2.13e-07, 1.47e-07, 6.75e-08, 3.00e-08]
      elif XX.shape[0] == 5:
        mosek  = [1.0e+00, 1.5e-01, 3.7e-02, 9.4e-03, 2.6e-03, 8.5e-04, 2.0e-04, 5.3e-05, 1.3e-05, 3.7e-06, 8.8e-07, 2.5e-07, 5.7e-08, 1.6e-08, 3.8e-09, 2.6e-09, 1.2e-09, 2.9e-10]
        sedumi = [1.0, 0.235, 0.0594, 0.0156, 0.00427, 0.00114, 0.00033, 0.000108, 4.19e-05, 2.04e-05, 4.32e-06, 8.62e-07, 2.24e-07, 7.53e-08, 2.96e-08, 1.64e-08, 4.21e-09, 1.22e-09, 3.06e-10, 1.47e-10, 3.99e-11]
        sdpa   = [1.0, 0.25, 0.058, 0.012, 0.003, 0.00076, 0.0002, 5.3e-05, 1.4e-05, 3.9e-06, 1.1e-06, 3e-07, 8.2e-08, 2.3e-08, 6.3e-09, 2e-09, 6.2e-10, 1.9e-10, 5.8e-11, 1.7e-11]
        sdpt3  = [1.00e+00, 1.54e-01, 3.20e-02, 1.18e-02, 1.26e-03, 6.60e-04, 7.80e-05, 3.80e-05, 4.40e-06, 2.60e-06, 5.60e-07, 1.96e-07, 2.80e-08, 8.80e-09, 1.64e-09, 4.20e-10, 6.40e-11, 1.78e-11] 
        sdpt3h = [1.00e+00, 1.78e-01, 1.66e-02, 1.12e-02, 1.00e-03, 1.28e-03, 1.20e-04, 1.34e-04, 1.32e-05, 1.56e-05, 1.54e-06, 1.84e-06, 1.80e-07, 2.20e-07, 8.20e-08, 3.40e-08, 1.48e-08, 5.20e-09, 2.20e-09]

      plt.plot(np.arange(len(mosek)), mosek, label='Mosek', color='purple', marker='^')
      plt.plot(np.arange(len(sedumi)), sedumi, label='SeDuMi', color='red', marker='^')
      plt.plot(np.arange(len(sdpa)), sdpa, label='SDPA', color='cyan', marker='*')
      plt.plot(np.arange(len(sdpt3)), sdpt3, label='SDPT3', color='green', marker='x')
      plt.plot(np.arange(len(sdpt3h)), sdpt3h, label='SDPT3-HSD', color='pink', marker='.')
    
    elif problem == 'PDG':
      if XX.shape[0] == 18:
        mosek = [1.0, 0.16, 0.025, 0.0051, 0.0011, 0.00019, 4.5e-05, 9.3e-06, 2e-06, 4.2e-07, 8.6e-08, 1.9e-08, 4.7e-09, 1.1e-09, 2.1e-10]
      elif XX.shape[0] == 24:
        mosek = [1.0, 0.16, 0.018, 0.0041, 0.00063, 0.0001, 1.8e-05, 2.9e-06, 5.1e-07, 8.5e-08, 1.7e-08, 2.2e-09, 1.4e-09, 4.5e-10, 5.5e-11]
      elif XX.shape[0] == 30:
        mosek = [1.0, 0.36, 0.031, 0.0029, 0.00034, 5e-05, 9e-06, 1.4e-06, 2.3e-07, 4.1e-08, 4.8e-09, 2.4e-09, 4.9e-10, 1e-10]
      elif XX.shape[0] == 36:
        mosek = [1.0, 0.38, 0.034, 0.0065, 0.0012, 0.00019, 3.7e-05, 6.6e-06, 1.3e-06, 2.2e-07, 3.5e-08, 6.5e-09, 2.9e-09, 7.5e-10, 2.3e-10, 2.3e-10, 2.2e-10, 5.5e-11]
      elif XX.shape[0] == 42:
        mosek = [1.0, 0.4, 0.04, 0.0051, 0.0014, 0.00016, 3.1e-05, 1.7e-05, 2e-06, 2.4e-07, 4.8e-08, 8.2e-09, 2e-09, 4.1e-10]
      elif XX.shape[0] == 48:
        mosek = [1.0, 0.42, 0.045, 0.0065, 0.00099, 0.00027, 3.2e-05, 3.5e-06, 6.6e-07, 8.7e-08, 1.5e-08, 6.2e-09, 1.6e-09, 3.9e-10, 3.5e-10, 3.3e-10, 3.2e-10, 3.2e-10, 3.2e-10, 3.2e-10, 1.4e-10, 1.3e-10]

      plt.plot(np.arange(len(mosek)), mosek, label='Mosek', color='purple', marker='^')

    elif problem == 'GenSDO':
      mosek = [1.0, 0.051, 0.0051, 0.00033, 5e-05, 1.2e-05, 3.6e-06, 1.2e-06, 2.9e-07, 5.8e-08, 1.2e-08, 3.1e-09] # ND-1
      plt.plot(np.arange(len(mosek)), mosek, label='Mosek', color='purple', marker='^')

      # ND-1
      # mosek = [1.0, 0.051, 0.0051, 0.00033, 5e-05, 1.2e-05, 3.6e-06, 1.2e-06, 2.9e-07, 5.8e-08, 1.2e-08, 3.1e-09]
      # ND-2
      # mosek = [1.0, 0.2, 0.054, 0.0095, 0.0025, 0.00048, 8.2e-05, 1.9e-05, 5.2e-06, 1.1e-06, 2.5e-07, 6.2e-08, 1.7e-08, 4.2e-09, 9.7e-10]
      # ND-3
      # mosek = [1.0, 0.23, 0.03, 0.0072, 0.0026, 0.00045, 5.5e-05, 8.9e-06, 1.4e-06, 3.2e-07, 9e-08, 2.6e-08, 6.1e-09, 1.1e-09]
      # ND-4
      # mosek = [1.0, 0.21, 0.032, 0.0038, 0.0012, 0.00033, 8.1e-05, 2.3e-05, 6.1e-06, 1.6e-06, 3.5e-07, 7.3e-08, 1.7e-08, 4.4e-09, 1.2e-09, 3.1e-10]
      # ND-5
      # mosek = [1.0, 0.12, 0.0075, 0.0019, 0.00097, 0.00025, 3.7e-05, 7.5e-06, 2e-06, 5.2e-07, 1.3e-07, 2.8e-08, 6.5e-09, 1.6e-09, 4.3e-10, 1.1e-10]
      # ND-6
      # mosek = [1.0, 0.12, 0.03, 0.01, 0.0023, 0.00052, 0.00011, 1.9e-05, 3.2e-06, 6.8e-07, 1.6e-07, 5.5e-08, 1.7e-08, 3.8e-09, 7.3e-10]

      # D-1
      # mosek = [1.0, 0.22, 0.026, 0.00019, 8.1e-07, 4.2e-08, 1.1e-09]
      # D-2
      # mosek = [1.0, 0.13, 0.015, 0.0035, 0.0013, 0.0005, 7.3e-05, 6.8e-06, 1.1e-07, 7.1e-09, 2.3e-10]
      # D-3
      # mosek = [1.0, 0.27, 0.033, 0.0075, 0.00021, 1.7e-05, 1.2e-06, 3.7e-08, 4.1e-10]

  plt.xlabel('Iteration')
  plt.ylabel('Normalized Complementarity Gap (log)')
  
  if problem == 'elliptope':
    max_len = len(sdpt3h)
  elif problem == 'EdK':
    if XX.shape[0] == 3:
      max_len = len(sedumi)
    elif XX.shape[0] == 4:
       max_len = max(len(muRecs[(rho, p)]) for rho, p in params)
    elif XX.shape[0] == 5:
      max_len = len(sedumi)
  else:
    if mosek:
      max_len = max(len(mosek), max(len(muRecs[(rho, p)]) for rho, p in params))
    else:
      max_len = max(len(muRecs[(rho, p)]) for rho, p in params)
  
  plt.xticks(range(0, max_len, 2))

  plt.yticks([10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16])
  plt.yscale('log')
  ##################################
  all_values = [v for rho,p in params for v in muRecs[(rho,p)]]

  ymin = min(all_values)
  ymax = max(all_values)

  upper = max(ymax, 10)
  lower = ymin / 10

  plt.ylim(lower, upper)
  ##################################
  plt.legend()
  plt.grid()
  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  plt.savefig(f'experiments/{problem}_size{n+1}_{timestamp}')
  plt.show()