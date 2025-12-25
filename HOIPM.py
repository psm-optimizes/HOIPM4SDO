import numpy as np
import scipy as sp
from copy import deepcopy
import math
import matplotlib.pyplot as plt
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
        RC  = svec((mu_cent)*W - XS)
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
        if np.linalg.norm((np.matmul(X_approx,S_approx) + np.matmul(S_approx,X_approx))/(2*(np.trace(np.matmul(X_approx,S_approx))/n)) - W) <= cent_tol:  # Changed identity to W
          X = X_approx
          y = y_approx
          S = S_approx
          # print('# of iteration centering steps:', iter_cent)
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
    plt.plot(np.arange(len(muRecs[(rho,p)])),muRecs[(rho,p)], label=f'rho = {rho}, p* = {max(orders_used[(rho,p)]) if orders_used[(rho,p)] else p}', marker='+')

  plt.xlabel('Iteration')
  plt.ylabel('Complementarity Gap (log)')

  # max_len = max(max(max(len(muRecs[(rho, p)]) for rho, p in params), len(mu_mosek)), len(ell_HSD))
  # max_len = max(max(len(muRecs[(rho, p)]) for rho, p in params), len(mu_mosek))
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
  plt.show()