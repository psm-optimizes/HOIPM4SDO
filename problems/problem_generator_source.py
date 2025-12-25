from scipy.io import loadmat
from dataclasses import dataclass
import numpy as np
from scipy.stats import ortho_group
from scipy.io import savemat

@dataclass
class GenSDODims:
    n: int
    n_B: int
    n_N: int
    m: int

def gen_sdo_dim(n, n_B, n_N, m):
    return GenSDODims(
        n= n,
        n_B= n_B,
        n_N= n_N,
        m= m
    )    

def instance_loader(instance):

  Problem = loadmat(f'/Users/pooyasm/Desktop/HIPM Experiments/Repo/HOIPM4SDO/problems/GeneratedProblems/{instance}')
  n      = Problem['n'][0][0]
  m      = Problem['m'][0][0]
  A      = Problem['A']
  b      = Problem['b'][0]
  C      = Problem['C']
  X_init = Problem['X_init']
  y_init = Problem['y_init'][0]
  S_init = Problem['S_init'] 
  Xopt   = Problem['Xopt']
  yopt   = Problem['yopt'][0]
  Sopt   = Problem['Sopt']

  return n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt

def SDOGen(gen_sdo_dims):
  '''
  Choose dimensions m,n with m < n(n+1)/2 (dim. of generated problem: (m+1)*(n+1)).
  Choose n_B, n_N such that n_B + n_N <= n.
  Define sets B = {1,...,n_B}, T = {n_B+1,...,n - n_N}, and N = {n - n_N +1,..., n}.
  '''
  n   = gen_sdo_dims.n
  n_B = gen_sdo_dims.n_B
  n_N = gen_sdo_dims.n_N
  m   = gen_sdo_dims.m

  if m >= n*(n+1)/2:
    print("Dimension Violation.")

  if 0.5*((n+1)-n_B)*((n+1)-n_B+1) > 0.5*(n+1)*((n+1)+1) - (m+1) or 0.5*((n+1)-(n_N+1))*((n+1)-(n_N+1)+1) > (m+1):
    print('Dimension conditions for primal non-degeneracy are violated.')

  if 0.5*((n+1)-(n_N+1))*((n+1)-(n_N+1)+1) > (m+1):
    print('Dimension conditions for dual non-degeneracy are violated.')

  n_T = n - n_B - n_N

  B = np.arange(n_B)
  if n_T != 0:
      T = np.arange(B[-1]+1, n_B+n_T)
  else:
    T = []
  N = np.arange(len(B)+len(T), n)

  ###
  sigma_B  = np.random.rand(n_B)
  lambda_N = np.random.rand(n_N)

  Sigma_B  = np.diag(sigma_B)
  Lambda_N = np.diag(lambda_N)

  ### 
  Qhat  = ortho_group.rvs(n)

  ### 
  Dx = np.block([[Sigma_B,              np.zeros((n_B, n_T)), np.zeros((n_B, n_N))],                                      ### Check
                 [np.zeros((n_T, n_B)), np.zeros((n_T, n_T)), np.zeros((n_T, n_N))],
                 [np.zeros((n_N, n_B)), np.zeros((n_N, n_T)), np.zeros((n_N, n_N))]])
  Ds = np.block([[np.zeros((n_B, n_B)), np.zeros((n_B, n_T)), np.zeros((n_B, n_N))],                                      ### Check
                 [np.zeros((n_T, n_B)), np.zeros((n_T, n_T)), np.zeros((n_T, n_N))],
                 [np.zeros((n_N, n_B)), np.zeros((n_N, n_T)), Lambda_N            ]])

  Xhat = np.matmul(np.matmul(Qhat, Dx),Qhat.T)
  Shat = np.matmul(np.matmul(Qhat, Ds),Qhat.T)

  ###
  yhat = np.random.rand(m)
  ###
  Ahat = np.zeros((m,n,n))
  bhat = np.zeros(m)
  Chat = np.zeros((n,n))
  c_temp = np.zeros((n,n))

  Pi = np.zeros((m,n,n))

  gamma_NB = np.random.rand(n_N, n_B)
  while np.isclose(np.linalg.norm(np.matmul(Qhat[:,n_B+n_T:],gamma_NB)-np.zeros((n,n_B))),0):
    gamma_NB = np.random.rand(n_N, n_B).T
  gamma_TT = np.diag(np.random.randint(1,10,size=n_T))
  gamma_NT = np.random.rand(n_N, n_T)

  gNN      = np.random.rand(n_N,n_N)
  gamma_NN = (gNN + gNN.T)/2

  Gamma = np.block([[np.zeros((n_B, n_B)), np.zeros((n_B, n_T)), gamma_NB.T],                                      ### Check
                    [np.zeros((n_T, n_B)),             gamma_TT, gamma_NT.T],
                    [gamma_NB,                         gamma_NT,   gamma_NN]])


  Ahat[0,] = np.matmul(np.matmul(Qhat, Gamma),Qhat.T)
  for i in range(1,m):
    Pi = np.random.randn(n,n_B)
    Ahat[i,] = (np.matmul(Pi,Qhat[:,:n_B].T) + np.matmul(Pi,Qhat[:,:n_B].T).T) /2

  ##################################################

  for i in range(m):
    bhat[i]  = np.trace(np.matmul(Ahat[i,],Xhat))
    c_temp = c_temp + yhat[i]*Ahat[i,]

  Chat = c_temp + Shat
  ###
  Q = np.block([[Qhat,            np.zeros((n,1))],
                [np.zeros((1,n)), 1              ]])
  ###
  sigma_B0   = np.random.rand(n_B) + 0.5
  sigma_T0   = np.random.rand(n_T) + 0.5
  sigma_N0   = np.random.rand(n_N) + 0.5
  sigma_n1_0 = np.random.rand()    + 0.5

  Sigma_B0 = np.diag(sigma_B0)
  Sigma_T0 = np.diag(sigma_T0)
  Sigma_N0 = np.diag(sigma_N0)
  ###
  DX = np.block([[Sigma_B,              np.zeros((n_B ,n_T)), np.zeros((n_B ,n_N)), np.zeros((n_B ,1))],                          ### Check
                 [np.zeros((n_T ,n_B)), np.zeros((n_T ,n_T)), np.zeros((n_T ,n_N)), np.zeros((n_T ,1))],
                 [np.zeros((n_N ,n_B)), np.zeros((n_N ,n_T)), np.zeros((n_N ,n_N)), np.zeros((n_N ,1))],
                 [np.zeros((1 ,n_B)),   np.zeros((1 ,n_T)),   np.zeros((1 ,n_N)),   np.zeros(1)       ]])
  Xopt = np.matmul(np.matmul(Q, DX),Q.T)

  Dx0 = np.block([[Sigma_B0,             np.zeros((n_B ,n_T)), np.zeros((n_B ,n_N)), np.zeros((n_B ,1))],                         ### Check
                  [np.zeros((n_T ,n_B)), Sigma_T0,             np.zeros((n_T ,n_N)), np.zeros((n_T ,1))],
                  [np.zeros((n_N ,n_B)), np.zeros((n_N ,n_T)), Sigma_N0,             np.zeros((n_N ,1))],
                  [np.zeros((1 ,n_B)),   np.zeros((1 ,n_T)),   np.zeros((1 ,n_N)),   sigma_n1_0        ]])
  X0 = np.matmul(np.matmul(Q, Dx0),Q.T)
  ###
  lambda_B0 = np.random.rand(n_B) + 0.5
  lambda_T0 = np.random.rand(n_T) + 0.5
  lambda_N0 = np.random.rand(n_N) + 0.5

  Lambda_B0 = np.diag(lambda_B0)
  Lambda_T0 = np.diag(lambda_T0)
  Lambda_N0 = np.diag(lambda_N0)
  ###

  delta = np.trace(np.matmul((Sigma_B0-Sigma_B),Lambda_B0))+np.trace(np.matmul(Sigma_T0,Lambda_T0))+np.trace(np.matmul((Lambda_N0-Lambda_N),Sigma_N0))

  ###
  lambda_n1_0 = max(0, -delta/sigma_n1_0) + np.random.rand()
  lambda_n1   = (delta/sigma_n1_0) + lambda_n1_0
  ###
  DS = np.block([[np.zeros((n_B, n_B)), np.zeros((n_B, n_T)), np.zeros((n_B ,n_N)), np.zeros((n_B, 1))],
                 [np.zeros((n_T, n_B)), np.zeros((n_T, n_T)), np.zeros((n_T ,n_N)), np.zeros((n_T, 1))],
                 [np.zeros((n_N, n_B)), np.zeros((n_N, n_T)), Lambda_N,             np.zeros((n_N, 1))],
                 [np.zeros((1,   n_B)), np.zeros((1,   n_T)), np.zeros((1,  n_N)),  lambda_n1         ]])
  Sopt = np.matmul(np.matmul(Q, DS),Q.T)

  Ds0 = np.block([[Lambda_B0,            np.zeros((n_B, n_T)), np.zeros((n_B, n_N)), np.zeros((n_B, 1))], 
                  [np.zeros((n_T, n_B)), Lambda_T0,            np.zeros((n_T, n_N)), np.zeros((n_T, 1))],
                  [np.zeros((n_N ,n_B)), np.zeros((n_N, n_T)), Lambda_N0,            np.zeros((n_N, 1))],
                  [np.zeros((1 ,n_B)),   np.zeros((1, n_T)),   np.zeros((1, n_N)),   lambda_n1_0       ]])
  S0 = np.matmul(np.matmul(Q, Ds0),Q.T)
  ###
  y0 = np.random.rand(m+1)                                                    
  while y0[m] == 0:
    y0[m] = np.random.rand()
  yopt = np.block([yhat, 0])
  ###
  Sig_alpha = np.block([[Sigma_B - Sigma_B0,    np.zeros((n_B, n_T)),  np.zeros((n_B, n_N))],
                        [np.zeros((n_T, n_B)), -Sigma_T0,              np.zeros((n_T, n_N))],
                        [np.zeros((n_N ,n_B)),  np.zeros((n_N, n_T)), -Sigma_N0            ]])
  alpha = np.zeros(m)
  for i in range(m):
    alpha[i] = (1/sigma_n1_0)*np.trace(np.matmul(np.matmul(Ahat[i],Qhat),np.matmul(Sig_alpha,Qhat.T)))
  ###
  A = np.zeros((m+1,n+1,n+1))
  for i in range(m):
    A[i,] = np.block([[Ahat[i,],        np.zeros((n,1))],
                      [np.zeros((1,n)), alpha[i]       ]])

  Lambda_A = np.block([[-Lambda_B0,            np.zeros((n_B, n_T)), np.zeros((n_B, n_N)), np.zeros((n_B, 1))     ],  
                       [np.zeros((n_T, n_B)), -Lambda_T0,            np.zeros((n_T, n_N)), np.zeros((n_T, 1))     ],
                       [np.zeros((n_N, n_B)),  np.zeros((n_N, n_T)), Lambda_N - Lambda_N0, np.zeros((n_N, 1))     ],
                       [np.zeros((1, n_B)),    np.zeros((1, n_T)),   np.zeros((1, n_N)),   lambda_n1 - lambda_n1_0]])
  yA = np.zeros((n+1,n+1))
  for i in range(m):
    yA = yA + (yhat[i] - y0[i])*A[i,]

  A[m,] = (1/y0[m])*(yA + Sopt-S0)
  ###
  yalpha = lambda_n1
  for i in range(m):
    yalpha = yalpha + yhat[i]*alpha[i]                                      
  C = np.block([[Chat,            np.zeros((n,1))],
                [np.zeros((1,n)), yalpha         ]])
  ###
  b = np.zeros(m+1)
  for i in range(m):
    b[i] = bhat[i]
  b[m] = np.trace(np.matmul(A[m,], Xopt))

  '''
  Return (A,b,C) with optimal solution (Xopt, yopt, Sopt) and interior solution (X0,y0,S0).
  '''
  return (Q,A,b,C,Xopt,yopt,Sopt,X0,y0,S0)