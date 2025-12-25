import numpy as np
import scipy
from problems.PDG_library import *
from utilities import *
import scipy.io
from scipy.linalg import block_diag
from pathlib import Path
import os

def matgen(m,i):
  mat = np.zeros((m,m))
  mat[i,i] = 1
  return mat

def eyegen(n,j,k):
  mat = np.zeros((n,n))
  mat[j,k] = 1
  return mat

def PDG_HSDE(instance):
    # Load the .mat file
    PDGProblemsLib = Path("problems/PDG_library").resolve()
    pdg_files = [f.name for f in PDGProblemsLib.glob("*.mat")]
    
    if instance+'.mat' in pdg_files: 
        data_dir = 'problems/PDG_library'
        file_name = f'{instance}.mat'
        full_path = os.path.join(data_dir, file_name)
        data = scipy.io.loadmat(full_path)

        # Dimensions
        n = int(data['n'][0, 0])
        # print('n:', n)
        m = int(data['m'][0, 0])
        # print('m:', m)

        # A
        Ain = data['Amat']
        A = np.zeros((m,n,n))
        for i in range(m):
            A[i,] = Ain[i,][0].astype(np.float64)

        # C
        C = data['C'].astype(np.float64)

        # b
        b = data['bSDP'].astype(np.float64)[0]

        # Temporary changing n and m
        n = n - 1
        m = m - 1

        aa = np.zeros((m+1,int(0.5*(n+1)*(n+2))))
        for i in range(m+1):
            t = svec(A[i,:,:])
            for j in range(len(t)):
                aa[i,j]=t[j];

        '''
        Parameters
        '''

        # Postive duality gap problems
        n  = int(data['n'][0, 0])
        m  = int(data['m'][0, 0])

        # print('(n,m):', (n,m))

        ns = int(0.5*n*(n+1))

        bb = np.zeros(m)
        for i in range(m):
            bb[i] = b[i] - np.trace(A[i,])-1

        bc = np.zeros(m)
        for i in range(m):
            bc[i] = b[i] - np.trace(A[i,])+1

        CC =  C - np.eye(n) - sum(A[i,] for i in range(m))

        Self_alpha = 1 + np.trace(C) - np.dot(b, np.ones(m))

        Self_beta = m+ns+2+m

        MM = np.block([[  np.zeros((m,m)), np.zeros((m,m)),               -aa,         b.reshape((m,1)),        -bb.reshape((m,1)),       -np.eye(m),  np.zeros((m,m)), np.zeros((m,ns)),  np.zeros((m,1)),  np.zeros((m,1))],
                       [  np.zeros((m,m)), np.zeros((m,m)),                aa,        -b.reshape((m,1)),         bc.reshape((m,1)),  np.zeros((m,m)),       -np.eye(m), np.zeros((m,ns)),  np.zeros((m,1)),  np.zeros((m,1))],
                       [             aa.T,           -aa.T, np.zeros((ns,ns)),  svec(C).reshape((ns,1)), -svec(CC).reshape((ns,1)), np.zeros((ns,m)), np.zeros((ns,m)),      -np.eye(ns), np.zeros((ns,1)), np.zeros((ns,1))],
                       [-b.reshape((1,m)),               b,          -svec(C),          np.zeros((1,1)),                Self_alpha,      np.zeros(m),      np.zeros(m), np.zeros((1,ns)),               -1,  np.zeros((1,1))],
                       [bb.reshape((1,m)),             -bc,          svec(CC),              -Self_alpha,           np.zeros((1,1)),      np.zeros(m),      np.zeros(m), np.zeros((1,ns)),  np.zeros((1,1)),               -1]
                      ])
            
        # A
        rows = m+m+ns+1+1
        cols = m+m+n+1+1+m+m+n+1+1
        ASelf = np.zeros((rows,cols,cols))

        for i in range(m):
            ASelf[i,] = block_diag(        np.zeros((m,m)),         np.zeros((m,m)),     -(A[i,:,:]),             b[i],             -bb[i],    -matgen(m,i), np.zeros((m,m)), np.zeros((n,n)),   0,  0)
        for i in range(m,m+m):
            ASelf[i,] = block_diag(        np.zeros((m,m)),         np.zeros((m,m)),      A[i-m,:,:],          -b[i-m],            bc[i-m], np.zeros((m,m)),  -matgen(m,i-m), np.zeros((n,n)),   0,  0)
        
        j = 0
        k = 0
        for i in range(m+m,m+m+ns):
            ASelf[i,] = block_diag(        np.diag(A[:,j,k]),      -np.diag(A[:,j,k]),  np.zeros((n,n)),           C[j,k],          -CC[j,k], np.zeros((m,m)), np.zeros((m,m)), -(0.5)*eyegen(n,j,k)-(0.5)*eyegen(n,k,j),   0,  0)
            j += 1
            if j == n:
                k += 1
                j = k

        for i in range(m+m+ns,m+m+ns+1):
            ASelf[i,] = block_diag(            -np.diag(b),              np.diag(b),              -C,                0,         Self_alpha, np.zeros((m,m)), np.zeros((m,m)), np.zeros((n,n)),  -1,  0)
        for i in range(m+m+ns+1,m+m+ns+1+1):
            ASelf[i,] = block_diag(            np.diag(bb),            -np.diag(bc),              CC,      -Self_alpha,                  0, np.zeros((m,m)), np.zeros((m,m)), np.zeros((n,n)),   0, -1)

        X_init = block_diag(np.eye(m), 2*np.eye(m), np.eye(n),   1,     1, np.eye(m),   np.eye(m), np.eye(n),   1,   1)

        S_init = block_diag(np.eye(m),   np.eye(m), np.eye(n),   1,     1, np.eye(m), 2*np.eye(m), np.eye(n),   1,   1)

        y_init = np.block([np.ones(m), 2*np.ones(m), svec(np.eye(n)), 1, 1])

        Self_beta = -np.trace(np.matmul(ASelf[rows-1,],X_init))

        CSelf = np.zeros((cols,cols))
        CSelf[m+m+n+1,m+m+n+1] = Self_beta

        bSelf = np.zeros(m+m+ns+1+1)
        bSelf[-1] = -(Self_beta)

        nSelf = cols
        mSelf = rows
    else:
        raise FileNotFoundError(f"There is no such file: {instance}. Choose between PDG-Instance1-6.")

    return (nSelf, mSelf, ASelf, bSelf, CSelf, X_init, y_init, S_init)