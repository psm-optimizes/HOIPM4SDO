import numpy as np
from utilities import svec, symet, smat, symkron


feas_tol = 1e-10
def init_prep(n, m, A,b,C, X_init, y_init, S_init):
    nn = n+1
    mm = m+1

    n2           = nn*nn
    ns           = int(0.5*nn*(nn+1))

    AA = np.zeros((mm,ns))
    for i in range(mm):
        t = svec(A[i,:,:])
        for j in range(len(t)):
            AA[i,j]=t[j]

    X1 = X_init
    y1 = y_init
    S1 = S_init

    X2 = None
    y2 = None
    S2 = None

    while np.linalg.norm( ((np.matmul(X1,S1) + np.matmul(S1,X1))/(2*(np.trace(np.matmul(X1,S1))/(nn)))) - np.eye(nn), 'fro') > 1.0e-15:

        P  = np.eye(nn) # AHO
        EE = symkron(P,np.matmul(np.linalg.inv(P).T,S1))
        FF = symkron(np.matmul(P,X1),np.linalg.inv(P).T)

        M  = np.block([[np.zeros((ns,ns)), AA.T,             np.eye(ns)      ],
                       [AA,                np.zeros((mm,mm)),  np.zeros((mm,ns))],
                       [EE,                np.zeros((ns,mm)), FF              ]])

        XS = symet(np.matmul(X1,S1), "AHO")
        RC = svec((1)*np.eye(nn) - XS)

        r  = np.block([np.zeros(ns+mm),RC])
        #-------------------------------------------------------------------
        # Linear system solver solution
        #-------------------------------------------------------------------
        z  = np.linalg.solve(M, r)
        #-------------------------------------------------------------------
        # Calculate delta_x and delta_s
        #-------------------------------------------------------------------
        vdx     = z[:ns]
        vdy     = z[ns:ns+mm]
        vds     = z[ns+mm:]

        delta_x = smat(vdx)
        delta_y = vdy
        delta_s = smat(vds)
        #-------------------------------------------------------------------
        # Update
        #-------------------------------------------------------------------
        S2 = S1 + delta_s
        X2 = X1 + delta_x
        y2 = y1 + delta_y

        X1 = X2
        y1 = y2
        S1 = S2

    # Primal feasibility check
    for i in range(mm):
        if np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X1))) >= feas_tol:
            print('Primal Infeasibility with error of', np.linalg.norm(b[i] - np.trace(np.matmul(A[i,],X1))))
    
    # Dual feasibility check
    c_temp = np.zeros((nn,nn))
    for i in range(mm):
        c_temp = c_temp + y1[i]*A[i,]
    if np.linalg.norm(S1 - (C - c_temp)) >= feas_tol:
        print('Dual Infeasibility with error of', np.linalg.norm(S1 - (C - c_temp)))
    
    if X1 is not None and X2.any():
        return X1, y1, S1
    else:
        return X_init, y_init, S_init