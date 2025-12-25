import numpy as np
def Hauen_ssc():    
    # Problem Data
    n = 1
    m = 0

    A = np.zeros((m+1,n+1,n+1))
    A[0,0,0] = 2
    A[0,1,1] = 3

    C=np.array([[2,1], [1,0]])
    b=np.zeros(m+1)
    b[0] = 1

    X_init = np.array([[0.1, 0.0],
                       [0.0, (1-2*0.1)/3]], dtype=float)

    y_init = -0.2
    S_init = C - y_init*A[0]

    return (n, m, A, b, C, X_init, y_init, S_init)