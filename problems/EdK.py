import numpy as np

def EdK(n):
    # Problem Data
    n = n - 1 # Decrease dimension by 1 to start the prep. Output is based on size n. 
    m = n

    A = np.zeros((m+1,n+1,n+1))
    A[m,n,n] = -1
    for i in range(m):
        A[i,0,i+1]=-1
        A[i,i+1,0]=-1
        if i != 0:
            A[i,i,i]=-1
        else:
            pass

    C=np.zeros((n+1,n+1))
    C[0,0] = 1

    b=np.zeros(m+1)
    b[-1] = -1

    # Initial Solution
    if n+1 == 3:
        x11 = 1
        x13 = -1/3
        x23 = 0

        X_init = np.array([[x11,      0, x13],
                           [  0, -2*x13, x23],
                           [x13,    x23,   1]])
        y_init = np.array([0.5,1,4])
        S_init = np.array([[  1, 0.5, 1],
                           [0.5,   1, 0],
                           [  1,   0, 4]])
        
        Xopt = np.zeros((n+1,n+1))
        Xopt[n,n] = 1

        yopt = np.zeros(m+1)

        Sopt = np.zeros((n+1,n+1))
        Sopt[0,0] = 1
    
    elif n+1 == 4:
        X_init = np.array([[ 3,  0, -2, -1],
                           [ 0,  4, -1, -1],
                           [-2, -1,  2,  1],
                           [-1, -1,  1,  1]])
        y_init = np.array([1/4, 1, 2, 16])
        S_init = np.array([[        1, y_init[0], y_init[1], y_init[2]],
                           [y_init[0], y_init[1],         0,         0],
                           [y_init[1],         0, y_init[2],         0],
                           [y_init[2],         0,         0, y_init[3]]])

        Xopt = np.zeros((n+1,n+1))
        Xopt[n,n] = 1

        yopt = np.zeros(m+1)

        Sopt = np.zeros((n+1,n+1))
        Sopt[0,0] = 1

    elif n+1 == 5: 
        X_init = np.array([[64,     0,   -2, -1,    -1],
                   [ 0,     4,   -1, -1, -0.25],
                   [-2,    -1,    2,  1,  -0.5],
                   [-1,    -1,    1,  2,    -1],
                   [-1, -0.25, -0.5, -1,     1]])

        y_init = [1/8, 1/4, 1, 2, 16]

        S_init = np.array([[        1, y_init[0], y_init[1], y_init[2], y_init[3]],
                        [y_init[0], y_init[1],         0,         0,         0],
                        [y_init[1],         0, y_init[2],         0,         0],
                        [y_init[2],         0,         0, y_init[3],         0],
                        [y_init[3],         0,         0,         0, y_init[4]]])
        Xopt = np.zeros((n+1,n+1))
        Xopt[n,n] = 1

        yopt = np.zeros(m+1)

        Sopt = np.zeros((n+1,n+1))
        Sopt[0,0] = 1
        
    return (n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt)