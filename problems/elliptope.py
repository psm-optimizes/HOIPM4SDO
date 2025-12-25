import numpy as np

def elliptope():
    # Problem Data
    n = 2
    m = 2
    
    A=np.zeros((m+1,n+1,n+1))
    for i in range(n+1):
        A[i,i,i]=1
    
    C=np.array([[0,2,-2],[2,0,-1],[-2,-1,0]])
    
    b=np.ones(n+1)

    # Initial Solution
    X_init = np.array([[ 1.        , -0.63590928,  0.63590928],
                       [-0.63590928,  1.        , -0.12671588],
                       [ 0.63590928, -0.12671588,  1.        ]])
    y_init = np.array([-3.54363714, -2.14510269, -2.14510269])
    S_init = np.array([[ 3.54363714,  2.        , -2.        ],
                       [ 2.        ,  2.14510269, -1.        ],
                       [-2.        , -1.        ,  2.14510269]])

    # Optimal Solution (unique)
    Xopt = np.array([[1, -1, 1],
                     [-1, 1,-1],
                     [1, -1, 1]])
    yopt = np.array([-4, -1, -1])
    Sopt = np.array([[4, 2,-2],
                     [2, 1,-1],
                     [-2,-1,1]])

    return (n, m, A, b, C, X_init, y_init, S_init, Xopt, yopt, Sopt)