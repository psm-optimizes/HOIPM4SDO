clear;
clc;

n = 2;
m = 2;

N = n + 1;   % size of SDP matrix X: 3x3
M = m + 1;   % number of constraints: 3

A0 = zeros(N);   % corresponds to Python A[0,:,:]
A1 = zeros(N);   % corresponds to Python A[1,:,:]
A2 = zeros(N);   % corresponds to Python A[2,:,:]

% Python: A[0,0,1] = -1, A[0,1,0] = -1
A0(1,2) = -1;
A0(2,1) = -1;

% Python: A[1,0,2] = -1, A[1,2,0] = -1, and A[1,1,1] = -1
A1(1,3) = -1;
A1(3,1) = -1;
A1(2,2) = -1;

% Python: A[2,2,2] = -1
A2(3,3) = -1;

% C[0,0] = 1
C = zeros(N);
C(1,1) = 1;

% b = [0, 0, -1]^T
b = [0; 0; -1];

% Proble Setup
nvar = N^2;   % = 9

% Build A (M x nvar), each row corresponds to <A_k, X> = b_k
A = sparse(M, nvar);
A(1,:) = A0(:)';   % first constraint: <A0, X> = b(1)
A(2,:) = A1(:)';   % second constraint: <A1, X> = b(2)
A(3,:) = A2(:)';   % third constraint: <A2, X> = b(3)

% Objective: minimize <C, X> = vec(C)' * vec(X)
c = C(:);

% Cone: one 3x3 semidefinite block, no linear or second-order parts
K.s = N;    % [3]
K.l = 0;
K.q = [];

pars.eps = 1e-10;

[x, y, info] = sedumi(A, b, c, K, pars);

% Recover the matrix X from x (SeDuMi uses column-major order)
Xopt = reshape(x, N, N);

disp('Optimal X =');
disp(Xopt);