clear;
clc;

% Data from the Python code: n = 3, m = 3
n = 3;
m = 3;

N = n + 1;    % size of SDP matrix X: 4x4
M = m + 1;    % number of equality constraints: 4

% --- Build A0, A1, A2, A3 to match the Python A[0],...,A[3] ---

A0 = zeros(N);   % corresponds to Python A[0,:,:]
A1 = zeros(N);   % corresponds to Python A[1,:,:]
A2 = zeros(N);   % corresponds to Python A[2,:,:]
A3 = zeros(N);   % corresponds to Python A[3,:,:]

% From Python output:
% A[0] =
% [[ 0., -1.,  0.,  0.],
%  [-1.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.]]
A0(1,2) = -1;
A0(2,1) = -1;

% A[1] =
% [[ 0.,  0., -1.,  0.],
%  [ 0., -1.,  0.,  0.],
%  [-1.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.]]
A1(1,3) = -1;
A1(3,1) = -1;
A1(2,2) = -1;

% A[2] =
% [[ 0.,  0.,  0., -1.],
%  [ 0.,  0.,  0.,  0.],
%  [ 0.,  0., -1.,  0.],
%  [-1.,  0.,  0.,  0.]]
A2(1,4) = -1;
A2(4,1) = -1;
A2(3,3) = -1;

% A[3] =
% [[ 0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0., -1.]]
A3(4,4) = -1;

% --- C and b (same pattern as in Python) ---

C = zeros(N);
C(1,1) = 1;        % C[0,0] = 1 in Python

b = zeros(M,1);    % length 4
b(end) = -1;       % b[-1] = -1 in Python

% --- Convert to SeDuMi format using x = vec(X) ---

nvar = N^2;        % number of scalar variables (entries of X)

% Each constraint: <A_k, X> = b_k  =>  A_k(:)' * x = b_k, with x = vec(X)
Adata = sparse(M, nvar);
Adata(1,:) = A0(:).';   % first constraint
Adata(2,:) = A1(:).';   % second constraint
Adata(3,:) = A2(:).';   % third constraint
Adata(4,:) = A3(:).';   % fourth constraint

% Objective: minimize <C, X> = vec(C)' * vec(X)
c = C(:);

% Cone: one 4x4 semidefinite block, no linear or SOC parts
K.s = N;    % [4]
K.l = 0;
K.q = [];

% Optional solver params
pars.eps = 1e-10;

% --- Call SeDuMi ---

[x, y, info] = sedumi(Adata, b, c, K, pars);

% Recover the matrix X from x: column-major reshape
Xopt = reshape(full(x), N, N);

disp('Optimal X =');
disp(Xopt);