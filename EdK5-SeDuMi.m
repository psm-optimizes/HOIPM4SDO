clear;
clc;

% Sturm Example: n = 4, m = 4
n = 4;
m = 4;

N = n + 1;    % size of SDP matrix X: 5x5
M = m + 1;    % number of equality constraints: 5

% ----- Build A0, A1, A2, A3, A4 to match Python -----

A0 = zeros(N);   % A[0]
A1 = zeros(N);   % A[1]
A2 = zeros(N);   % A[2]
A3 = zeros(N);   % A[3]
A4 = zeros(N);   % A[4]

% A[0] =
% [[ 0., -1.,  0.,  0.,  0.],
%  [-1.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.]]
A0(1,2) = -1;
A0(2,1) = -1;

% A[1] =
% [[ 0.,  0., -1.,  0.,  0.],
%  [ 0., -1.,  0.,  0.,  0.],
%  [-1.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.]]
A1(1,3) = -1;
A1(3,1) = -1;
A1(2,2) = -1;

% A[2] =
% [[ 0.,  0.,  0., -1.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0., -1.,  0.,  0.],
%  [-1.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.]]
A2(1,4) = -1;
A2(4,1) = -1;
A2(3,3) = -1;

% A[3] =
% [[ 0.,  0.,  0.,  0., -1.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0., -1.,  0.],
%  [-1.,  0.,  0.,  0.,  0.]]
A3(1,5) = -1;
A3(5,1) = -1;
A3(4,4) = -1;

% A[4] =
% [[ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0.,  0.],
%  [ 0.,  0.,  0.,  0., -1.]]
A4(5,5) = -1;

% ----- C and b from Python pattern -----

C = zeros(N);
C(1,1) = 1;              % C[0,0] = 1

b = zeros(M,1);          % length 5
b(end) = -1;             % b[-1] = -1

% ----- Convert to SeDuMi format using x = vec(X) -----

nvar = N^2;              % 25 variables (entries of X)

% Each constraint: <A_k, X> = b_k  <=>  A_k(:)' * x = b_k, x = vec(X)
Adata = sparse(M, nvar);
Adata(1,:) = A0(:).';
Adata(2,:) = A1(:).';
Adata(3,:) = A2(:).';
Adata(4,:) = A3(:).';
Adata(5,:) = A4(:).';

% Objective: minimize <C, X> = vec(C)' * vec(X)
c = C(:);

% Cone: one 5x5 semidefinite block, no linear or SOC parts
K.s = N;                 % [5]
K.l = 0;
K.q = [];

% Optional solver parameters
pars.eps = 1e-10;

% ----- Call SeDuMi -----

[x, y, info] = sedumi(Adata, b, c, K, pars);

% Recover matrix X from x: column-major reshape
Xopt = reshape(full(x), N, N);

disp('Optimal X =');
disp(Xopt);