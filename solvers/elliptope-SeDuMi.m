clear;
clc;

%%% MAXCUT Example
n = 3;
A = sparse(1:n,1:n+1:n^2, ones(1,n),n,n^2);
At = full(A);
b = ones(n,1);
c = [0 2 -2 2 0 -1 -2 -1 0]';
K.s = [n];
K.q = [];
K.l = [];
pars.eps    = 1e-10;

[X,Y,INFO] = sedumi(At,b,c,K, pars);
% S = c - Y'*A;

disp('Optimal X =');
disp(mat(X));