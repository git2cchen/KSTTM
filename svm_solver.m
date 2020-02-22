function [ alpha, b] = svm_solver( K, Y, C, N)
%  [ alpha, b] = svm_solver( K, Y, C, N)
% -------------
% svm solver function for K-STTM-Prod and K-STTM-Prod given the learned 
% kernel matrix, the labels of training data, the performance tradeoff
% parameter, the number of training data. This function Solves the dual 
% quadratic program of the L1-regularized SVM problem with CVX toolbox.
%
% K         =   the learned kernel matrix,
%
% Y         =   the labels of training data,
%
% C         =   the performance tradeoff parameter,
%
% N         =   the number of training data,
%
% alpha     =   the learned Lagrange multipliers,
%
% b         =   the learned bias in SVM solver.
%
% Reference
% ---------
%
% Kernelized Support Tensor Train Machines

% 20/02/2020, Cong CHEN

m=N;
one = ones(m,1);
alpha = zeros(m,1);
cvx_begin
%         variables alpha(n)
    variables alpha(m)
    maximize(-1/2 * (alpha.*Y)'*K*(alpha.*Y) + one'*alpha)
    subject to
        0 <= alpha <= C
        alpha'*Y == 0
cvx_end

% To calculate b we pick an arbitrary support vector
ind = find(alpha>=C*1e-10);%original 1e-5
ind = ind(1);
b = Y(ind) - alpha'*(Y.*K(:,ind));
end