# KSTTM
Codes for Kernelized Support Tensor Train Machines

Kernelized Support Tensor Train Machines (KSTTM) [1] for Matlab&copy;/Octave&copy;
--------------------------------------------------------------------------------------------------

This package contains Matlab/Octave code for the methods mentioned in Kernelized Support Tensor Train Machines, namely K-STTM-Prod and K-STTM-Sum.

1. Requirements
------------

* Matlab or Octave.

* CVX toolbox [3].

2. Functions
------------

* fmri_demo

Demonstrates the usage of the KSTTM algorithm. 

* [K] = kernel_mat(X, N, d, sigma, weight, flag)

Kernel matrix construction for K-STTM-Prod and K-STTM-Prod given the training TT-format data, number of training smaples, the order of tensor data, gaussian kernel parameter sigma, weight on the first and second modes of the tensor data, and the flag.

* [Ypred] = predict(XX, alpha, b, X, Y, sigma, d, weight, flag)

Label prediction function for K-STTM-Prod and K-STTM-Prod given the TT-format data for prediction ,the learned Lagrange multipliers, the learnedbias in SVM solver, the training TT-format data, the labels of training data, gaussian kernel parameter sigma, the order of tensor data, weight on the first and second modes of the tensor data, and the flag.

* [alpha, b] = svm_solver(K, Y, C, N)

SVM solver function for K-STTM-Prod and K-STTM-Prod given the learned kernel matrix, the labels of training data, the performance tradeoff parameter, the number of training data. This function Solves the dual quadratic program of the L1-regularized SVM problem with CVX toolbox [3].

* [U, D, V, post] = VBMF(Y, cacb, sigma2, H)

Perform variational Bayesian matrix factorization [2] for FULLY OBSERVED matrix, Gaussian noise and isotropic Gaussian prior. We use this function to replace the SVD operation in TT-SVD.


3. Reference
------------
[1]. "Kernelized Support Tensor Train Machines"

Authors: Cong Chen, Kim Batselier, Wenjian Yu, Ngai Wong


[2]. "Condition for Perfect Dimensionality Recovery by Variational Bayesian PCA"

Authors: S. Nakajima, R. Tomioka, M. Sugiyama, S. D. Babacan

[3]. http://cvxr.com/cvx/
