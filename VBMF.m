function [U, D, V, post] = VBMF(Y, cacb, sigma2, H)
% Variationl Bayesian Matrix Factorization for fully observed matrix
%
% Overview:
%   Perform variational Bayesian matrix factorization for FULLY OBSERVED matrix, 
%   Gaussian noise and isotropic Gaussian prior.
%   See the reference below for more detail.
%
% Usage:
%   d = VBMF(Y)
%   d = VBMF(Y, cacb, sigma2, H)
%   [U, D, V, data] = VBMF(Y, cacb, sigma2, H)
%
% Input:
%    Y         :Observed matrix.
%    cacb      :(OPTIONAL) Product of prior variance.  Estimated if omitted or empty.
%    sigma2    :(OPTIONAL) Noise varaince.  Estimated if omitted or empty.
%    H         :(OPTIONAL) Rank.  Set to the be full rank if omitted or empty. 
%
% Output:
%    d         :Estimated NON-ZERO singular values.
%    [U, D, V] :Resulting low rank matrix given in the SVD form.
%    post :Information for the complete VB posterior reconstruction.
%
% Reference:
%   Shinichi Nakajima, Ryota Tomioka, Masashi Sugiyama, S. Derin Babacan,
%   "Condition for Perfect Dimensionality Recovery by Variational Bayesian PCA," JMLR2015,
%   http://sites.google.com/site/shinnkj23/home/manuscript_jmlr2015.pdf .
% 
% Copyright(c) 2015 Shinichi Nakajima, TU Berlin, Germany.
% http://sites.google.com/site/shinnkj23/     
% This software is distributed under the MIT license. See license.txt.

  if nargin < 2 cacb = {}; end
  if nargin < 3 sigma2 = {}; end
  if nargin < 4 H = {}; end

  computepost = 0;
  if nargout > 3
    computepost = 1;
  end

  [L, M] = size(Y);
  if L > M
    if computepost
      [U, D, V, post] = VBMF(Y', cacb, sigma2, H);    
      temp = post.ma;  post.ma = post.mb;  post.mb = temp;
      temp = post.sa2;  post.sa2 = post.sb2;  post.sb2 = temp;
    else
      [U, D, V] = VBMF(Y', cacb, sigma2, H);
    end
    temp = U;  U = V;  D = D'; V = temp;
    return
  end

  alpha = L / M;
  tauubar = 2.5129 * sqrt(alpha);

  if isempty(H)
    H = L;
  end

  [mlU, mlD, mlV] = fast_svds(Y, H);
  mld = diag(mlD)';
  residual = 0;
  if H < L
    residual = sum(Y(:).^2 - mld.^2);
  end

  if isempty(sigma2) %%% Noise variance estimation
    sigma2 = estimatesigma2(mld, residual, L, M, cacb, tauubar);
  end

  if ~isempty(cacb) %%% VB
    threshold = sqrt(sigma2 * ((L + M) / 2 + sigma2 / (2 * cacb^2) + sqrt(((L + M) / 2 + sigma2 / (2*cacb^2))^2 - L * M)));

    pos = max(find(mld > threshold));
    d = mld(1:pos) .* (1 - sigma2 ./ (2 * mld(1:pos).^2) .* (L + M + sqrt((M-L)^2 + 4 * mld(1:pos).^2 / cacb^2)));

    if computepost       
      zeta = sigma2 / (2 * L * M) * (L + M + sigma2 / cacb^2 - sqrt((L + M + sigma2 / cacb^2)^2 - 4 * L * M));
      post.ma = zeros(size(mld));  post.sa2 = cacb * (1 - L * zeta / sigma2) * ones(size(mld));
      post.mb = zeros(size(mld));  post.sb2 = cacb * (1 - M * zeta / sigma2) * ones(size(mld));

      delta = cacb / sigma2 * (mld(1:pos) - d - L * sigma2 ./ mld(1:pos));
      post.ma(1:pos) = sqrt(d .* delta);  post.sa2(1:pos) = sigma2 * delta ./ mld(1:pos);     post.cacb(1:pos) = cacb;
      post.mb(1:pos) = sqrt(d ./ delta);  post.sb2(1:pos) = sigma2 ./ (delta .* mld(1:pos));  post.sigma2 = sigma2;
      post.F = 0.5 * (L * M * log(2 * pi * sigma2) + (residual + sum(mld.^2)) / sigma2 - (L + M) * H ...
                 + sum(M * log(cacb ./ post.sa2) + L * log(cacb ./ post.sb2) ...
                      + (post.ma.^2 + M * post.sa2) / cacb + (post.mb.^2 + L * post.sb2) / cacb ...
                      + (-2 * post.ma .* post.mb .* mld + (post.ma.^2 + M * post.sa2) .* (post.mb.^2 + L * post.sb2)) / sigma2));
    end                             
  else %%% EVB 
    threshold = sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar));

    pos = max(find(mld > threshold));
    d = mld(1:pos) / 2 .* (1 - (L + M) * sigma2 ./ mld(1:pos).^2 ...
                             + sqrt((1 - (L + M) * sigma2 ./ mld(1:pos).^2).^2 - 4 * L * M * sigma2^2 ./ mld(1:pos).^4));
    if computepost       
      post.ma = zeros(size(mld));  post.sa2 = eps * ones(size(mld));  post.cacb = eps * ones(size(mld));
      post.mb = zeros(size(mld));  post.sb2 = eps * ones(size(mld));

      tau = d .* mld(1:pos) / (M * sigma2);
      delta = sqrt(M * d ./ (L * mld(1:pos))) .* (1 + alpha ./ tau);
      post.ma(1:pos) = sqrt(d .* delta);  post.sa2(1:pos) = sigma2 * delta ./ mld(1:pos);     post.cacb(1:pos) = sqrt(d .* mld(1:pos) / (L * M));
      post.mb(1:pos) = sqrt(d ./ delta);  post.sb2(1:pos) = sigma2 ./ (delta .* mld(1:pos));  post.sigma2 = sigma2;
      post.F = 0.5 * (L * M * log(2 * pi * sigma2) + (residual + sum(mld.^2)) / sigma2 ...
                                     + sum(M * log(tau + 1) + L * log(tau / alpha + 1) - M * tau));
    end                           
  end

  if nargout == 1
    U = d;
  else
    U = mlU(:, 1:length(d));
    D = diag(d);  
    V = mlV(:, 1:length(d));
  end
end

%%% Subroutines %%%

function [U, S, V] = fast_svds(X,H)
  [U, sqS, ~] = svds(X*X',H);
  S = diag(sqrt(diag(sqS)));
  V = (X'*U) ./ repmat(diag(S)',size(X,2),1);
end

function sigma2 = estimatesigma2(mld, residual, L, M, cacb, tauubar)
  H = length(mld);
  alpha = L / M;
  if ~isempty(cacb)
    sigma2_ub = (sum(mld.^2) + residual) / (L + M);
    if L == H
      sigma2_lb = mld(end)^2 / M + eps;
    else
      sigma2_lb = residual / ((L - H) * M) + eps;
    end
    [sigma2, obj, flag] = fminbnd(@(sigma2)objVB(mld, residual, L, M, cacb, sigma2), sigma2_lb, sigma2_ub);
  else  
    xubar = (1 + tauubar) * (1 + alpha / tauubar);
    eH_ub = min(ceil(L / (1 + alpha)) - 1, H);
    sigma2_ub = (sum(mld.^2) + residual) / (L * M);
    sigma2_lb = max([mld(eH_ub + 1)^2 / (M * xubar), mean(mld(eH_ub+1:end).^2) / M]);
    
    scale = 1 / sigma2_lb;
    %scale = 100 / sigma2_lb;
    %scale = sqrt(1 / sigma2_lb);
    %scale = 1;
    mld = mld * sqrt(scale);
    residual = residual * scale;
    sigma2_ub = sigma2_ub * scale;
    sigma2_lb = sigma2_lb * scale;

    [sigma2, obj, flag] = fminbnd(@(sigma2)objEVB(mld, residual, L, M, sigma2, xubar), sigma2_lb, sigma2_ub);
%    fprintf('sigma2 %f, obj%f\n',sigma2/scale,obj);
%    [sigma2, obj, flag] = fminbnd(@(sigma2)objEVB(mld, residual, L, M, sigma2, xubar), sigma2*0.8, sigma2*1.2);
%    fprintf('sigma2 %f, obj%f\n',sigma2/scale,obj);
    if 0
      fprintf('ub%g, lb%g, est%g, estscaled%g\n', sigma2_ub, sigma2_lb, sigma2, sigma2 / scale);
    end
    if 0
%      tempsigma2 = sigma2_lb:(sigma2_ub-sigma2_lb) / 100: sigma2_ub;
      %tempsigma2 = sigma2_lb/3:(sigma2*3-sigma2_lb/3) / 100: sigma2 * 3;
      tempsigma2 = sigma2_lb/3:(sigma2*3-sigma2_lb/3) / 1000: sigma2 * 3;
      for i = 1: length(tempsigma2)
        sss(i) = objEVB(mld, residual, L, M, tempsigma2(i), xubar);
      end
      figure; hold on; plot(tempsigma2, sss,'.-');
      plot(sigma2_lb,objEVB(mld, residual, L, M, sigma2_lb, xubar),'r*');
      plot(sigma2,objEVB(mld, residual, L, M, sigma2, xubar),'r*');
      %plot(sigma2_ub,objEVB(mld, residual, L, M, sigma2_ub, xubar),'r*');
      %keyboard
    end   
            
    sigma2 = sigma2 / scale;
  end  
end

function [obj] = objVB(mld, residual, L, M, cacb, sigma2) %%% Objective for sigma2 estimation in VB
  H = length(mld);
  threshold = sqrt(sigma2 * ((L+M)/2 + sigma2 / (2 * cacb^2) + sqrt(((L + M) / 2 + sigma2 / (2*cacb^2))^2 - L*M)));
  pos = max(find(mld > threshold));
  d = mld(1:pos) .* (1 - sigma2 ./ (2 * mld(1:pos).^2) .* (L + M + sqrt((M-L)^2 + 4 * mld(1:pos).^2 / cacb^2)));

  zeta = sigma2 / (2 * L * M) * (L + M + sigma2 / cacb^2 - sqrt((L + M + sigma2 / cacb^2)^2 - 4 * L * M));
  post.ma = zeros(size(mld));  post.sa2 = cacb * (1 - L * zeta / sigma2) * ones(size(mld));
  post.mb = zeros(size(mld));  post.sb2 = cacb * (1 - M * zeta / sigma2) * ones(size(mld));

  delta = cacb / sigma2 * (mld(1:pos) - d - L * sigma2 ./ mld(1:pos));
  post.ma(1:pos) = sqrt(d .* delta);  post.sa2(1:pos) = sigma2 * delta ./ mld(1:pos);     post.cacb(1:pos) = cacb;
  post.mb(1:pos) = sqrt(d ./ delta);  post.sb2(1:pos) = sigma2 ./ (delta .* mld(1:pos));  post.sigma2 = sigma2;
  obj = 0.5 * (L * M * log(2 * pi * sigma2) + (residual + sum(mld.^2)) / sigma2 - (L + M) * H ...
             + sum(M * log(cacb ./ post.sa2) + L * log(cacb ./ post.sb2) ...
                  + (post.ma.^2 + M * post.sa2) / cacb + (post.mb.^2 + L * post.sb2) / cacb ...
                  + (-2 * post.ma .* post.mb .* mld + (post.ma.^2 + M * post.sa2) .* (post.mb.^2 + L * post.sb2)) / sigma2));
end

function obj = objEVB(mld, residual, L, M, sigma2, xubar) %%% Objective for sigma2 estimation in EVB
  H = length(mld);
  alpha = L / M;
  x = mld.^2 / (M * sigma2);
  
  obj = sum(phi0(x) + double(x > xubar) .* phi1(x, alpha)) + residual / (M * sigma2) + (L - H) * log(sigma2);  
  
  obj_backup = obj;
  z1 = x(x>xubar);
  z2 = x(x<=xubar);
  %obj = sum(phi0(x)) + sum( phi1(z1,alpha) ) + residual / (M * sigma2) + (L - H) * log(sigma2);
  tau_z1 = tau(z1,alpha);      
  term1 = sum(z2 - log(z2));                  
  term2 = sum(z1 - tau_z1); % Very Important!!!!!
  %term2 = sum(z1) - sum(tau_z1);
  term3 = sum( log( (tau_z1+1)./z1 ) );
  %term3 = sum(log(tau_z1+1) - log(z1));
  term4 = alpha*sum(log(tau_z1/alpha+1));
  obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * log(sigma2);  
  if 0
      fprintf('sigma2= %f: %f, %f, diff %f\n',sigma2, obj_backup, obj, obj_backup - obj);
  end
end

function y = phi0(x)
  y = x - log(x);
end

function y = phi1(x, alpha)
  y = log(tau(x, alpha) + 1) + alpha * log(tau(x, alpha) / alpha + 1) - tau(x, alpha);
end

function y = tau(x, alpha)
  y = 0.5 * (x - (1 + alpha) + sqrt((x - (1 + alpha)).^2 - 4 * alpha));
end
