%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name: MDL.m
%
% referenced paper:
% Wax, Mati, and Thomas Kailath. "Detection of signals by information theoretic criteria." Acoustics, Speech and Signal Processing, IEEE Transactions on 33.2 (1985): 387-392.
%
% This code was implemented by T. Yokota
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U S V mdl0] = MDL(X)

  [p N] = size(X);
  if p > N
    [N p] = size(X);
    X = X';
  end

  l = svd(X*X'/N) + 1e-8;

  for k = 1:p

    if k == p
      v = 1e-8;
    else
      v = mean(l(k+1:p));
    end    

    mdl0(k) = -sum(log(l(k+1:p))) +(p-k)*log(v) + 0.5*log(N)*k*(2*p-k)/N;

  end

  [val R] = min(mdl0);
  [U S V] = svd(X*X'/N);
  U = U(:,1:R);
  sig_r = sqrt(mean(l(R+1:end)));
  S = diag(sqrt(l(1:R) - (sig_r^2)));
  G = U*S;
  V = (( (G'*G) + (sig_r^2)*eye(R) ) \ (G'*X))';


