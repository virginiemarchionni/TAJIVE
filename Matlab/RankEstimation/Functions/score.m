%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name: score.m
%
% SCORE estimates multilinear tensor rank of a tensor Z using the HOSVD core tensor and MDL(BIC).
% rho : is a N-dimensional vector that each entry is a parameter (0 < rho(n) < 1).
%
% This code was implemented by T. Yokota
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R curves Ur Sr Zest] = score(Z,rho)

  II = size(Z);
  N  = length(II);

  for n = 1:N
    Zn = unfold(Z,n);
    [U{n} d v] = svd(double(Zn*Zn'));
  end
  S = tensor_allprod(Z,U,1);
  for n = 1:3
    Sn = unfold(S,n);
    [mdl l2 rho2 v2] = calc_curve(Sn,rho(n),'bic');
    [val R(n)]= min(mdl);
    Ur{n} = U{n}(:,1:R(n));
    curves{n} = mdl;
  end
  Sr = S(1:R(1),1:R(2),1:R(3));
  Zest = tensor_allprod(Sr,Ur,0);

