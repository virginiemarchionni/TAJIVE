function randSSVs = RandDirSSVMJ(lengthDim, dimsSubsp, nresample)
% RandDirSSVMJ Empirical distribution of largest squared singular value
% of M generating from of random subspaces with given ranks.
%
%   Inputs:
%       lengthDim - dimension of the whole space
%       vecr - a vector of dimensions of each subspaces
%       nresample - number of simulated samples
%   Outputs:
%       randSSVs - largest squared singular values in random M.
%
%   Copyright (c) Meilei Jiang 2017

if ~exist('nresample', 'var')
    nresample = 1000;
end

nb = length(dimsSubsp);
M = zeros(sum(dimsSubsp), lengthDim);

randSSVs = zeros(1, nresample);

for i = 1:nresample
    for ib = 1:nb
        irow_start = sum(dimsSubsp(1 : (ib - 1))) + 1;
        irow_end = sum(dimsSubsp(1 : ib));
        M(irow_start : irow_end, :) =  orth(randn(lengthDim, dimsSubsp(ib)))';
    end
    randSSVs(i) = norm(M, 2)^2;
end

end

