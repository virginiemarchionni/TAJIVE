function [M, angleBound, threshold] = TAJIVEInitExtract(datablock, vecr, iC, ...
    nresample, dataname, ierror0)
%AJIVEInitExtractMJ First step of AJIVE: Signal Space Inital Extraction.
%   Inputs:
%       datablock        - cells of data matrices {datablock 1, ..., datablock nb}
%                          Each data matrix is a d x n matrix that each row is
%                          a feature and each column is a data object
%                          Matrices are required to have same number of objects
%                          i.e. 'n'. Number of features i.e. 'd' can be different
%       vecr             - a matrix of estimated signal ranks for each data block.
%       iC               - mode wrt we Extract initial signal
%       nresample        - number of re-samples in the AJIVE step 2 for
%                          estimating the perturbation bounds; default value is 1000;
%       dataname         - a cellarray of strings: name of each data block; default
%                          name is {'datablock1', ..., 'datablocknb'}
%       ierror0          - a 1 x nb binary vector indicating whether setting 0
%                          perturbation angle for each datablock. The default is
%                          1 x nb zero vector.
%   Outputs:
%       M                - a sum(vecr) x n matrix which stacks the
%                          transpose extracted right singular matrix of each
%                          data block.
%       angleBound       - a nb x nresample matrix containing the resampled
%                          Wedin bound for the SVD perturbation angle of each
%                          data matrix.
%       threshold        - a nb x 1 vector containing the singular value
%                          threshold for noise and signal.

%    Copyright (c)  Meilei Jiang 2017
addpath 'tensor_toolbox/'


nb = length(datablock); % number of blocks

for ib=1:nb
    datablock_unfolted{ib}=tenmat(tensor(datablock{ib}),iC).data';
    %datablock{ib} = bsxfun(@minus,datablock{ib},mean(datablock{ib}));
end

% low rank svd of each data block separately
M = zeros(sum(vecr(iC,:)), size(datablock_unfolted{1}, 2)); % stack of each row space basis vectors
threshold = zeros(1, nb);
angleBound = zeros(nb, nresample);
for ib = 1:nb
    % fprintf('The input initial signal rank for mode %a %s: %d \t', iC, dataname{ib}, vecr(iC,ib))
    [u, s, v] = svds(datablock_unfolted{ib}, vecr(iC,ib) + 1);
    if vecr(iC,ib) + 1 <= size(s, 2)
        threshold(ib) = (s(vecr(iC,ib), vecr(iC,ib)) + s(vecr(iC,ib) + 1, vecr(iC,ib) + 1))/2; % threshold of singular values
    else
        threshold(ib) = s(end, end)/2;
    end
    
    U0 = u(:, 1:vecr(iC,ib));
    V0 = v(:, 1:vecr(iC,ib));
    S0 = s(1:vecr(iC,ib), 1:vecr(iC,ib));
    if ib == 1
        M(1:vecr(iC,ib), :) = V0';
    else
        M((sum(vecr(iC,1:(ib-1))) + 1):sum(vecr(iC,1:ib)), :) =  V0';
    end
    % estimating approximation accuracy
    if ierror0(ib) == 1
        angleBound(ib, :) = 0;
    else
        angleBound(ib, :) = SVDBoundWedinMJ(datablock_unfolted{ib}, vecr(iC,ib), nresample, U0, S0, V0);
        
    end
    % report the 95 percentile perturbation angle of SVD approximation
    % fprintf('SVD perturbation angle: %.2f \n', prctile(angleBound(ib,:), 95) )
end

end

