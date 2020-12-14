function outstruct = TAJIVEReconstruct(datablock, iC, threshold, dataname, row_joint, ioutput, ferror)
%  function for AJIVE Matrix re-construction
% Inputs:
%   datablock        - cells of data matrices {datablock 1, ..., datablock nb}
%                    - Each data matrix is a d x n matrix that each row is
%   dataname         - a cell of strings: name of each data block; default
%                    - name is {'datablock1', ..., 'datablocknb'}
%   row_joint        - orthonormal basis of estimated joint row space
%                      output of 'AJIVEJointSelectMJ.m'
%   ioutput          - 0-1 indicator vector of output's structure
%                      [CommonSignalScore,
%                       CommonSignalScoreloading,
%                       BlockSpecificJoint,
%                       BlockSpecificJointLoading,
%                       BlockSpecificIndiv,
%                       BlockSpecificIndivLoading,
%                       MatrixJoint,
%                       MatrixIndiv,
%                       MatrixResid
%                       ]
%    ferror            a small number add to singular value to avoid float
%                      error. The default value is 10^(-10).
%
% Outputs:
%   outstruct        - a structure contains all elements in ioutput
%
%    Copyright (c) Meilei Jiang, Qing Feng, Jan Hannig & J. S. Marron 2017
if ~exist('ferror', 'var')
    ferror = 10^-10;
end

nb = length(datablock);
n = size(datablock{1}, iC);
% check and re-adjust the joint rank

DeleteJointRows = []; % Delete the rows in the row_joint, which have low
% variance projected on some datablocks.
for ib = 1:nb
    db=tenmat(tensor(datablock{ib}),iC).data';
    JointDirection_unfolded = db * row_joint';

%     JointDirection = ttm(tensor(datablock{ib}),row_joint,iC).data;
%     JointDirection_unfolded=tenmat(JointDirection,iC).data;
%     
    % Rows in the joint space basis have sd lower than threshold.
    LowSdJointRow = find(sqrt(sum(JointDirection_unfolded.^2, 1)) <= threshold(ib) + ferror );
    
    if size(LowSdJointRow) > 0
        if size(LowSdJointRow) == 1
            disp(['Note: The ' num2order(LowSdJointRow) ' joint space basis vector has low variance in ' dataname{ib} '.'])
        else
            disp(join(['Note: The ' join(cellstr(num2order(LowSdJointRow)), ', ') ' joint space basis vectors have low variance in ' dataname{ib} '.'],''))
        end
        DeleteJointRows = union(DeleteJointRows, LowSdJointRow);
    end
end

if size(DeleteJointRows) >  0
    if size(DeleteJointRows) == 1
        disp(['Note: The ' num2order(DeleteJointRows) ' joint space basis vector will be dropped.'])
    else
        disp(join(['Note: The ' join(cellstr(num2order(DeleteJointRows)), ', ') ' joint space basis vectors will be dropped.'],''))
    end
end

row_joint( DeleteJointRows, :) = [];
rjoint = size(row_joint, 1);
disp(['Final Joint rank: ' num2str(rjoint)]);

% Reconstruction the AJIVE decomposition in each datablock

CommonSignalScore = row_joint;

CommonSignalScoreloading = cell(1,nb);
BlockSpecificJoint = cell(1,nb);
BlockSpecificJointLoading = cell(1,nb);
BlockSpecificJointAutoval=cell(1,nb);
BlockSpecificIndiv = cell(1,nb);
BlockSpecificIndivLoading = cell(1,nb);
BlockSpecificIndivAutoval=cell(1,nb);

MatrixJoint = cell(1,nb);
MatrixIndiv = cell(1,nb);
MatrixResid = cell(1,nb);

rankI = zeros(1, nb);

for ib = 1:nb
    db=tenmat(tensor(datablock{ib}),iC).data';

    % Joint reconstruction
     % Joint Block in each data block
    MatrixJoint{ib} = db * row_joint' * row_joint;
    
    % Block Specitic Scores
    [t1,t2,t3] = svds(MatrixJoint{ib},rjoint);
    BlockSpecificJointLoading{ib} = t1;
    BlockSpecificJoint{ib} = t2*t3';
    BlockSpecificJointAutoval{ib}=diag(t2);
    
    %  Individual reconstruction
    % orthogonal basis od null space of joint
    
    indiv = db - db * row_joint' * row_joint;
    s_indiv = svd(indiv);
    
    rI = length(find(s_indiv + ferror >threshold(ib)));
    [i1,i2,i3] = svds(indiv,rI);
    MatrixIndiv{ib} = i1*i2*i3';
    BlockSpecificIndivLoading{ib} = i1;
    BlockSpecificIndiv{ib} = i2*i3';
    BlockSpecificIndivAutoval{ib}=diag(i2);

    rankI(ib) = rI;
    disp(['Final individual ' dataname{ib} ' rank: ' num2str(rI)]);
    
    % Residual reconstruction
    
    MatrixResid{ib} = db - MatrixJoint{ib} - MatrixIndiv{ib};
end



% return needed results based on ioutput
outstruct.rankJoint=rjoint;
outstruct.rankIndividual=rankI;

if ioutput(1) == 1 % output common normalized score
    outstruct.CommonSignalScore = CommonSignalScore';
else
    outstruct.CommonSignalScore = [];
end

if ioutput(2) == 1 % output projection loadings of common normalized score
    outstruct.CommonSignalScoreloading = CommonSignalScoreloading';
else
    outstruct.CommonSignalScoreloading = {};
end

if ioutput(3) == 1 % output block specific scores of each joint
    outstruct.BlockSpecificJoint = BlockSpecificJoint';
else
    outstruct.BlockSpecificJoint = {};
end

if ioutput(4) == 1 % output the loading matrix of each joint block specific score
    outstruct.BlockSpecificJointLoading = BlockSpecificJointLoading';
    outstruct.BlockSpecificJointAutoval = BlockSpecificJointAutoval;
    else
    outstruct.BlockSpecificJointLoading = {};
    outstruct.BlockSpecificJointAutoval = {};
end


if ioutput(5) == 1 % output block specific scores of each individual
    outstruct.BlockSpecificIndiv = BlockSpecificIndiv';
else
    outstruct.individualBlockSpecific = {};
end

if ioutput(6) == 1 % output the loading matrix of each individual block specific score
    outstruct.BlockSpecificIndivLoading = BlockSpecificIndivLoading';
    outstruct.BlockSpecificIndivAutoval = BlockSpecificIndivAutoval;
else
    outstruct.BlockSpecificIndivLoading = {};
    outstruct.BlockSpecificIndivAutoval = {};
end

if ioutput(7) == 1 % output joint matrices
    outstruct.MatrixJoint = MatrixJoint';
else
    outstruct.MatrixJoint = {};
end

if ioutput(8) == 1 % output individual matrices
    outstruct.MatrixIndiv = MatrixIndiv';
else
    outstruct.MatrixIndiv = {};
end

if ioutput(9) == 1 % output residual matrices
    outstruct.MatrixResid = MatrixResid';
else
    outstruct.MatrixResid = {};
end
end

