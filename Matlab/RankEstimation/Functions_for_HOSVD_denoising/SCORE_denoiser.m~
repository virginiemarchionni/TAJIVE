%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name: SCORE_denoiser.m
%
% Denoising gray-scale image by using patch-matching
% and low-rank higher order singular value decomposition
% (HOSVD) for each patch group. 
%
% Xn  : noisy gray scale-image (a matrix)
% p   : patch-size, a patch becomes a (p,p)-matrix.
% K   : number of patches in each group, a group becomes a (p,p,K)-tensor.
% sk  : step-size of patch-based processing.
% rho : is a three-dimensional vector that each entry is a parameter (0 < rho(n) < 1).
% verb: to monitor mid-flow of denoising procedure if verb=1.
%
% This code was implemented by T. Yokota
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X RR] = SCORE_denoiser(Xn,p,K,sk,rho,verb)

  warning off; 

  SR = 20;

  min_value = min(Xn(:));
  max_value = max(Xn(:));

  [I1 I2] = size(Xn);
  Xd = zeros(I1,I2);
  Np = zeros(I1,I2);
  one= ones(p,p);

  J1 = I1 - p + 1;
  J2 = I2 - p + 1;

  if verb
    figure();clf;
    imagesc(uint8(Xn));
    colormap(gray);
    fig = figure();clf;
    imagesc(Xd);
    colormap(gray);
    drawnow;
  end

  count = 1;
  JJ1 = 1:sk(1):J1;
  JJ2 = 1:sk(2):J2;
  for ii = 1:length(JJ1)
  for jj = 1:length(JJ2)

    j1 = JJ1(ii);
    j2 = JJ2(jj);

    P  = Xn(j1:j1+p-1,j2:j2+p-1);  %% a reference patch

    h1 = max(1,j1-SR);
    h2 = min(I1,j1+SR);
    w1 = max(1,j2-SR);
    w2 = min(I2,j2+SR);

    PP = Xn(h1:h2,w1:w2);  %% region for search

    [Z index distance] = find_similar_patches_mex(PP,P,K); %% set of similar patches (3d-tensor);
    %[Z index distance] = find_similar_patches(PP,P,K); %% if you cannot compile mexfile please use this
    index(:,1) = index(:,1) + h1 - 1;
    index(:,2) = index(:,2) + w1 - 1;

    %% rank estimation by SCORE
    [R curves] = score(Z,rho);

    %% HOSVD and reconstruct low-rank (truncated) HOSVD
    for n = 1:3
      Zn = unfold(Z,n);
      [U{n} d v] = svd(double(Zn*Zn'));
      Ur{n} = U{n}(:,1:R(n));
    end
    S = tensor_allprod(Z,U,1);
    Sr = S(1:R(1),1:R(2),1:R(3));
    Zest = tensor_allprod(Sr,Ur,0);
    RR(ii,jj,1) = R(1);
    RR(ii,jj,2) = R(2);
    RR(ii,jj,3) = R(3);

    %% return denoised patches
    for k = 1:K
      h = index(k,1);
      w = index(k,2);
      if k == 1
        h = j1;
        w = j2;
      end
      Xd(h:h+p-1,w:w+p-1) = Xd(h:h+p-1,w:w+p-1) + Zest(:,:,k);
      Np(h:h+p-1,w:w+p-1) = Np(h:h+p-1,w:w+p-1) + 1;
    end

    %% show progress
    if mod(count,100) == 0;
      fprintf('%d/%d = %f %%, [%d %d %d] \n',count,length(JJ1)*length(JJ2),count/length(JJ1)/length(JJ2)*100,R(1),R(2),R(3));
    end
    count = count + 1;

    %% monitor result
    if verb == 1
      set(0,'CurrentFigure',fig);
      imagesc(uint8(Xd./Np));
      caxis([min_value max_value]);
      drawnow;
    end

  %end

    if verb == 2 & mod(ii,J1) == 0
      set(0,'CurrentFigure',fig);
      imagesc(uint8(Xd./Np));
      caxis([min_value max_value]);
      drawnow;
    end

  end
  end

  X = uint8(Xd./Np);


