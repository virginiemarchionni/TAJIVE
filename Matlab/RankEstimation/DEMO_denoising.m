close all;

addpath('Functions_for_HOSVD_denoising');

X = double(imread('lena.pgm'));

sig   = 40;

Xn = double(uint8(X + sig*randn(size(X))));
PSNR(X,Xn)

p   = 16;
K   = 30;
step= [7 7];
rho = [0.01 0.01 0.01];
verb = 1;

[Xest RR_score] = SCORE_denoiser(Xn,p,K,step,rho,verb);
PSNR(X,Xest)

figure(1);clf;
subplot(1,2,1)
imagesc(Xn);
title('Noisy image')
subplot(1,2,2)
imagesc(Xest);
title('Denoised image')

figure(2);clf;
subplot(1,3,1)
imagesc(RR_score(:,:,1))
title('Estimated rank of R1')
colorbar
subplot(1,3,2)
imagesc(RR_score(:,:,2))
title('Estimated rank of R2')
colorbar
subplot(1,3,3)
imagesc(RR_score(:,:,3))
title('Estimated rank of R3')
colorbar

