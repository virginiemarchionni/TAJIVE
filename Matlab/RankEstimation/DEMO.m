
addpath('Functions')

II = [100 100 100];
R  = [20 30 40];

for n = 1:length(II)
  U{n} = randn(II(n),R(n));
end
G = randn(R);
Z = tensor_allprod(G,U,0); % low-rank tensor

Zn= Z + 400*randn(II); % noisy full-rank tensor

snr = 10*log10( (Z(:)'*Z(:)) / ((Z(:)-Zn(:))'*(Z(:)-Zn(:))) );

% try MDL
for n = 1:3
  [ u s v curves_mdl{n}] = MDL(unfold(Zn,n));
  [ val Rmdl(n) ] = min(curves_mdl{n});
end

% try SCORE
rho= [0.001 0.001 0.001];
[Rest curves] = score(Zn,rho);

fprintf('-------------------------------------\n')
fprintf('SNR = %f \n',snr);
fprintf('Original  rank     [%d, %d, %d] \n',R(1),R(2),R(3));
fprintf('Est. rank by MDL   [%d, %d, %d] \n',Rmdl(1),Rmdl(2),Rmdl(3));
fprintf('Est. rank by SCORE [%d, %d, %d] \n',Rest(1),Rest(2),Rest(3));
fprintf('-------------------------------------\n')

figure(1);clf;
subplot(1,3,1);hold on;
h(1) = plot(curves_mdl{1},'r-','linewidth',2);
h(2) = plot(curves{1},'b--','linewidth',2);
legend(h,'MDL','SCORE')

subplot(1,3,2);hold on;
h(1) = plot(curves_mdl{2},'r-','linewidth',2);
h(2) = plot(curves{2},'b--','linewidth',2);
legend(h,'MDL','SCORE')

subplot(1,3,3);hold on;
h(1) = plot(curves_mdl{3},'r-','linewidth',2);
h(2) = plot(curves{3},'b--','linewidth',2);
legend(h,'MDL','SCORE')


