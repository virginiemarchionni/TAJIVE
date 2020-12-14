function [Z index distance] = find_similar_patches(PP,P,K)

  p = size(P,1);

  [Ip1 Ip2] = size(PP);
  Jp1 = Ip1 - p + 1;
  Jp2 = Ip2 - p + 1;

  ZZ = zeros(p,p,Jp1*Jp2);  

  n = 1;
  for j1 = 1:Jp1
  for j2 = 1:Jp2
    
    index(n,1) = j1;
    index(n,2) = j2;
    Pn = PP(j1:j1+p-1,j2:j2+p-1);
    distance(n)= sum(sum((Pn - P).^2))/p/p;
    ZZ(:,:,n) = Pn;

    n = n + 1;

  end
  end

  [val idd] = sort(distance);

  Z = ZZ(:,:,idd(1:K));
  distance = distance(idd(1:K));
  index = index(idd(1:K),:);  

