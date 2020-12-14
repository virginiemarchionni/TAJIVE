function [Z index distance] = find_similar_patches_mex(PP,P,K)

  p = size(P,1);

  [Ip1 Ip2] = size(PP);
  Jp1 = Ip1 - p + 1;
  Jp2 = Ip2 - p + 1;

  [ZZ index distance] = calc_dist_make_patches(double(PP),double(P));
  index = reshape(index,[2 Jp1*Jp2])'+1;
  ZZ    = reshape(ZZ,[p p Jp1*Jp2]);

  [val idd] = sort(distance);

  Z = ZZ(:,:,idd(1:K));
  distance = distance(idd(1:K));
  index = index(idd(1:K),:);  

