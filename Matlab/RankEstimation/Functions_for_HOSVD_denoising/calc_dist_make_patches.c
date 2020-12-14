/* 

   [Z index distance] = calc_dist_make_patches(PP,P);

*/

#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  int p, Ip1, Ip2, Jp1, Jp2, j1, j2, p1, p2, l, pp, ii;
  double *PP, *P, *ZZ, *index, *distance;
  double temp, Pi;

  /* load 1st input variable */
  PP  = mxGetPr(prhs[0]);
  Ip1 = mxGetM(prhs[0]);
  Ip2 = mxGetN(prhs[0]);

  /* load 2nd input variable */
  P  = mxGetPr(prhs[1]);
  p  = mxGetM(prhs[1]);
  pp = p*p;

  /* define 1st output variable */
  Jp1 = Ip1 - p + 1;
  Jp2 = Ip2 - p + 1;
  plhs[0] = mxCreateDoubleMatrix(p*p*Jp1*Jp2,1,mxREAL);
  ZZ      = mxGetPr(plhs[0]);

  /* define 2nd output variable */
  plhs[1] = mxCreateDoubleMatrix(2*Jp1*Jp2,1,mxREAL);
  index   = mxGetPr(plhs[1]);

  /* define 3rd output variable */
  plhs[2] = mxCreateDoubleMatrix(Jp1*Jp2,1,mxREAL);
  distance= mxGetPr(plhs[2]);

  /* main program */
  l = 0;
  
  for(j2=0;j2<Jp2;j2++){
    for(j1=0;j1<Jp1;j1++){

      index[l*2]  =(double)j1;
      index[l*2+1]=(double)j2;

      temp = 0.0;
      ii   = 0;
      for(p2=0;p2<p;p2++){
        for(p1=0;p1<p;p1++){
          Pi   = PP[(j2+p2)*Ip1+(j1+p1)];
          temp = temp + (P[ii] - Pi)*(P[ii] - Pi)/pp;
          ZZ[l*(pp) + ii] = Pi;
          ii = ii + 1;
        }
      }
      
      distance[l] = temp;
      l = l+1;

    }
  }

}



































