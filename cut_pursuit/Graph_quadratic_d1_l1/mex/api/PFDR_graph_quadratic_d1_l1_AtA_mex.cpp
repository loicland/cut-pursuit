/*==================================================================
 * [X, it, Obj, Dif] = PFDR_graph_quadratic_d1_l1_AtA_mex(AtY, AtA, Eu, Ev, La_d1, La_l1, positivity, L, rho, condMin, difRcd, difTol, itMax, verbose)
 *
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/PFDR_graph_quadratic_d1_l1.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int V = mxGetNumberOfElements(prhs[0]);
    const int E = mxGetNumberOfElements(prhs[2]);
    const int *Eu = (int*) mxGetData(prhs[2]);
    const int *Ev = (int*) mxGetData(prhs[3]);
    const int pos = (int) mxGetScalar(prhs[6]);
    const Lipschtype Ltype = (mxGetNumberOfElements(prhs[7]) == 1) ? SCAL : DIAG;
    const int itMax = (int) mxGetScalar(prhs[12]);
    const int verbose = (int) mxGetScalar(prhs[13]);

    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[1]);

    if (mxIsDouble(prhs[0])){
        const double *AtY = (double*) mxGetData(prhs[0]);
        const double *AtA = (double*) mxGetData(prhs[1]);
        const double *La_d1 = (double*) mxGetData(prhs[4]);
        const double *La_l1 = NULL;
        if (mxGetNumberOfElements(prhs[5]) > 1){
            La_l1 = (double*) mxGetData(prhs[5]);
        }
        const double *L = (double*) mxGetData(prhs[7]);
        const double rho = (double) mxGetScalar(prhs[8]);
        const double condMin = (double) mxGetScalar(prhs[9]);
        const double difRcd = (double) mxGetScalar(prhs[10]);
        const double difTol = (double) mxGetScalar(prhs[11]);

        plhs[0] = mxCreateNumericMatrix(V, 1, mxDOUBLE_CLASS, mxREAL);
        double *X = (double*) mxGetData(plhs[0]);
        double *Obj = NULL;
        if (nlhs > 2){
            plhs[2] = mxCreateNumericMatrix(1, itMax+1, mxDOUBLE_CLASS, mxREAL);
            Obj = (double*) mxGetData(plhs[2]);
        }
        double *Dif = NULL;
        if (nlhs > 3){
            plhs[3] = mxCreateNumericMatrix(1, itMax, mxDOUBLE_CLASS, mxREAL);
            Dif = (double*) mxGetData(plhs[3]);
        }

        PFDR_graph_quadratic_d1_l1<double>(V, E, -V, X, AtY, AtA, Eu, Ev, La_d1, \
                                La_l1, pos, Ltype, L, rho, condMin, \
                                difRcd, difTol, itMax, it, Obj, Dif, verbose);
                                /* 22 arguments */
    }else{
        const float *AtY = (float*) mxGetData(prhs[0]);
        const float *AtA = (float*) mxGetData(prhs[1]);
        const float *La_d1 = (float*) mxGetData(prhs[4]);
        const float *La_l1 = NULL;
        if (mxGetNumberOfElements(prhs[5]) > 1){
            La_l1 = (float*) mxGetData(prhs[5]);
        }
        const float *L = (float*) mxGetData(prhs[7]);
        const float rho = (float) mxGetScalar(prhs[8]);
        const float condMin = (float) mxGetScalar(prhs[9]);
        const float difRcd = (float) mxGetScalar(prhs[10]);
        const float difTol = (float) mxGetScalar(prhs[11]);

        plhs[0] = mxCreateNumericMatrix(V, 1, mxSINGLE_CLASS, mxREAL);
        float *X = (float*) mxGetData(plhs[0]);
        float *Obj = NULL;
        if (nlhs > 2){
            plhs[2] = mxCreateNumericMatrix(1, itMax+1, mxSINGLE_CLASS, mxREAL);
            Obj = (float*) mxGetData(plhs[2]);
        }
        float *Dif = NULL;
        if (nlhs > 3){
            plhs[3] = mxCreateNumericMatrix(1, itMax, mxSINGLE_CLASS, mxREAL);
            Dif = (float*) mxGetData(plhs[3]);
        }

        PFDR_graph_quadratic_d1_l1<float>(V, E, -V, X, AtY, AtA, Eu, Ev, La_d1, \
                                La_l1, pos, Ltype, L, rho, condMin, \
                                difRcd, difTol, itMax, it, Obj, Dif, verbose);
                                /* 22 arguments */
    }

    /* check inputs
    mexPrintf("V = %d, E = %d, AtY[0] = %g, AtA[0] = %g, X[0] = %f\n \
    Eu[0] = %d, Ev[0] = %d, Ltype = %d, L[0] = %g, \
    La_d1[0] = %g, La_l1 = %p\n \
    rho = %g, condMin = %g, difRcd = %g, \
    difTol = %g, itMax = %d, *it = %d\n \
    objRec = %d, difRec = %d, verbose = %d\n", \
    V, E, AtY[0], AtA[0], X[0], Eu[0], Ev[0], Ltype, L[0], \
    La_d1[0], (void*) La_l1, rho, condMin, difRcd, \
    difTol, itMax, *it, Obj != NULL, Dif != NULL, verbose);
    mexEvalString("pause");
    */
    
}
