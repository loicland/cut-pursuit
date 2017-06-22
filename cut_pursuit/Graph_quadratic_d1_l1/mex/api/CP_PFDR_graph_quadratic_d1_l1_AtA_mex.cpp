/*==================================================================
 * [Cv, rX, CP_it, Time, Obj, Dif] = CP_PFDR_graph_quadratic_d1_l1_AtA_mex(AtY, AtA, Eu, Ev, La_d1, La_l1, positivity, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
 * 
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/CP_PFDR_graph_quadratic_d1_l1.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int V = mxGetNumberOfElements(prhs[0]);
    const int E = mxGetNumberOfElements(prhs[2]);
    const int *Eu = (int*) mxGetData(prhs[2]);
    const int *Ev = (int*) mxGetData(prhs[3]);
    const int pos = (int) mxGetScalar(prhs[6]);
    const int CP_itMax = (int) mxGetScalar(prhs[8]);
    const int PFDR_itMax = (int) mxGetScalar(prhs[13]);
    const int verbose = (int) mxGetScalar(prhs[14]);

    plhs[0] = mxCreateNumericMatrix(V, 1, mxINT32_CLASS, mxREAL);
    int *Cv = (int*) mxGetData(plhs[0]);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *CP_it = (int*) mxGetData(plhs[2]);
    int rV;
    double *Time = NULL;
    if (nlhs > 3){
        plhs[3] = mxCreateNumericMatrix(1, CP_itMax+1, mxDOUBLE_CLASS, mxREAL);
        Time = (double*) mxGetData(plhs[3]);
    }

    if (mxIsDouble(prhs[0])){
        const double *AtY = (double*) mxGetData(prhs[0]);
        const double *AtA = (double*) mxGetData(prhs[1]);
        const double *La_d1 = (double*) mxGetData(prhs[4]);
        const double *La_l1 = NULL;
        if (mxGetNumberOfElements(prhs[5]) > 1){
            La_l1 = (double*) mxGetData(prhs[5]);
        }
        const double CP_difTol = (double) mxGetScalar(prhs[7]);
        const double PFDR_rho = (double) mxGetScalar(prhs[9]);
        const double PFDR_condMin = (double) mxGetScalar(prhs[10]);
        const double PFDR_difRcd = (double) mxGetScalar(prhs[11]);
        const double PFDR_difTol = (double) mxGetScalar(prhs[12]);

        plhs[1] = mxCreateNumericMatrix(0, 1, mxDOUBLE_CLASS, mxREAL);
        double *rX;
        double *Obj = NULL;
        if (nlhs > 4){
            plhs[4] = mxCreateNumericMatrix(1, CP_itMax+1, mxDOUBLE_CLASS, mxREAL);
            Obj = (double*) mxGetData(plhs[4]);
        }
        double *Dif = NULL;
        if (nlhs > 5){
            plhs[5] = mxCreateNumericMatrix(1, CP_itMax, mxDOUBLE_CLASS, mxREAL);
            Dif = (double*) mxGetData(plhs[5]);
        }

        CP_PFDR_graph_quadratic_d1_l1<double>(V, E, -V, &rV, Cv, &rX, AtY, AtA, \
                                Eu, Ev, La_d1, La_l1, pos, CP_difTol, \
                                CP_itMax, CP_it, PFDR_rho, PFDR_condMin, \
                                PFDR_difRcd, PFDR_difTol, PFDR_itMax, \
                                Time, Obj, Dif, verbose, NULL);
                                /* 26 arguments */
        mxSetData(plhs[1], rX);
    }else{
        const float *AtY = (float*) mxGetData(prhs[0]);
        const float *AtA = (float*) mxGetData(prhs[1]);
        const float *La_d1 = (float*) mxGetData(prhs[4]);
        const float *La_l1 = NULL;
        if (mxGetNumberOfElements(prhs[5]) > 1){
            La_l1 = (float*) mxGetData(prhs[5]);
        }
        const float CP_difTol = (float) mxGetScalar(prhs[7]);
        const float PFDR_rho = (float) mxGetScalar(prhs[9]);
        const float PFDR_condMin = (float) mxGetScalar(prhs[10]);
        const float PFDR_difRcd = (float) mxGetScalar(prhs[11]);
        const float PFDR_difTol = (float) mxGetScalar(prhs[12]);

        plhs[1] = mxCreateNumericMatrix(0, 1, mxSINGLE_CLASS, mxREAL);
        float *rX;
        float *Obj = NULL;
        if (nlhs > 4){
            plhs[4] = mxCreateNumericMatrix(1, CP_itMax+1, mxSINGLE_CLASS, mxREAL);
            Obj = (float*) mxGetData(plhs[4]);
        }
        float *Dif = NULL;
        if (nlhs > 5){
            plhs[5] = mxCreateNumericMatrix(1, CP_itMax, mxSINGLE_CLASS, mxREAL);
            Dif = (float*) mxGetData(plhs[5]);
        }

        CP_PFDR_graph_quadratic_d1_l1<float>(V, E, -V, &rV, Cv, &rX, AtY, AtA, \
                                Eu, Ev, La_d1, La_l1, pos, CP_difTol, \
                                CP_itMax, CP_it, PFDR_rho, PFDR_condMin, \
                                PFDR_difRcd, PFDR_difTol, PFDR_itMax, \
                                Time, Obj, Dif, verbose, NULL);
                                /* 26 arguments */
        mxSetData(plhs[1], rX);
    }
    mxSetM(plhs[1], rV);
    /* check inputs
    mexPrintf("V = %d, E = %d, N = %d, AtY[0] = %g, AtA[0] = %g\n \
    Eu[0] = %d, Ev[0] = %d, La_d1[0] = %g, La_l1[0] = %g,, Cv[0] = %d\n \
    difTol = %g, CP_itMax = %d, *CP_it = %d,\n \
    timeRec = %d, difRec = %d verbose = %d\n", \
    V, E, N, AtY[0], AtA[0], Eu[0], Ev[0], La_d1[0], La_l1[0], Cv[0], \
    difTol, CP_itMax, *CP_it, Time != NULL, Dif != NULL, verbose);
    mexEvalString("pause");
    */
}
