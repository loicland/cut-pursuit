/*==================================================================
 * [Cv, rX, CP_it, Time, Obj, Dif] = CP_PFDR_graph_l22_d1_l1_mex(Y, La_l2, Eu, Ev, La_d1, La_l1, positivity, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
 * 
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/CP_PFDR_graph_quadratic_d1_l1.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i;
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
        const double *Y = (double*) mxGetData(prhs[0]);
        const double *La_l2 = NULL;
        if (mxGetNumberOfElements(prhs[1]) > 1){
            La_l2 = (double*) mxGetData(prhs[1]);
        }
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

        /* precompute the weighted observation */
        double *La_l2Y;
        if (La_l2 != NULL){
            La_l2Y = (double*) mxMalloc(V*sizeof(double));
            for (i = 0; i < V; i++){ La_l2Y[i] = La_l2[i]*Y[i]; }
            Y = La_l2Y;
        }

        CP_PFDR_graph_quadratic_d1_l1<double>(V, E, 0, &rV, Cv, &rX, Y, La_l2, \
                                Eu, Ev, La_d1, La_l1, pos, CP_difTol, \
                                CP_itMax, CP_it, PFDR_rho, PFDR_condMin, \
                                PFDR_difRcd, PFDR_difTol, PFDR_itMax, \
                                Time, Obj, Dif, verbose, NULL);
                                /* 26 arguments */

        if (La_l2 != NULL){
            Y = (double*) mxGetData(prhs[0]);
            mxFree(La_l2Y);
        }
        /* correct the objective functional values for the constant 
         * 1/2 ||y||_{l2,La_l2}^2 */
        if (Obj != NULL){
            double y2 = 0.;
            if (La_l2 != NULL){
                for (i = 0; i < V; i++){ y2 += La_l2[i]*Y[i]*Y[i]; }
            }else{
                for (i = 0; i < V; i++){ y2 += Y[i]*Y[i]; }
            }
            y2 /= 2.;
            for (i = 0; i <= *CP_it; i++){ Obj[i] += y2; }
        }

        mxSetData(plhs[1], rX);
    }else{
        const float *Y = (float*) mxGetData(prhs[0]);
        const float *La_l2 = NULL;
        if (mxGetNumberOfElements(prhs[1]) > 1){
            La_l2 = (float*) mxGetData(prhs[1]);
        }
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

        /* precompute the weighted observation */
        float *La_l2Y;
        if (La_l2 != NULL){
            La_l2Y = (float*) mxMalloc(V*sizeof(float));
            for (i = 0; i < V; i++){ La_l2Y[i] = La_l2[i]*Y[i]; }
            Y = La_l2Y;
        }

        CP_PFDR_graph_quadratic_d1_l1<float>(V, E, 0, &rV, Cv, &rX, Y, La_l2, \
                                Eu, Ev, La_d1, La_l1, pos,  CP_difTol, \
                                CP_itMax, CP_it, PFDR_rho, PFDR_condMin, \
                                PFDR_difRcd, PFDR_difTol, PFDR_itMax, \
                                Time, Obj, Dif, verbose, NULL);
                                /* 26 arguments */

        if (La_l2 != NULL){
            Y = (float*) mxGetData(prhs[0]);
            mxFree(La_l2Y);
        }
        /* correct the objective functional values for the constant 
         * 1/2 ||y||_{l2,La_l2}^2 */
        if (Obj != NULL){
            float y2 = 0.f;
            if (La_l2 != NULL){
                for (i = 0; i < V; i++){ y2 += La_l2[i]*Y[i]*Y[i]; }
            }else{
                for (i = 0; i < V; i++){ y2 += Y[i]*Y[i]; }
            }
            y2 /= 2.f;
            for (i = 0; i <= *CP_it; i++){ Obj[i] += y2; }
        }
        mxSetData(plhs[1], rX);
    }
    mxSetM(plhs[1], rV);
    /* check inputs
    mexPrintf("V = %d, E = %d, Y[0] = %g, La_l2[0] = %g\n \
    Eu[0] = %d, Ev[0] = %d, La_d1[0] = %g, La_l1[0] = %g,, Cv[0] = %d\n \
    CP_difTol = %g, CP_itMax = %d, *CP_it = %d,\n \
    timeRec = %d, objRec = %d, difRec = %d verbose = %d\n", \
    V, E, Y[0], La_l2[0], Eu[0], Ev[0], La_d1[0], La_l1[0], Cv[0], \
    CP_difTol, CP_itMax, *CP_it, Time != NULL, Obj != NULL, Dif != NULL, verbose);
    mexEvalString("pause");
    */
}
