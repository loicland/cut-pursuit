/*==================================================================
 * [X, it, Obj, Dif] = PFDR_graph_l22_d1_l1_mex(Y, La_l2, Eu, Ev, La_d1, La_l1, positivity, rho, condMin, difRcd, difTol, itMax, verbose)
 * 
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/PFDR_graph_quadratic_d1_l1.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i;
    const int V = mxGetNumberOfElements(prhs[0]);
    const int E = mxGetNumberOfElements(prhs[2]);
    const int *Eu = (int*) mxGetData(prhs[2]);
    const int *Ev = (int*) mxGetData(prhs[3]);
    const int pos = (int) mxGetScalar(prhs[6]);
    const int itMax = (int) mxGetScalar(prhs[11]);
    const int verbose = (int) mxGetScalar(prhs[12]);

    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[1]);

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
        const double rho = (double) mxGetScalar(prhs[7]);
        const double condMin = (double) mxGetScalar(prhs[8]);
        const double difRcd = (double) mxGetScalar(prhs[9]);
        const double difTol = (double) mxGetScalar(prhs[10]);

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

        /* precompute the weighted observation */
        double *La_l2Y;
        if (La_l2 != NULL){
            La_l2Y = (double*) mxMalloc(V*sizeof(double));
            for (i = 0; i < V; i++){ La_l2Y[i] = La_l2[i]*Y[i]; }
            Y = La_l2Y;
        }
        PFDR_graph_quadratic_d1_l1<double>(V, E, 0, X, Y, La_l2, Eu, Ev, La_d1, \
                                    La_l1, pos, DIAG, La_l2, rho, condMin, \
                                    difRcd, difTol, itMax, it, Obj, Dif, verbose);
                                   /* 22 arguments */
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
            for (i = 0; i <= *it; i++){ Obj[i] += y2; }

        }
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
        const float rho = (float) mxGetScalar(prhs[7]);
        const float condMin = (float) mxGetScalar(prhs[8]);
        const float difRcd = (float) mxGetScalar(prhs[9]);
        const float difTol = (float) mxGetScalar(prhs[10]);

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

        /* precompute the weighted observation */
        float *La_l2Y;
        if (La_l2 != NULL){
            La_l2Y = (float*) mxMalloc(V*sizeof(float));
            for (i = 0; i < V; i++){ La_l2Y[i] = La_l2[i]*Y[i]; }
            Y = La_l2Y;
        }

        PFDR_graph_quadratic_d1_l1<float>(V, E, 0, X, Y, La_l2, Eu, Ev, La_d1, \
                                    La_l1, pos, DIAG, La_l2, rho, condMin, \
                                    difRcd, difTol, itMax, it, Obj, Dif, verbose);
                                   /* 22 arguments */

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
            for (i = 0; i <= *it; i++){ Obj[i] += y2; }
        }

    }
    
    /* check inputs
    mexPrintf("V = %d, E = %d, N = %d, Y[0] = %g, La_l2 = %p, X[0] = %f\n \
    Eu[0] = %d, Ev[0] = %d, La_d1[0] = %g, La_l1 = %p, Ltype = %d, L[0] = %g\n \
    rho = %g, condMin = %g, difRcd = %g, difTol = %g, itMax = %d, *it = %d,\n \
    objRec = %d, difRec = %d verbose = %d\n", \
    V, E, N, Y[0], (void*) La_l2, X[0], Eu[0], Ev[0], La_d1[0], (void*) La_l1, Ltype, L[0], rho, \
    condMin, difRcd, difTol, itMax, *it, Obj != NULL, Dif != NULL, verbose);
    mexEvalString("pause");
    */
}
