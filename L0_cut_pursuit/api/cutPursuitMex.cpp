#include <iostream>
#include <vector>
#include "mex.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "../include/API.h"
#include "../include/CutPursuit_L2.h"
#include "../include/CutPursuit_Linear.h"
#include "../include/CutPursuit_KL.h"
#include "../include/CutPursuit_LogLinear.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//MULTI FORMAT SPATIAL REGULARIZATION ALGORITHM
//SOLVE PROBLEM OF THE FOLLOWING FORMS:
//ARGMIN \SUM_{I \IN V}{W_I * PHI(X_I, Y_I)} + \SUM_{(I,J) \IN E}{W_{I,J} 1(X_I != X_J)}
//WITH
//G=(V,E) THE GRAPH CAPTURING THE SPATIAL STRUCTURE OF THE DATA
//W_I THE WEIGHT ENCODING THE IMPORTANCE OF EACH NODE
//Y THE OBSERVED VALUE TO REGULARIZE
//PHI(X,Y) A FIDELITY FUNCTION (4 FIDELITY FUNCTIONS ARE IMPLEMENTED)
//W_{I,J} THE WEIGHT OF EDGE (I,J), ENCODING PROXIMITY
//1(x != y) IS THE FUNCTION EQUAL TO 0 WHEN x != y  AND 0 EVERYWHERE ELSE
//
// LOIC LANDRIEU 2016
//
//=======================SYNTAX===================================================
//L IS THE REGULARIZATION STRENGTH
//F IS THE FIDELITY FUNCTION
//CP(T * observation, int * Eu, int * Ev, T * edgeWeight, T * nodeWeight, T, lambda,  T mode)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int nNod = mxGetM(prhs[0]);
    const int nObs = mxGetN(prhs[0]);
    const int nEdg = mxGetNumberOfElements(prhs[1]);
    const int *Eu = (int*) mxGetData(prhs[1]);
    const int *Ev = (int*) mxGetData(prhs[2]);
    plhs[0] = mxDuplicateArray(prhs[0]);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    
    const float * observation = (float *) mxGetData(prhs[0]);
    const float * edgeWeight  = (float *) mxGetData(prhs[3]);
    const float * nodeWeight  = (float *) mxGetData(prhs[4]);
    const float lambda        = (float) mxGetScalar(prhs[5]);
    const float mode          = (float) mxGetScalar(prhs[6]);
    float * solution          = (float *) mxGetData(plhs[0]);

    CP::CutPursuit<float>(nNod, nEdg, nObs, observation
      , Eu, Ev, edgeWeight, nodeWeight, solution, lambda, mode);
}
