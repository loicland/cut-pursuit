#include <iostream>
#include <vector>
#include "mex.h"
//#include <opencv2/opencv.hpp>
#include "../include/API.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalzed minimal 
//partition problem
//
//Cut Pursuit: fast algorithms to learn piecewise constant functions on 
//general weighted graphs, Loic Landrieu and Guillaume Obozinski,2016.
//
//Produce a piecewise constant approximation of signal $y$ structured
//by the graph G=(V,e,mu,w) with mu the node weight and w the edgeweight:
//argmin \sum_{i \IN V}{mu_i * phi(x_I, y_I)} 
//+ \sum_{(i,j) \IN E}{w_{i,j} 1(x_I != x_J)}
//
//phi(X,Y) the fidelity function (3 are implemented)
//(x != y) the funciton equal to 1 if x!=y and 0 else
//
// LOIC LANDRIEU 2017
//
//=======================SYNTAX===================================================
//
//[solution, inComponent, components, Eu_red, Ev_red, edgeWeight_red, nodeWeight_red]
// = L0_cut_pursuit_segmentation(observation, Eu, Ev, lambda = 1, edgeWeight = [1 ... 1]
//                 , nodeWeight = [1 ... 1], mode = 1, speed = 1, verbose = false)
//-----INPUT-----
// DxN float observation : the observed signal
// Ex1 int Eu, Ev: the origin and destination of each node
// /!\ INDEX OF THE FIRST NODE MUST BE ZERO AND LAST NODE nNodes - 1
// Ex1 float  edgeWeight: the edge weight
// Nx1 float  nodeWeight: the node weight
// 1x1 float lambda : the regularization strength
// 1x1 float mode : the fidelity function
//      0 : linear (for simplex bound data)
//      1 : quadratic (default)
//   0<a<1: KL with a smoothing (for simplex bound data)
// 1x1 float speed : parametrization impacting performance
//      0 : slow but precise
//      1 : recommended (default)
//      2 : fast but approximated (no backward step)
//      3 : ludicrous - for prototyping (no backward step)
// 1x1 bool verose : verbosity
//      0 : silent
//      1 : recommended (default)
//      2 : chatty
//-----OUTPUT-----
// Nx1 float  solution: piecewise constant approximation
// Nx1 int inComponent: for each node, in which component it belongs
// n_node_redx1 cell components : for each component, list of the nodes
// n_node_red : number of components
// int n_edges_red : number of edges in reduced graph
// n_edges_redx1 int Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //---read dimensions----//
    const int nNod = mxGetN(prhs[0]); //number of nodes
    const int nObs = mxGetM(prhs[0]); // size of observation for each node
    const int nEdg = mxGetNumberOfElements(prhs[1]); //number of edges
    //---read inputs----
    std::vector< std::vector<float> > observation(nNod);
    const std::vector<int> Eu((int*)mxGetData(prhs[1]), (int*)mxGetData(prhs[1])+nEdg);
    const std::vector<int> Ev((int*)mxGetData(prhs[2]), (int*)mxGetData(prhs[2])+nEdg);
    const float lambda        = (float) mxGetScalar(prhs[3]); //reg strength
    const std::vector<float> edgeWeight((float*)mxGetData(prhs[4]), (float*)mxGetData(prhs[4])+nEdg);
    const std::vector<float> nodeWeight((float*)mxGetData(prhs[5]), (float*)mxGetData(prhs[5])+nNod);
    const float mode          = (float) mxGetScalar(prhs[6]); //fidelity
    const float speed         = (float) mxGetScalar(prhs[7]); //speed mode
    const float verbose       = (float) mxGetScalar(prhs[8]); //verbosity*/
    //--fill the observation----
    const float *observation_array = (float*) mxGetData(prhs[0]);
    std::vector< std::vector<float> > solution(nNod);
    int ind_obs = 0;
    for (int ind_nod = 0; ind_nod < nNod; ind_nod++)
    {
        observation[ind_nod] = std::vector<float>(nObs);
        solution[ind_nod]    = std::vector<float>(nObs);
        for (int ind_dim = 0; ind_dim < nObs; ind_dim++)
        {   
            observation[ind_nod][ind_dim] = observation_array[ind_obs];
            ind_obs++;
        }
    }
    //---set up output--------------------------------
    //plhs[0] = mxDuplicateArray(prhs[0]);

    if (nlhs == 1)
    {
        CP::cut_pursuit<float>(nNod, nEdg, nObs
                , observation, Eu, Ev, edgeWeight, nodeWeight
                , solution
                , lambda, mode, speed,verbose);

    } else if (nlhs > 1)
    {   //we want the reduced structure
        int n_nodes_red, & n_edges_red;
        std::vector<int> in_component, Eu_red, Ev_red;
        std::vector< std::vector<int> > components;
        std::vector<T> & edgeWeight_red, nodeWeight_red;
        
        CP::cut_pursuit<float>(nNod, nEdg, nObs
                , y, Eu, Ev, edgeWeight, nodeWeight
                , solution
                , in_component, components
                , n_nodes_red, n_edges_red,  Eu_red, Ev_red
                , edgeWeight_red,nodeWeight_red
                ,lambda, mode, speed,verbose);   

        plhs[1] = mxCreateNumericMatrix(nNod, 1 , mxINT32_CLASS, mxREAL);
        in_component_array = (int*) mxGetData(plhs[1]);
        std::copy(in_component.begin(), in_component.end(), in_component_array);

        //    plhs[2] = mxCreateNumericMatrix(1, 1 , mxINT32_CLASS, mxREAL);
        //    n_nodes_red_array = (int*) mxGetData(plhs[2]);

        plhs[3] = mxCreateNumericMatrix(n_edges_red, 1 , mxINT32_CLASS, mxREAL);
        Eu_red_array = (int*) mxGetData(plhs[3]);
        std::copy(Eu_red.begin(), Eu_red.end(), Eu_red_array);

        plhs[4] = mxCreateNumericMatrix(n_edges_red, 1 , mxINT32_CLASS, mxREAL);
        Ev_red_array = (int*) mxGetData(plhs[4]);
        std::copy(Ev_red.begin(), Ev_red.end(), Ev_red_array);

        plhs[5] = mxCreateNumericMatrix(n_edges_red, 1 ,mxSINGLE_CLASS, mxREAL);
        edgeWeight_red_array = (float*) mxGetData(plhs[5]);
        std::copy(edgeWeight_red.begin(), edgeWeight_red.end(), edgeWeight_red_array);

        plhs[6] = mxCreateNumericMatrix(n_nodes_red, 1 , mxSINGLE_CLASS, mxREAL);
        nodeWeight_red_array = (float*) mxGetData(plhs[6]);
        std::copy(nodeWeight_red.begin(), nodeWeight_red.end(), nodeWeight_red_array);
        
    }
 
    plhs[0] = mxDuplicateArray(prhs[0]);
    float * solution_array = (float *) mxGetData(plhs[0]);
    ind_obs = 0;
    for (int ind_nod = 0; ind_nod < nNod; ind_nod++)
    {
        for (int ind_dim = 0; ind_dim < nObs; ind_dim++)
        {   
            solution_array[ind_obs] = solution[ind_nod][ind_dim];
            ind_obs++;
        }
    }   
    
    //mxArray * mx = mxCreateDoubleMatrix(1,nNod, mxREAL);
    //std::copy(solution.begin(), solution.end(), mxGetPr(mx));
    //plhs[0] = mx;
    return;
}

