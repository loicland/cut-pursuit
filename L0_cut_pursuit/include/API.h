#pragma once
#include <omp.h>
#include "CutPursuit_L2.h"
#include "CutPursuit_Linear.h"
#include "CutPursuit_KL.h"

//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalized minimal
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
//C style inputs
//void cut_pursuit(const int nNodes, const int nEdges, const int nObs
//          ,const T * observation, const int * Eu, const int * Ev
//          ,const T * edgeWeight, const T * nodeWeight
//          ,T * solution,  const T lambda, const T mode, const T speed
//          , const float verbose)
//C++ style input
//void cut_pursuit(const int nNodes, const int nEdges, const int nObs
//          , std::vector< std::vector<T> > & observation
//          , const std::vector<int> & Eu, const std::vector<int> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector< std::vector<T> > & solution,  const T lambda, const T mode, const T speed
 //         , const float verbose)
//-----INPUT-----
// 1x1 int nNodes = number of nodes
// 1x1 int nEdges = number of edges
// 1x1 int nObs   = dimension of data on each node
// NxD float observation : the observed signal
// Ex1 int Eu, Ev: the origin and destination of each node
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
// 1x1 n_node_red : number of components
// 1x1 int n_edges_red : number of edges in reduced graph
// n_edges_redx1 int Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes

namespace CP {

//===========================================================================
//=====================    CREATE_CP      ===================================
//===========================================================================

template<typename T>
CutPursuit<T> * create_CP(const T mode, const float verbose)
{
    CP::CutPursuit<float> * cp = NULL;
    fidelityType fidelity = L2;
    if (mode == 0)
    {
        if (verbose > 0)
        {
            std::cout << " WITH LINEAR FIDELITY" << std::endl;
        }
        fidelity = linear;
        cp = new CP::CutPursuit_Linear<float>();
     }
     else if (mode == 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH L2 FIDELITY" << std::endl;
        }
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     else if (mode > 0 && mode < 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH KULLBACK-LEIBLER FIDELITY SMOOTHING : "
                      << mode << std::endl;
        }
        fidelity = KL;
        cp = new CP::CutPursuit_KL<float>();
        cp->parameter.smoothing = mode;
     }
     else
     {
        std::cout << " UNKNOWN MODE, SWICTHING TO L2 FIDELITY"
                << std::endl;
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     cp->parameter.fidelity = fidelity;
     cp->parameter.verbose  = verbose;
     return cp;
}

//===========================================================================
//=====================  cut_pursuit  C-style  ==============================
//===========================================================================
template<typename T>
void cut_pursuit(const int nNodes, const int nEdges, const int nObs
          ,const T * observation, const int * Eu, const int * Ev
          ,const T * edgeWeight, const T * nodeWeight
          ,T * solution,  const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, nNodes, nEdges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    std::size_t ind_sol = 0;	
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {        
        for(int i_dim=0; i_dim < nObs; i_dim++)
        {
            solution[ind_sol] = vertex_attribute_map[*ite_nod].value[i_dim];
            ind_sol++;
        }
        ite_nod++;
   }
    //delete cp;
    return;
}

//===========================================================================
//=====================  cut_pursuit  C++-style  ============================
//===========================================================================
template<typename T>
void cut_pursuit(const int nNodes, const int nEdges, const int nObs
          , std::vector< std::vector<T> > & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          ,std::vector< std::vector<T> > & solution,  const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, nNodes, nEdges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {        
        for(int ind_dim=0; ind_dim < nObs; ind_dim++)
        {
            solution[ind_nod][ind_dim] = vertex_attribute_map[*ite_nod].value[ind_dim];
        }
        ite_nod++;
   }
    //delete cp;
    return;
}
//===========================================================================
//=====================  cut_pursuit        ===================================
//===========================================================================

template<typename T>
void cut_pursuit(const int nNodes, const int nEdges, const int nObs
          , const T * observation, const int * Eu, const int * Ev
          , const T * edgeWeight, const T * nodeWeight
	  , T * solution
	  , int * in_component, std::vector< std::vector<int> > components
          , int * n_nodes_red, int * n_edges_red
          , int * Eu_red, int * Ev_red
          , T * edgeWeight_red, T * nodeWeight_red
          , const T lambda, const T mode, const T speed
          , const float verbose)
{
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, nNodes, nEdges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------resize the pointers-----------------------------
    n_nodes_red[0] = boost::num_vertices(cp->reduced_graph);
    n_edges_red[0] = boost::num_edges(cp->reduced_graph);
    in_component = new int[n_nodes_red[0]];
    components.resize(n_nodes_red[0]);
    Eu_red = new int[n_edges_red[0]];
    Ev_red = new int[n_edges_red[0]];
    edgeWeight_red = new T[n_edges_red[0]];
    nodeWeight_red = new T[n_nodes_red[0]];
   //------------write the solution-----------------------------
    std::size_t ind_sol = 0;	
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {        
        for(int i_dim=0; i_dim < nObs; i_dim++)
        {
            solution[ind_sol] = vertex_attribute_map[*ite_nod].value[i_dim];
            ind_sol++;
        }
        in_component[ind_sol] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    VertexAttributeMap<T> edges_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    n_nodes_red[0] = boost::num_vertices(cp->reduced_graph);
    n_edges_red[0] = boost::num_edges(cp->reduced_graph);

    ind_sol = 0;
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes_red[0]; ind_nod++ )
    {
        nodeWeight_red[ind_sol] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }

    delete cp;
    return;
}


//===========================================================================
//=====================     SET_UP_CP C style   =============================
//===========================================================================
template<typename T>
void set_up_CP(CutPursuit<T> * cp, const int nNodes, const int nEdges, const int nObs
               ,const T * observation, const int * Eu, const int * Ev
               ,const T * edgeWeight, const T * nodeWeight)
{
    cp->main_graph = Graph<T>(nNodes);
    cp->dim = nObs;
    //--------fill the vertices--------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    //the node attributes used to fill each node
    std::size_t ind_obs = 0;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {
        VertexAttribute<T> v_attribute (nObs);
        for(int i_dim=0; i_dim < nObs; i_dim++)
        { //fill the observation of v_attribute
            v_attribute.observation[i_dim] = observation[ind_obs];
            ind_obs++;
        }//and its weight
        v_attribute.weight = nodeWeight[ind_nod];
        //set the attributes of the current node
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
    //--------build the edges-----------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle
            , cp->main_graph);
    for( int ind_edg = 0; ind_edg < nEdges; ind_edg++ )
    {   //add edges in each direction
        addDoubledge(cp->main_graph, boost::vertex(Eu[ind_edg]
                    , cp->main_graph), boost::vertex(Ev[ind_edg]
                    , cp->main_graph), edgeWeight[ind_edg],2 * ind_edg
                    , edge_attribute_map);
    }
}

//===========================================================================
//=====================     SET_UP_CP C++ style  ============================
//===========================================================================
template<typename T>
void set_up_CP(CutPursuit<T> * cp, const int nNodes, const int nEdges, const int nObs
               ,const std::vector< std::vector<T>> observation, const std::vector<int> Eu, const std::vector<int> Ev
               ,const std::vector<T> edgeWeight, const std::vector<T> nodeWeight)
{
    cp->main_graph = Graph<T>(nNodes);
    cp->dim = nObs;
    //--------fill the vertices--------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    //the node attributes used to fill each node
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {
        VertexAttribute<T> v_attribute (nObs);
        for(int i_dim=0; i_dim < nObs; i_dim++)
        { //fill the observation of v_attribute
            v_attribute.observation[i_dim] = observation[ind_nod][i_dim];
        }//and its weight
        v_attribute.weight = nodeWeight[ind_nod];
        //set the attributes of the current node
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
    //--------build the edges-----------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle
            , cp->main_graph);
    for( int ind_edg = 0; ind_edg < nEdges; ind_edg++ )
    {   //add edges in each direction
        addDoubledge(cp->main_graph, boost::vertex(Eu[ind_edg]
                    , cp->main_graph), boost::vertex(Ev[ind_edg]
                    , cp->main_graph), edgeWeight[ind_edg],2 * ind_edg
                    , edge_attribute_map);
    }
}
//===========================================================================
//=====================      SET SPEED    ===================================
//===========================================================================
template<typename T>
void set_speed(CutPursuit<T> * cp, const T speed, const float verbose)
{
    if (speed == 3)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = LUDICROUS SPEED" << std::endl;
        }
        cp->parameter.flow_steps  = 1;
        cp->parameter.kmeans_ite  = 3;
        cp->parameter.kmeans_resampling = 1;
        cp->parameter.max_ite_main = 5;
        cp->parameter.backward_step = false;
        cp->parameter.stopping_ratio = 0.001;
    }
    if (speed == 2)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = FAST" << std::endl;
        }
        cp->parameter.flow_steps  = 2;
        cp->parameter.kmeans_ite  = 5;
        cp->parameter.kmeans_resampling = 2;
        cp->parameter.max_ite_main = 5;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0.001;
    }
    else if (speed == 0)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = SLOW" << std::endl;
        }
        cp->parameter.flow_steps  = 4;
        cp->parameter.kmeans_ite  = 8;
        cp->parameter.kmeans_resampling = 5;
        cp->parameter.max_ite_main = 20;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0;
    }
    else if (speed == 1)
    {
        if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = STANDARD" << std::endl;
        }
        cp->parameter.flow_steps  = 3;
        cp->parameter.kmeans_ite  = 5;
        cp->parameter.kmeans_resampling = 2;
        cp->parameter.max_ite_main = 10;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0.0001;
    }
*/
}

}
