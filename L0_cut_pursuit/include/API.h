#pragma once
#include "Common.h"
#include "CutPursuit_L2.h"
#include "CutPursuit_Linear.h"
#include "CutPursuit_KL.h"
#include "CutPursuit_LogLinear.h"
namespace CP {
template<typename T>
void CutPursuit(const int nNodes, const int nEdges, const int nObs
          ,const T * observation, const int * Eu, const int * Ev, const T * edgeWeight, const T * nodeWeight
          ,T * solution, const T lambda, const T mode)
{
    CP::CutPursuitProblem<float> * cp = NULL;
    fidelityType fidelity;
    
    if (mode == 0)
    {
    	std::cout << " WITH L2 FIDELITY" << std::endl;
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     else if (mode == 1)
     {
        std::cout << " WITH LINEAR FIDELITY" << std::endl;
        fidelity = linear;
        cp = new CP::CutPursuit_Linear<float>();
     }
     else if (mode < 1)
     {
        std::cout << " WITH KULLBACK-LEIBLER FIDELITY" << std::endl;
        fidelity = KL;
        cp = new CP::CutPursuit_KL<float>();
	cp->parameter.smoothing = mode;
     }
     else
     {
        std::cout << " WITH LOG-LINEAR FIDELITY" << std::endl;
        fidelity = loglinear;
        cp = new CP::CutPursuit_LogLinear<float>();
        cp->parameter.smoothing = mode - 1;
     }
    
    cp->mainGraph = Graph<T>(nNodes);
    //--------fill the vertices--------------------------------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(boost::vertex_bundle, cp->mainGraph);
    VertexIterator<T> ite_nod = boost::vertices(cp->mainGraph).first;
    VertexAttribute<T> v_attribute (nObs);
    std::size_t ind_obs = 0;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {
        for(int i_dim=0; i_dim < nObs; i_dim++)
        {
            v_attribute.observation[i_dim] = observation[ind_obs];
            ind_obs++;
        }
        v_attribute.weight = nodeWeight[ind_nod];
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
   //--------build the edges--------------------------------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle, cp->mainGraph);
    for( int ind_edg = 0; ind_edg < nEdges; ind_edg++ )
    {
        addDoubledge(cp->mainGraph, boost::vertex(Eu[ind_edg], cp->mainGraph)
                    , boost::vertex(Ev[ind_edg], cp->mainGraph)
                    , edgeWeight[ind_edg],2 * ind_edg, edge_attribute_map);
    }
    //--------return the cut pursuit problem-------------------------------------------------------------
    cp->dim = nNodes;
    cp->parameter.reg_strenth = lambda;
    
    switch (fidelity)
    {
        case L2:
        {
            CP::CutPursuit_L2<float> * cpx = dynamic_cast<CP::CutPursuit_L2<float> *>(cp);
            cpx->run();
            break;
        }
        case linear:
        {
            CP::CutPursuit_Linear<float> * cpx = dynamic_cast<CP::CutPursuit_Linear<float> *>(cp);
            cpx->run();
            break;
        }
        case KL:
        {
            CP::CutPursuit_KL<float> * cpx = dynamic_cast<CP::CutPursuit_KL<float> *>(cp);
            cpx->run();
            break;
        }
        case loglinear:
        {
            CP::CutPursuit_LogLinear<float> * cpx = dynamic_cast<CP::CutPursuit_LogLinear<float> *>(cp);
            cpx->run();
            break;
        }
    }
    
    //------------write the solution-----------------------------
    std::size_t ind_sol = 0;	
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {
        for(int i_dim=0; i_dim < nObs; i_dim++)
        {
            solution[ind_sol] = vertex_attribute_map[*ite_nod].value[i_dim];
            ind_sol++;
        }
        ite_nod++;
    }
    delete cp;
    return;
}
}
