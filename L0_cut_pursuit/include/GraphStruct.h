#pragma once
#include "CutPursuitProblem.h"
namespace CP {

template<typename T>
void bin2cp(CutPursuitProblem<T> * cp, int nNodes, int nEdges, int nObs
          ,T * observation, int * Eu, int * Ev, T * edgeWeight, T * nodeWeight
	  ,T * solution, int * component)
{
    cp->mainGraph = Graph<T>(nNodes);
    //--------fill the vertices--------------------------------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(boost::vertex_bundle, cp->mainGraph);
    VertexIterator<T> ite_nod = boost::vertices(cp->mainGraph).first;
    VertexAttribute<T> v_attribute (nObs);
    std::size_t ind_obs = 0;
    for(int ind_nod = 0; ind_nod < nNodes; ind_nod++ )
    {
        for(int i_dim=0; i_dim < dim; i_dim++)
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
    return;
}
}
