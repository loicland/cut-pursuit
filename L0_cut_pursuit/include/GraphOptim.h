#pragma once
#include "Common.h"
#include <boost/graph/properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>
//#include "boost/graph/strong_components.hpp"

//#include <boost/property_map/iterator_property_map.hpp>


namespace CP {

typedef boost::graph_traits<boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> >::edge_descriptor
    EdgeDescriptor;

template <typename T> class VertexAttribute
{
public:
    typedef T calc_type;
public:
    T weight; //weight of the observation
    std::vector<T> observation; //obseserved value
    std::vector<T> value; //current value
    int color; //field use for the graph cut
    bool isSourceSink; //is the node the source or sink
    bool isBorder; //is the node part of an activated edge
    std::size_t inComponent; //index of the component in which the node belong
public:
    int Dim(){return value.size();}
    T & Value(unsigned int i) {return value[i];}
    T & Observation(unsigned int i) {return observation[i];}
    T & Weight() {return weight;}
    VertexAttribute(int dim = 1, T weight=1., bool sourceOrSink = false)
        :weight(weight), observation(dim,0.),value(dim,0.),color(-1),
         isSourceSink(sourceOrSink),isBorder(false){}
};

template <typename T> class EdgeAttribute
{
public:
    typedef T calc_type;
public:
    std::size_t index; //index of the edge (necessary for graph cuts)
    EdgeDescriptor reverseEdge; //pointer to the reverse edge, also necessary for graph cuts
    T weight; //weight of the edge
    T capacity; //capacity in the flow graph
    T residualCapacity; //necessary for graph cuts
    bool isActive; //is the edge in the support of the values
    bool realEdge; //is the edge between real nodes or link to source/sink
public:
    EdgeAttribute(T weight=1., std::size_t eIndex = 0, bool real = true)
        :index(eIndex), weight(weight), capacity(weight), residualCapacity(0), isActive(!real), realEdge(real) {}
};

template< typename T>
using Graph = typename boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexAttribute<T>, EdgeAttribute<T> >;

template< typename T>
using VertexDescriptor = typename boost::graph_traits<CP::Graph<T>>::vertex_descriptor;
template< typename T>
using VertexIndex    = typename boost::graph_traits<CP::Graph<T>>::vertices_size_type;
template< typename T>
using  EdgeIndex     = typename boost::graph_traits<CP::Graph<T>>::edges_size_type;
template< typename T>
using VertexIterator = typename boost::graph_traits<Graph<T>>::vertex_iterator;
template< typename T>
using EdgeIterator   = typename boost::graph_traits<Graph<T>>::edge_iterator;

template<typename T>
using VertexAttributeMap = boost::vec_adj_list_vertex_property_map<Graph<T>, Graph<T>*
, VertexAttribute<T>, VertexAttribute<T> &,  boost::vertex_bundle_t >;
template<typename T>
using EdgeAttributeMap = boost::adj_list_edge_property_map<
boost::directed_tag, EdgeAttribute<T>, EdgeAttribute<T> &
, long unsigned int, CP::EdgeAttribute<T>, boost::edge_bundle_t>;
template<typename T>
using VertexIndexMap = typename boost::property_map<Graph<T>, boost::vertex_index_t>::type;
template<typename T>
using EdgeIndexMap   = typename boost::property_map<Graph<T>, std::size_t EdgeAttribute<T>::*>::type;

template<typename T>
using componentIterator = typename std::vector<std::vector<VertexDescriptor<T>>>::iterator;
template<typename T>
using VertexComponentIterator = typename std::vector<VertexDescriptor<T>>::iterator;


template <typename T>
void addDoubledge(Graph<T> & g, VertexDescriptor<T>  source, VertexDescriptor<T>  target, T weight
                 , std::size_t eIndex, EdgeAttributeMap<T> & edge_attribute_map, bool real = true)
{
////        // Add edges between grid vertices. We have to create the edge and the reverse edge,
////        // then add the reverseEdge as the corresponding reverse edge to 'edge', and then add 'edge'
////        // as the corresponding reverse edge to 'reverseEdge'
    EdgeDescriptor edge, reverseEdge;
    edge             = boost::add_edge(source, target, g).first;
    reverseEdge      = boost::add_edge(target, source, g).first;
    EdgeAttribute<T> attrib_edge(weight,eIndex,real);
    EdgeAttribute<T> attrib_reverseEdge(weight,eIndex+1,real);
    attrib_edge.reverseEdge         = reverseEdge;
    attrib_reverseEdge.reverseEdge  = edge;
    edge_attribute_map(edge)        = attrib_edge;
    edge_attribute_map(reverseEdge) = attrib_reverseEdge;
}

template<typename T>
std::list<VertexDescriptor<T>> neighbors(Graph<T> g, VertexDescriptor<T> ver)
{   //produce the list of neighbors of a node
    std::list<VertexDescriptor<T>> neighs;
    typename boost::graph_traits<Graph<T>>::out_edge_iterator i_edg, i_edg_end;
    for (boost::tie(i_edg,i_edg_end) = boost::out_edges(ver, g);
        i_edg !=  i_edg_end; ++i_edg)
    {
        neighs.push_back(boost::target(*i_edg, g));
    }
    for (boost::tie(i_edg,i_edg_end) = boost::in_edges(ver, g);
        i_edg !=  i_edg_end; ++i_edg)
    {
        neighs.push_back(boost::source(*i_edg, g));
    }
    return neighs;
}

}







