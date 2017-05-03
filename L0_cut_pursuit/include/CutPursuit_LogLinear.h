#pragma once
#include "CutPursuitProblem.h"

namespace CP {

template <typename T>
class CutPursuit_LogLinear : public CutPursuitProblem<T>
{
    public:
    std::vector<std::vector<T>> componentVector;
        // only used with backward step - the sum of all observation in the component
    CutPursuit_LogLinear(int nbVertex = 1) : CutPursuitProblem<T>(nbVertex)
    {
        this->componentVector  = std::vector<std::vector<T>>(1);
    }

    virtual std::pair<T,T> compute_energy() override
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        std::pair<T,T> energyPair;
        T energy = 0;
        VertexIterator<T> i_ver;
        for (i_ver = boost::vertices(this->mainGraph).first;
             i_ver!= this->lastIterator; ++i_ver)
        {
            for(int i_dim=0; i_dim<this->dim; i_dim++)
            {
                energy -= vertex_attribute_map(*i_ver).weight
                        * log(vertex_attribute_map(*i_ver).observation[i_dim] + 1e-6)
                        * vertex_attribute_map(*i_ver).value[i_dim];
            }
        }
        energyPair.first = energy;
        energy = 0;
        EdgeIterator<T> i_edg, i_edg_end;
        for (boost::tie(i_edg, i_edg_end) = boost::edges(this->mainGraph);
             i_edg != i_edg_end; ++i_edg)
        {
            if (!edge_attribute_map(*i_edg).realEdge)
            {
                continue;
            }
            energy += .5 * edge_attribute_map(*i_edg).isActive * this->parameter.reg_strenth
                    * edge_attribute_map(*i_edg).weight;
        }
        energyPair.second = energy;
        return energyPair;
    }

    //=============================================================================================
    //=============================        SPLIT        ===========================================
    //=============================================================================================
    virtual std::size_t split(int ite = 0)
    { // split the graph by trying to find the best binary partition
      // each components is split into B and notB
        std::size_t saturation;
        //initialize h_1 and h_2 with kmeans
        //--------initilializing labels------------------------------------------------------------
        std::size_t ** corners = new std::size_t * [this->components.size()];
        for (std::size_t i_com = 0;i_com < this->components.size(); ++i_com)
        {
            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            corners[i_com] = new std::size_t[2];
        }
        this->computeCorners(corners);
        this->setCapacities(corners, this->parameter.reg_strenth);
        //compute flow
        boost::boykov_kolmogorov_max_flow(
                   this->mainGraph,
                   get(&EdgeAttribute<T>::capacity        , this->mainGraph),
                   get(&EdgeAttribute<T>::residualCapacity, this->mainGraph),
                   get(&EdgeAttribute<T>::reverseEdge     , this->mainGraph),
                   get(&VertexAttribute<T>::color         , this->mainGraph),
                   get(boost::vertex_index                , this->mainGraph),
                   this->source,
                   this->sink);
        for (std::size_t i_com = 0; i_com < this->components.size(); ++i_com)
        {
            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            delete [] corners[i_com];
        }
        delete [] corners;
        saturation = this->activate_edges();
        return saturation;
    }
    //=============================================================================================
    //=============================      COMPUTE CORNERS        ===================================
    //=============================================================================================
    inline void computeCorners( std::size_t ** corners)
    { //-----compute the 2 most populous labels------------------------------
        int i_com_int = -1;
        for (componentIterator<T> i_com = this->components.begin();
             i_com != this->components.end(); ++i_com)
        {
            i_com_int++;
            if (this->saturatedComponent[i_com_int])
            {
                continue;
            }
            std::pair<std::size_t, std::size_t> corners_pair = find_corners(i_com);
            corners[i_com_int][0] = corners_pair.first;
            corners[i_com_int][1] = corners_pair.second;
        }
        return;
    }
    //=============================================================================================
    //=============================     FIND_CORNERS        =======================================
    //=============================================================================================
    std::pair<std::size_t, std::size_t> find_corners(componentIterator<T> i_com)
    {
        // given a component will output the pairs of the most popoulous two labels
        VertexAttributeMap<T> vertex_attribute_map
                                    = boost::get(boost::vertex_bundle, this->mainGraph);
        T average_vector[this->dim];
        for(int i_dim=0; i_dim < this->dim; i_dim++)
        {
            average_vector[i_dim] = 0;
        }
        for (VertexComponentIterator<T> i_ver = i_com->begin();
             i_ver != i_com->end(); ++i_ver)
        {
            for(int i_dim=0; i_dim < this->dim; i_dim++)
            {
            average_vector[i_dim] += log(vertex_attribute_map[*i_ver].observation[i_dim]+1e-6)
                                *  vertex_attribute_map[*i_ver].weight;
            }
        }
        std::size_t indexOfMax = 0;
        for(int i_dim=1; i_dim < this->dim; i_dim++)
        {
            if(average_vector[indexOfMax] < average_vector[i_dim])
            {
                indexOfMax = i_dim;
            }
        }
        std::size_t indexOfSndMax = 0;
        for(int i_dim=1; i_dim < this->dim; i_dim++)
        {
            if (i_dim==indexOfMax)
            {
                continue;
            }
            if(average_vector[indexOfSndMax] < average_vector[i_dim])
            {
                indexOfSndMax = i_dim;
            }
        }
        return std::pair<std::size_t, std::size_t>(indexOfMax, indexOfSndMax);
    }
    //=============================================================================================
    //=============================       SET_CAPACITIES    =======================================
    //=============================================================================================
    inline void setCapacities(std::size_t ** corners, T iterativ_reg_strength)
    {
        VertexDescriptor<T> desc_v;
        EdgeDescriptor   desc_source2v, desc_v2sink, desc_v2source;
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        T cost_B, cost_notB; //the cost of being in B or not B, local for each component
        //----first compute the capacity in sink/node edges------------------------------------
        for (int i_com = 0; i_com < this->components.size(); i_com++)
        {

            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            for (int i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
            {
                desc_v    = this->components[i_com][i_ver];
                // because of the adjacency structure NEVER access edge (source,v) directly!
                desc_v2source = boost::edge(desc_v, this->source,this->mainGraph).first;
                desc_source2v = edge_attribute_map(desc_v2source).reverseEdge; //use reverseEdge instead
                desc_v2sink   = boost::edge(desc_v, this->sink,this->mainGraph).first;
                cost_B    = 0;
                cost_notB = 0;
                if (vertex_attribute_map(desc_v).weight==0)
                {
                    edge_attribute_map(desc_source2v).capacity = 0;
                    edge_attribute_map(desc_v2sink).capacity   = 0;
                    continue;
                }
                cost_B    += log(vertex_attribute_map(desc_v).observation[corners[i_com][0]] + 1e-6);
                cost_notB += log(vertex_attribute_map(desc_v).observation[corners[i_com][1]] + 1e-6);

                if (cost_B>cost_notB)
                {
                    edge_attribute_map(desc_source2v).capacity = cost_B - cost_notB;
                    edge_attribute_map(desc_v2sink).capacity   = 0.;
                }
                else
                {
                    edge_attribute_map(desc_source2v).capacity = 0.;
                    edge_attribute_map(desc_v2sink).capacity   = cost_notB - cost_B;
                }
            }
        }
        //----then set the vertex to vertex edges ---------------------------------------------
        EdgeIterator<T> i_edg, i_edg_end;
        for (boost::tie(i_edg, i_edg_end) = boost::edges(this->mainGraph);
             i_edg != i_edg_end; ++i_edg)
        {
            if (!edge_attribute_map(*i_edg).realEdge)
            {
                continue;
            }
            if (!edge_attribute_map(*i_edg).isActive)
            {
                edge_attribute_map(*i_edg).capacity
                        = edge_attribute_map(*i_edg).weight * iterativ_reg_strength;
            }
            else
            {
                edge_attribute_map(*i_edg).capacity = 0;
            }
        }
    }
    //=============================================================================================
    //=================================   COMPUTE_VALUE   =========================================
    //=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_value(std::size_t i_com) override
    {
        VertexAttributeMap<T> vertex_attribute_map
                                    = boost::get(boost::vertex_bundle, this->mainGraph);
        if (i_com == 0)
        {  // we allocate the space necessary for the component vector at the first read of the component
           this-> componentVector = std::vector<std::vector<T>>(this->components.size());
        }
        std::vector<T> average_vector(this->dim), component_value(this->dim);
        T totalWeight = 0;
        for(int i_dim=0; i_dim < this->dim; i_dim++)
        {
            average_vector[i_dim] = 0;
        }
        for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
                 i_ver != this->components[i_com].end(); ++i_ver)
            {
            for(int i_dim=0; i_dim < this->dim; i_dim++)
            {
            average_vector[i_dim] += log(vertex_attribute_map[*i_ver].observation[i_dim] + 1e-6)
                                *  vertex_attribute_map[*i_ver].weight;
            }
            totalWeight += vertex_attribute_map[*i_ver].weight;
            vertex_attribute_map(*i_ver).inComponent = i_com;
        }
        this->componentVector[i_com] = average_vector;
        std::size_t indexOfMax = 0;
        for(int i_dim=1; i_dim < this->dim; i_dim++)
        {
            if(average_vector[indexOfMax] < average_vector[i_dim])
            {
                indexOfMax = i_dim;
            }
        }
        for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
             i_ver != this->components[i_com].end(); ++i_ver)
        {
            for(int i_dim=0; i_dim<this->dim; i_dim++)
            {
               if (i_dim == indexOfMax)
               {
                   component_value[i_dim] = 1;
                   vertex_attribute_map(*i_ver).value[i_dim] = 1;
               }
               else
               {
                   component_value[i_dim] = 0;
                   vertex_attribute_map(*i_ver).value[i_dim] = 0;
               }
            }
        }
        return std::pair<std::vector<T>, T>(component_value, totalWeight);
    }
    //=============================================================================================
    //=================================   COMPUTE_MERGE_GAIN   =========================================
    //=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_merge_gain(VertexDescriptor<T> comp1, VertexDescriptor<T> comp2) override
    {
        VertexAttributeMap<T> reduced_vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->reducedGraph);
        VertexIndexMap<T> reduced_vertex_IndexMap = get(boost::vertex_index, this->reducedGraph);
        std::vector<T> mergedValue(this->dim), mergedVector(this->dim);
        T gain = 0;
        // compute the value obtained by mergeing the two connected components
        for(int i_dim=0; i_dim<this->dim; i_dim++)
        {
            mergedVector[i_dim] = this->componentVector[reduced_vertex_IndexMap(comp1)][i_dim]
                                + this->componentVector[reduced_vertex_IndexMap(comp2)][i_dim];
        }
        std::size_t indexOfMax = 0;
        for(int i_dim=1; i_dim < this->dim; i_dim++)
        {
            if(mergedVector[indexOfMax] < mergedVector[i_dim])
            {
                indexOfMax = i_dim;
            }
        }
//        std::cout << "Checking merge of " << reduced_vertex_IndexMap(comp1) << " and "
//                  << reduced_vertex_IndexMap(comp2) << " : " << std::endl;
        for(int i_dim=0; i_dim<this->dim; i_dim++)
        {
            if (i_dim == indexOfMax)
            {
                mergedValue[i_dim] = 1;
            }
            else
            {
                mergedValue[i_dim] = 0;
            }
            //std::cout << i_dim  <<"=> " << mergedVector[i_dim]  << " - " << mergedValue[i_dim] << std::endl;
            //std::cout << this->componentVector[reduced_vertex_IndexMap(comp1)][i_dim]  << " - " << reduced_vertex_attribute_map(comp1).value[i_dim] << std::endl;
            //std::cout << this->componentVector[reduced_vertex_IndexMap(comp2)][i_dim]  << " - " << reduced_vertex_attribute_map(comp2).value[i_dim] << std::endl;
            gain += mergedVector[i_dim] *  mergedValue[i_dim]
                  - this->componentVector[reduced_vertex_IndexMap(comp1)][i_dim]
                  * reduced_vertex_attribute_map(comp1).value[i_dim]
                  - this->componentVector[reduced_vertex_IndexMap(comp2)][i_dim]
                  * reduced_vertex_attribute_map(comp2).value[i_dim];
        }

        return std::pair<std::vector<T>, T>(mergedValue, gain);
    }
};
}
