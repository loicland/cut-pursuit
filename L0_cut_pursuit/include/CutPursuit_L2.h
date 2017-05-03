#pragma once
#include "CutPursuitProblem.h"

namespace CP {
template <typename T>
class CutPursuit_L2 : public CutPursuitProblem<T>
{
    public:
    virtual std::pair<T,T> compute_energy() override
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        std::pair<T,T> energyPair;
        T energy = 0;
        VertexIterator<T> i_ver;
//        std::cout << "===="<< std::endl;
        for (i_ver = boost::vertices(this->mainGraph).first;
             i_ver!= this->lastIterator; ++i_ver)
        {
            for(int i_dim=0; i_dim<this->dim; i_dim++)
            {
                energy += .5*vertex_attribute_map(*i_ver).weight
                        * pow(vertex_attribute_map(*i_ver).observation[i_dim]
                            - vertex_attribute_map(*i_ver).value[i_dim],2);
            }

            //std::cout << "=>" << *i_ver << " " << vertex_attribute_map(*i_ver).inComponent <<std::endl;
                      //<< " = " << vertex_attribute_map(*i_ver).weight
//                      << " " << vertex_attribute_map(*i_ver).observation[0]
//                      << " " << vertex_attribute_map(*i_ver).value[0] << std::endl;
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
    virtual std::size_t split(int ite = 0) override
    { // split the graph by trying to find the best binary partition
      // each components is split into B and notB
      // for each components we associate the value h_1 and h_2 to vertices in B or notB
      // the affectation as well as h_1 and h_2 are computed alternatively
        //tic();
        //--------loading structures---------------------------------------------------------------

        int nbComp = this->components.size();
        VertexAttributeMap<T> vertex_attribute_map
                   = boost::get(boost::vertex_bundle, this->mainGraph);
        VertexIndexMap<T> indexMap = boost::get(boost::vertex_index, this->mainGraph);
        std::size_t saturation;
        T iterativ_reg_strength;
        //initialize h_1 and h_2 with kmeans
        bool labels[this->nVertex];   //stores wether each vertex is B or notB
        this->init_labels(labels);
        T *** centers = new T ** [nbComp];
        for (int i_com = 0; i_com < nbComp; i_com++)
        {
            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            centers[i_com]    = new T * [2];
            centers[i_com][0] = new T [this->dim];
            centers[i_com][1] = new T [this->dim];
        }
        //-----main loop----------------------------------------------------------------
                // the optimal flow is iteratively approximated
        for (int i_step = 1; i_step <= this->parameter.flow_steps; i_step++)
        {
            //the regularization strength at this step
            iterativ_reg_strength = this->parameter.reg_strenth;// * i_step / this->parameter.flow_steps;
            //compute h_1 and h_2
            this->computeCenters(centers, nbComp,labels);
            this->setCapacities(nbComp, centers, iterativ_reg_strength);
            // update the capacities of the flow graph
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
            for (int i_com = 0; i_com < nbComp; i_com++)
            {
                if (this->saturatedComponent[i_com])
                {
                    continue;
                }
                for (int i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
                {
                    labels[indexMap(this->components[i_com][i_ver])]
                          = (vertex_attribute_map(this->components[i_com][i_ver]).color
                          == vertex_attribute_map(this->sink).color);
                    /*std::cout << i_com << " " << i_ver << " " << " " << indexMap(this->components[i_com][i_ver])
                              << " " << vertex_attribute_map(this->components[i_com][i_ver]).color << " "
                              << labels[indexMap(this->components[i_com][i_ver])]  << std::endl;*/
                 }
             }
        }
        for (std::size_t i_com = 0; i_com < this->components.size(); ++i_com)
        {
            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            delete [] centers[i_com][0];
            delete [] centers[i_com][1];
        }
        delete [] centers;
        saturation = this->activate_edges();
        return saturation;
    }
    //=============================================================================================
    //=============================      INIT_L2        ===========================================
    //=============================================================================================
    inline void init_labels(bool * labels)
    { //-----initialize the labelling for each components with kmeans------------------------------
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        VertexIndexMap<T> indexMap = boost::get(boost::vertex_index, this->mainGraph);
        int i_com_int = -1;
        for (componentIterator<T> i_com = this->components.begin();
             i_com != this->components.end(); ++i_com)
        {            
            i_com_int++;
            //std::cout << "===>" << i_com_int << "  = " << i_com->size();
            if (this->saturatedComponent[i_com_int])
            {
                continue;
            }
            cv::Mat data(i_com->size(), this->dim, CV_32F,0.);//, labels_kmean(i_com->size(), 1, int,0.);
            std::vector<int> labels_kmean;
            std::size_t indexInData = 0;
            for (VertexComponentIterator<T> i_ver = i_com->begin();
                 i_ver != i_com->end(); ++i_ver)
            {
                if (vertex_attribute_map[*i_ver].weight>0)
                {
                    for(int i_dim=0; i_dim < this->dim; i_dim++)
                    {
                        data.at<float>(indexInData,i_dim) = vertex_attribute_map[*i_ver].observation[i_dim];
                    }
                    indexInData++;
                }
            }
            if (indexInData>1)
            {
                int sumLabel = 0;
                data = data(cv::Rect_<T> (0,0,this->dim,indexInData));
                cv::kmeans(data,2,labels_kmean,cv::TermCriteria(CV_TERMCRIT_ITER, this->parameter.kmeans_ite, 1.0)
                          ,this->parameter.kmeans_resampling, cv::KMEANS_PP_CENTERS);
                indexInData = 0;
                for (VertexComponentIterator<T> i_ver = i_com->begin();
                     i_ver != i_com->end(); ++i_ver)
                {
                    if (vertex_attribute_map[*i_ver].weight>0)
                    {
                        labels[indexMap(*i_ver)] = (labels_kmean.at(indexInData)==0);
                        sumLabel+= (labels_kmean.at(indexInData)==0);
                        indexInData++;
                    }
                    else
                    {
                        labels[indexMap(*i_ver)] = true;
                    }
                }
                //std::cout << " / " << sumLabel << std::endl;
            }
            else
            {
                 labels[indexMap(*(i_com->begin()))] = true;
                 //std::cout <<  std::endl;
            }
        }
    }
    //=============================================================================================
    //=============================  COMPUTE_CENTERS_L2  ==========================================
    //=============================================================================================
    inline void computeCenters(T*** center, int nbComp, bool * labels)
    {
        //compute for each component the values of h_1 and h_2
        //T *** center = new T ** [nbComp];
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        VertexIndexMap<T> indexMap = boost::get(boost::vertex_index, this->mainGraph);
        for (int i_com = 0; i_com < nbComp; i_com++)
        {
            if (this->saturatedComponent[i_com])
            {
                continue;
            }
            //std::cout << i_com << " / " << nbComp << " * " << this->components[i_com].size() << std::endl;
            // first compute h_1 and h_2
            //center[i_com]    = new T * [2];
            //center[i_com][0] = new T [this->dim];
            //center[i_com][1] = new T [this->dim];
            T totalWeight[2];
            totalWeight[0] = 0.;
            totalWeight[1] = 0.;
            for(int i_dim=0; i_dim < this->dim; i_dim++)
            {
                center[i_com][0][i_dim] = 0.;
                center[i_com][1][i_dim] = 0.;
            }
            for (int i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
            {
                if (vertex_attribute_map(this->components[i_com][i_ver]).weight==0)
                {
                    continue;
                }
                if (labels[indexMap(this->components[i_com][i_ver])])
                {
                    totalWeight[0] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                    for(int i_dim=0; i_dim < this->dim; i_dim++)
                    {
                       center[i_com][0][i_dim] += vertex_attribute_map(this->components[i_com][i_ver]).observation[i_dim]
                                                * vertex_attribute_map(this->components[i_com][i_ver]).weight ;
                    }
                }
                else
                {
                   totalWeight[1] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                   for(int i_dim=0; i_dim < this->dim; i_dim++)
                   {
                      center[i_com][1][i_dim] += vertex_attribute_map(this->components[i_com][i_ver]).observation[i_dim]
                                               * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                   }
                }
            }
            if ((totalWeight[0] == 0)||(totalWeight[1] == 0))
            {
                //the component is saturated
                this->saturateComponent(i_com);
                for(int i_dim=0; i_dim < this->dim; i_dim++)
                {
                    center[i_com][0][i_dim] = vertex_attribute_map(this->components[i_com].back()).value[i_dim];
                    center[i_com][1][i_dim] = vertex_attribute_map(this->components[i_com].back()).value[i_dim];
                }
            }
            else
            {
                for(int i_dim=0; i_dim < this->dim; i_dim++)
                {
                    center[i_com][0][i_dim] = center[i_com][0][i_dim] / totalWeight[0];
                    center[i_com][1][i_dim] = center[i_com][1][i_dim] / totalWeight[1];
                    //std::cout << center[i_com][0][i_dim] << " | " << center[i_com][1][i_dim] << std::endl;
                }
                //std::cout << totalWeight[0] << " | " << totalWeight[1] << std::endl;
            }
        }
        return;
    }
    //=============================================================================================
    //=============================   SET_CAPACITIES     ==========================================
    //=============================================================================================
    inline void setCapacities(std::size_t nbComp, T*** centers, T iterativ_reg_strength)
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        VertexDescriptor<T> desc_v;
        EdgeDescriptor   desc_source2v, desc_v2sink, desc_v2source;
        T cost_B, cost_notB; //the cost of being in B or not B, local for each component
        //----first compute the capacity in sink/node edges------------------------------------
        for (int i_com = 0; i_com < nbComp; i_com++)
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
                for(int i_dim=0; i_dim < this->dim; i_dim++)
                {
                   cost_B += 0.5*vertex_attribute_map(desc_v).weight
                              * (pow(centers[i_com][0][i_dim],2) - 2 * (centers[i_com][0][i_dim]
                              * vertex_attribute_map(desc_v).observation[i_dim]));
                   cost_notB += 0.5*vertex_attribute_map(desc_v).weight
                              * (pow(centers[i_com][1][i_dim],2) - 2 * (centers[i_com][1][i_dim]
                              * vertex_attribute_map(desc_v).observation[i_dim]));
                }
                /*std::cout << i_ver << ": " << vertex_attribute_map(desc_v).weight << " = "
                          << vertex_attribute_map(desc_v).observation[0] << " , "
                          << centers[i_com][0][0] << " / "
                          << centers[i_com][1][0] << " , "
                          << cost_B << " / " << cost_notB << " " <<  cost_B - cost_notB <<std::endl;*/
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
        T totalWeight = 0;
        std::vector<T> compValue(this->dim);
        std::fill((compValue.begin()),(compValue.end()),0);
        for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
                 i_ver != this->components[i_com].end(); ++i_ver)
            {
                totalWeight += vertex_attribute_map(*i_ver).weight;
                //std::cout << i_com << " " << *i_ver << " : " << totalWeight << " " << vertex_attribute_map(*i_ver).observation[0] << std::endl;
                for(int i_dim=0; i_dim<this->dim; i_dim++)
                {
                     compValue[i_dim] += vertex_attribute_map(*i_ver).observation[i_dim]
                                       * vertex_attribute_map(*i_ver).weight;
                }
                vertex_attribute_map(*i_ver).inComponent = i_com;
                //std::cout << i_com << " = " << *i_ver << std::endl;
            }
            for(int i_dim=0; i_dim<this->dim; i_dim++)
            {
                compValue[i_dim] = compValue[i_dim] / totalWeight;
            }
            for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
                 i_ver != this->components[i_com].end(); ++i_ver)
            {
                for(int i_dim=0; i_dim<this->dim; i_dim++)
                {
                    vertex_attribute_map(*i_ver).value[i_dim] = compValue[i_dim];
                }
            }
        //std::cout << i_com << " : " << compValue[0]<< " : " << compValue[1] << " : " <<  compValue[2]<< " : "  << compValue[3] <<  " : "<< compValue[4] << std::endl;
        return std::pair<std::vector<T>, T>(compValue, totalWeight);
    }
    //=============================================================================================
    //=================================   COMPUTE_MERGE_GAIN   =========================================
    //=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_merge_gain(VertexDescriptor<T> comp1, VertexDescriptor<T> comp2) override
    {
        VertexAttributeMap<T> reduced_vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->reducedGraph);
        std::vector<T> mergedValue(this->dim);
        T gain = 0;
        // compute the value obtained by mergeing the two connected components
        for(int i_dim=0; i_dim<this->dim; i_dim++)
        {
            mergedValue[i_dim] =
                    (reduced_vertex_attribute_map(comp1).weight *
                     reduced_vertex_attribute_map(comp1).value[i_dim]
                    +reduced_vertex_attribute_map(comp2).weight *
                     reduced_vertex_attribute_map(comp2).value[i_dim])
                   /(reduced_vertex_attribute_map(comp1).weight
                    +reduced_vertex_attribute_map(comp2).weight);
            gain += 0.5 * (pow(mergedValue[i_dim],2)
                  * (reduced_vertex_attribute_map(comp1).weight
                    +reduced_vertex_attribute_map(comp2).weight)
                  - pow(reduced_vertex_attribute_map(comp1).value[i_dim],2)
                  * reduced_vertex_attribute_map(comp1).weight
                  - pow(reduced_vertex_attribute_map(comp2).value[i_dim],2)
                  * reduced_vertex_attribute_map(comp2).weight);
        }
        return std::pair<std::vector<T>, T>(mergedValue, gain);
    }
};
}
