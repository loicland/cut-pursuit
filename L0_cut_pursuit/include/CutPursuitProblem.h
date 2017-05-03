#pragma once
#include "GraphOptim.h"
#include <math.h>
#include <queue>
#include <iostream>
#include <fstream>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
namespace CP {

template <typename T>
struct CPparameter
{
    T   reg_strenth;
    int flow_steps;
    int kmeans_ite;
    int kmeans_resampling;
    int verbose;
    int maxIte;
    bool backwardStep;
    double stoppingCriteria;
    fidelityType fidelity;
    double smoothing; //smoothing for Kl divergence
};

template <typename T>
class CutPursuitProblem
{
    public:
    Graph<T> mainGraph;
    Graph<T> reducedGraph;
    std::vector<std::vector<VertexDescriptor<T>>> components;
    std::vector<VertexDescriptor<T>> rootVertex; //a root for each connected components
    std::vector<bool> saturatedComponent;
    std::vector<std::vector<EdgeDescriptor>> borders;
    VertexDescriptor<T> source;
    VertexDescriptor<T> sink;
    std::size_t dim;     // the dimension of the data
    std::size_t nVertex; // the number Of data point
    std::size_t nEdge;   // the number Of edges between vertices (not counting the edge to source/sink)
    CP::VertexIterator<T> lastIterator;
    CPparameter<T> parameter;
    GenericParameter * targetParameter;
    //cv::Mat  Image;
    int  natureOfData;
    public:
    CutPursuitProblem(int nbVertex = 1)
    {
        this->mainGraph      = Graph<T>(nbVertex);
        this->reducedGraph   = Graph<T>(1);
        this->components     = std::vector<std::vector<VertexDescriptor<T>>>(1);
        this->rootVertex     = std::vector<VertexDescriptor<T>>(1,0);
        this->saturatedComponent = std::vector<bool>(1,false);
        this->source         = VertexDescriptor<T>();
        this->sink           = VertexDescriptor<T>();
        this->dim            = 1;
        this->nVertex        = 1;
        this->nEdge          = 0;
        this->parameter.flow_steps  = 3;
        this->parameter.kmeans_ite  = 5;
        this->parameter.kmeans_resampling = 3;
        this->parameter.verbose = 2;
        this->parameter.maxIte = 6;
        this->parameter.backwardStep = true;
        this->parameter.stoppingCriteria = 0.0001;
        this->parameter.fidelity = L2;
        this->parameter.smoothing = 0.1;
    }
    virtual ~CutPursuitProblem()
    {
       delete this->targetParameter;
    }
    //=============================================================================================
    std::pair<std::vector<T>, std::vector<T>> run()
    {
        this->initialize();
        std::cout << "Graph " << boost::num_vertices(this->mainGraph)  << " vertices and "
                  <<   boost::num_edges(this->mainGraph) << " edges and observation of dimension "
                  << this->dim << std::endl;
        std::cout << "Graph structure built - starting optimization " << std::endl;
        T energyZero = this->compute_energy().first;
        printf("%4.3f +  %4.3f \n ", this->compute_energy().first, this->compute_energy().second);
        T oldEnergy = energyZero;
        std::vector<T> ene( this->parameter.maxIte ),tim(this->parameter.maxIte);
        std::ofstream benchamrkFile;
        benchamrkFile.open ("benchMarking.txt");
        benchamrkFile << "     T               F    \n";
        TimeStack ts; ts.tic();
        for (int ite = 1; ite <= this->parameter.maxIte; ite++)
        {
             //this->check3();
            std::size_t saturation = this->split(ite);
            this->reduce(ite);
            std::pair<T,T> energy = this->compute_energy();
            printf("Iteration %3i - %4i components - ", ite, (int)this->components.size());
            printf("Saturation %5.1f %% - ",100*saturation / (double) this->nVertex);
            switch (this->targetParameter->fidelity)
            {
                case L2:
                {
                    printf("Energy %4.3f %% - ", 100 * (energy.first + energy.second) / energyZero);
                    break;
                }
                case linear:
                {
                    printf("Energy %10.1f - ", energy.first + energy.second);
                    break;
                }
                case KL:
                {
                    printf("Energy %4.3f %% - ", 100 * (energy.first + energy.second) / energyZero);
                    break;
                }
                case loglinear:
                {
                    printf("Energy %10.1f - ", energy.first + energy.second);
                    break;
                }
            }
            std::cout << "Timer  " << ts.toc() << std::endl;
            printf("%4.3f +  %4.3f \n ", energy.first, energy.second);
            switch (this->targetParameter->fidelity)
            {
                case L2:
                {
                    ene.push_back((energy.first + energy.second) / energyZero);
                }
                case linear:
                {
                    ene.push_back((energy.first + energy.second));
                }
                case KL:
                {
                    ene.push_back((energy.first + energy.second) / energyZero);
                }
                case loglinear:
                {
                    ene.push_back((energy.first + energy.second));
                }
            }

            tim.push_back(ts.tocDouble());
            benchamrkFile << tim.back() << "         "  << ene.back()   << "\n";
            if (saturation == (double) this->nVertex)
            {
                break;
            }
            if ((oldEnergy - energy.first - energy.second) / (energyZero - energy.first - energy.second)
               < this->parameter.stoppingCriteria)
            {
                break;
            }
            if (ite>=this->parameter.maxIte)
            {
                break;
            }
            oldEnergy = energy.first + energy.second;
           //std::cout << "Time :"   << toc() << std::endl;
        }
        this->compute_reduced_graph(0);
        benchamrkFile.close();
        return std::pair<std::vector<T>, std::vector<T>>(ene, tim);
    }
    //=============================================================================================
    //=========== VIRTUAL METHODS DEPENDING ON THE CHOICE OF FIDELITY FUNCTION =====================
    //=============================================================================================
    //
    //=============================================================================================
    //=============================        SPLIT        ===========================================
    //=============================================================================================
    virtual std::size_t split(int ite = 0)
    {
        std::cout << "SHOULD NOT BE CALLED : SPLIT" << std::endl;
        return 0;
    }
    //=============================================================================================
    //================================     compute_energy_L2      ====================================
    //=============================================================================================
    virtual std::pair<T,T> compute_energy()
    {
        std::cout << "SHOULD NEVER BE CALLED : COMPUTE_ENERGY" << std::endl;
        return std::pair<T,T>(0,0);
    }
    //=============================================================================================
    //=================================   COMPUTE_VALUE   =========================================
    //=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_value(std::size_t i_com)
    {
        std::cout << "SHOULD NEVER BE CALLED : COMPUTE_VALUE" << std::endl;
        return std::pair<std::vector<T>, T>(std::vector<T>(0),0);
    }
//=============================================================================================
//=================================   COMPUTE_MERGE_GAIN   =========================================
//=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_merge_gain(VertexDescriptor<T> comp1, VertexDescriptor<T> comp2)
    {
    std::cout << "SHOULD NEVER BE CALLED : COMPUTE_MERGE_GAIN" << std::endl;
    return std::pair<std::vector<T>, T>(std::vector<T>(0),0);
    }
    //=============================================================================================
    //========================== END OF VIRTUAL METHODS ===========================================
    //=============================================================================================
    //
    //=============================================================================================
    //=============================     INITIALIZE      ===========================================
    //=============================================================================================
    void initialize()
    {
        //build the reduced graph with one component, fill the first vector of components
        //and add the sink and source nodes
        VertexIterator<T> vi, vi_end;
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        this->components[0] = std::vector<VertexDescriptor<T>> (0);//(this->nVertex);
        this->rootVertex[0] = *boost::vertices(this->mainGraph).first;
        this->nVertex = boost::num_vertices(this->mainGraph);
        this->nEdge   = boost::num_edges(this->mainGraph);
        //--------compute the first reduced graph----------------------------------------------------------
        //std::cout << vertex_attribute_map(*boost::vertices(this->mainGraph).first).observation[0] << std::endl;
        for (boost::tie(vi, vi_end) = boost::vertices(this->mainGraph);
             vi != vi_end; ++vi)
        {
            this->components[0].push_back(*vi);
        }
        this->compute_value(0);
        //--------build the link to source and sink--------------------------------------------------------
        this->source = boost::add_vertex(this->mainGraph);
        this->sink   = boost::add_vertex(this->mainGraph);
        vertex_attribute_map(this->source).isSourceSink = true;
        vertex_attribute_map(this->sink).isSourceSink = true;
        std::size_t eIndex = boost::num_edges(this->mainGraph);
        vi = boost::vertices(this->mainGraph).first;
        for (int i_ver = 0;  i_ver < this->nVertex ; i_ver++)
        {
            // note that source and edge will have many nieghbors, and hence boost::edge should never be called to get
            // the in_edge. use the out_edge and then reverse_Edge
            addDoubledge<T>(this->mainGraph, this->source, boost::vertex(i_ver, this->mainGraph), 0.,
                         eIndex, edge_attribute_map , false);
            eIndex +=2;
            addDoubledge<T>(this->mainGraph, boost::vertex(i_ver, this->mainGraph), this->sink, 0.,
                         eIndex, edge_attribute_map, false);
            eIndex +=2;
            ++vi;
        }
        this->lastIterator = vi;
    }
    //=============================================================================================
    //================================  COMPUTE_REDUCE_VALUE  ====================================
    //=============================================================================================
    void compute_reduced_value()
    {
        for (std::size_t i_com = 0;  i_com < this->components.size(); i_com++)
        {
            compute_value(i_com);
        }
    }
    //=============================================================================================
    //=============================   ACTIVATE_EDGES     ==========================================
    //=============================================================================================
    std::size_t activate_edges()
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        VertexAttributeMap<T> vertex_attribute_map_reduced
                = boost::get(boost::vertex_bundle, this->reducedGraph);
        std::size_t saturation = 0;
        int nbComp = this->components.size();
        for (int i_com = 0; i_com < nbComp; i_com++)
        {
            if (this->saturatedComponent[i_com])
            {
                saturation += this->components[i_com].size();
                continue;
            }
            T totalWeight[2];
            totalWeight[0] = 0.;
            totalWeight[1] = 0.;
            int totalCard[2];
            for (int i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
            {
                bool isSink
                        = (vertex_attribute_map(this->components[i_com][i_ver]).color
                        ==  vertex_attribute_map(this->sink).color);
                if (isSink)
                {
                    totalWeight[0] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                    totalCard[0]++;
                }
                else
                {
                   totalWeight[1] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                   totalCard[1]++;
                }
            }
           // std::cout << i_com << " => " <<vertex_attribute_map_reduced(i_com).weight << " = "
           //          << totalWeight[0] << " / " <<totalWeight[1] << std::endl;
            if ((totalWeight[0] == 0)||(totalWeight[1] == 0))
            {
                //the component is saturated
                //std::cout << "<========================================"<< std::endl;
                this->saturateComponent(i_com);
                saturation += this->components[i_com].size();
            }
        }
        EdgeIterator<T> i_edg, i_edg_end;
        int color_v1, color_v2, color_combination;
        for (boost::tie(i_edg, i_edg_end) = boost::edges(this->mainGraph);
             i_edg != i_edg_end; ++i_edg)
        {
            if (!edge_attribute_map(*i_edg).realEdge )
            {
                continue;
            }
            color_v1 = vertex_attribute_map(boost::source(*i_edg, this->mainGraph)).color;
            color_v2 = vertex_attribute_map(boost::target(*i_edg, this->mainGraph)).color;
            //color_source = 0, color_sink = 4, uncolored = 1
            //we want an edge when a an interface source/sink
            //for the case of uncolored nodes we arbitrarily chose source-uncolored
            color_combination = color_v1 + color_v2;
            if ((color_combination == 0)||(color_combination == 2)||(color_combination == 5)
              ||(color_combination == 8))
            {
                continue;
            }
            //the edge is active!
            edge_attribute_map(*i_edg).isActive = true;
            edge_attribute_map(*i_edg).capacity = 0;
            vertex_attribute_map(boost::source(*i_edg, this->mainGraph)).isBorder = true;
            vertex_attribute_map(boost::target(*i_edg, this->mainGraph)).isBorder = true;
        }
        return saturation;
    }

    //=============================================================================================
    //=============================        REDUCE       ===========================================
    //=============================================================================================
    void reduce(int ite = 0)
    {
                //this->check3();
        this->connected_comp_roots();
                //this->check3();
        if (this->parameter.backwardStep)
        {
            this->compute_reduced_graph(0);
            //std::cout << "REDUCED GRAPH COMPUTED" << std::endl;
            this->merge();
        }
        else
        {
            this->compute_reduced_value();
        }
    }

    //=============================================================================================
    //==============================  CONNECTED_COMP_ROOTS=========================================
    //=============================================================================================
    void connected_comp_roots()
    {
        //this function compute the connected components of the graph with active edges removed
        std::vector<VertexDescriptor<T>> heapComponent(1); //the vertices in the current connected component
        heapComponent.reserve(this->nVertex);
        //the boolean vector indicating wether or not the edges and vertices have been seen already
        std::vector<bool> edgeSeen (this->nEdge);
        std::fill((edgeSeen.begin()),(edgeSeen.end()),false);
        std::vector<bool> nodeSeen (this->nVertex+2);
        std::fill((nodeSeen.begin()),(nodeSeen.end()),false);
        //the indexMap is needed
        VertexIndexMap<T> vIndexMap =get(boost::vertex_index, this->mainGraph);
       // EdgeIndexMap<T>   eIndexMap = get(&EdgeAttribute<T>::index, this->mainGraph);
        nodeSeen[vIndexMap(this->source)] = true;
        nodeSeen[vIndexMap(this->sink)]   = true;
        //std::size_t nComp =  this->components.size();
         VertexDescriptor<T> root;
        //-------- start with the known roots------------------------------------------------------
        for (std::size_t i_com = 0; i_com < this->rootVertex.size(); i_com++)
        {
            //std::cout << "Component  " << i_com << "of size " << this->components[i_com].size();
            root = this->rootVertex[i_com];
            if (this->saturatedComponent[i_com])
            {
                //std::cout << " is saturated" << std::endl;
                for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
                     i_ver != this->components[i_com].end(); ++i_ver)
                {
                    nodeSeen[vIndexMap(*i_ver)] = true;
                }
            }
            else
            {
                connected_comp_from_root(root, heapComponent, nodeSeen , edgeSeen);
                //std::cout << " now a " << heapComponent.size() << " vertices component" << std::endl;
                this->components[i_com] = heapComponent;
            }
        }
        VertexIterator<T> i_root, i_root_end;
        for (i_root = boost::vertices(this->mainGraph).first;
             i_root != this->lastIterator; ++i_root)
        {
            if (nodeSeen[vIndexMap(*i_root)])
            {
                 continue;
            }
            root = *i_root;
            connected_comp_from_root(root, heapComponent, nodeSeen , edgeSeen);
            this->components.push_back(heapComponent);
            this->rootVertex.push_back(heapComponent[0]);
            this->saturatedComponent.push_back(false);
        }
        this->components.shrink_to_fit();
    }
    //=============================================================================================
    //==============================  CONNECTED_COMP_FROM_ROOT=========================================
    //=============================================================================================
    inline void connected_comp_from_root(VertexDescriptor<T> & root, std::vector<VertexDescriptor<T>> & heapComponent
                , std::vector<bool> & nodeSeen , std::vector<bool> & edgeSeen)
    {
        //this function compute the connected component of the graph with active edges removed
        // associated with the root ROOT
        EdgeAttributeMap<T> edge_attribute_map
                                    = boost::get(boost::edge_bundle, this->mainGraph);
        VertexIndexMap<T> vIndexMap = get(boost::vertex_index, this->mainGraph);
        EdgeIndexMap<T>   eIndexMap = get(&EdgeAttribute<T>::index, this->mainGraph);
        VertexDescriptor<T> currentNode;
        EdgeDescriptor      currentEdge, reverseEdge;
        std::vector<VertexDescriptor<T>> heapExplore; //the vertices in the current connected component
        heapExplore.reserve(this->nVertex);
        //the boolean vector indicating wether or not the edges and vertices have been seen already
        heapExplore.push_back(root);
        heapComponent.clear();
        while (heapExplore.size()>0)
        {
            currentNode = heapExplore.back();
            heapExplore.pop_back();
            if (nodeSeen[vIndexMap(currentNode)])
            {   //this node has already been treated
                continue;
            }
            heapComponent.push_back(currentNode);
            nodeSeen[vIndexMap(currentNode)] = true ;
            typename boost::graph_traits<Graph<T>>::out_edge_iterator i_edg, i_edg_end;
            for (boost::tie(i_edg,i_edg_end) = boost::out_edges(currentNode, this->mainGraph);
                i_edg !=  i_edg_end; ++i_edg)
                {
                    currentEdge = *i_edg;
                    if (edge_attribute_map(*i_edg).isActive)
                    {
                        continue;
                    }
                    reverseEdge = edge_attribute_map(currentEdge).reverseEdge;
                    if (edgeSeen[eIndexMap(currentEdge)]||edgeSeen[eIndexMap(reverseEdge)])
                    {
                        continue;
                    }
                    edgeSeen[eIndexMap(currentEdge)] = true;
                    edgeSeen[eIndexMap(reverseEdge)] = true;
                    heapExplore.push_back(boost::target(currentEdge, this->mainGraph));
               }
            }
            //std::cout << "  size  = " << compSize << std::endl;
            heapComponent.shrink_to_fit();
    }
    //=============================================================================================
    //================================  COMPUTE_REDUCE_GRAPH   ====================================
    //=============================================================================================
    void compute_reduced_graph(int count)
    {
        EdgeAttributeMap<T> edge_attribute_map
                                    = boost::get(boost::edge_bundle, this->mainGraph);
        VertexAttributeMap<T> vertex_attribute_map
                                    = boost::get(boost::vertex_bundle, this->mainGraph);
        this->reducedGraph = Graph<T>(this->components.size());
        VertexAttributeMap<T> reduced_vertex_attribute_map = boost::get(boost::vertex_bundle, this->reducedGraph);
        //------compute the value of each connected component  and set the values and weight of the main
        // Grapĥ as well as the reduc ed graph-----------------------------------------------------
        for (std::size_t i_com = 0;  i_com < this->components.size(); i_com++)
        {
            //std::cout << "=>" << i_com << " / " <<  this->components.size() << " = " << this->components[i_com].size()<< std::endl;
            std::pair<std::vector<T>, T> component_values = this->compute_value(i_com);
            //----fill the value and weight field of the reduced graph-----------------------------
            VertexDescriptor<T> reduced_vertex = boost::vertex(i_com, this->reducedGraph);
            reduced_vertex_attribute_map[reduced_vertex] = VertexAttribute<T>(this->dim);
            reduced_vertex_attribute_map(reduced_vertex).weight
                    = component_values.second;
            for(int i_dim=0; i_dim<this->dim; i_dim++)
            {
                reduced_vertex_attribute_map(reduced_vertex).value[i_dim]
                        = component_values.first[i_dim];
            }
        }
        //------compute the edges of the reduced graph
        EdgeAttributeMap<T> reduced_edge_attribute_map = boost::get(boost::edge_bundle, this->reducedGraph);
        this->borders.clear();
        EdgeDescriptor currentEdge, currentEdgeReduced;
        int compSource, compTarget, comp1,comp2, indexBorder;
        std::size_t indexEdgeReduced = 0;
        VertexDescriptor<T> vertexSourceReduced, vertexTargetReduced;
        bool reducedEdgeExists;
        typename boost::graph_traits<Graph<T>>::edge_iterator i_edg, i_edg_end;
        std::cout << boost::num_edges(this->mainGraph)<< std::endl;
        for (boost::tie(i_edg,i_edg_end) = boost::edges(this->mainGraph); i_edg !=  i_edg_end; ++i_edg)
        {
            if (!edge_attribute_map(*i_edg).realEdge)
            {
                continue;
            }
            currentEdge = *i_edg;
            comp1       = vertex_attribute_map(boost::source(currentEdge, this->mainGraph)).inComponent;
            comp2       = vertex_attribute_map(boost::target(currentEdge, this->mainGraph)).inComponent;
            //std::cout <<  comp1 << " , " << comp2 << "  added" << std::endl;
            if (comp1==comp2)
            {
                continue;
            }
            compSource  = std::min(comp1,comp2);
            compTarget  = std::max(comp1,comp2);
            vertexSourceReduced = boost::vertex(compSource, this->reducedGraph);
            vertexTargetReduced = boost::vertex(compTarget, this->reducedGraph);
            boost::tie(currentEdgeReduced, reducedEdgeExists)
                    = boost::edge(vertexSourceReduced, vertexTargetReduced, this->reducedGraph);
            if (!reducedEdgeExists)
            {
                currentEdgeReduced = boost::add_edge(vertexSourceReduced, vertexTargetReduced, this->reducedGraph).first;
                //std::cout <<  compSource<< " , " << compTarget << "  added" << boost::num_edges(this->reducedGraph) << std::endl;
                reduced_edge_attribute_map(currentEdgeReduced).index  = indexEdgeReduced;
                reduced_edge_attribute_map(currentEdgeReduced).weight = 0;
                indexEdgeReduced++;
                std::vector<EdgeDescriptor> newBorder;
                this->borders.push_back(newBorder);
            }
            if (count==0)
            {
                reduced_edge_attribute_map(currentEdgeReduced).weight += 0.5*edge_attribute_map(currentEdge).weight;
            }
            {
                reduced_edge_attribute_map(currentEdgeReduced).weight += 0.5;
            }
            indexBorder = reduced_edge_attribute_map(currentEdgeReduced).index;
            this->borders[indexBorder].push_back(currentEdge);
        }
    }
    //=============================================================================================
    //================================          MERGE          ====================================
    //=============================================================================================
    void merge()
    {
        //std::cout << "MERGEING STEP" << std::endl;
        //check wether the energy can be decreased by removing edges from the reduced graph
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->mainGraph);
        VertexAttributeMap<T> reduced_vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->reducedGraph);
        EdgeAttributeMap<T> reduced_edge_attribute_map
                = boost::get(boost::edge_bundle, this->reducedGraph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->mainGraph);
        VertexIndexMap<T> vIndexMapReduced = boost::get(boost::vertex_index, this->reducedGraph);
        EdgeDescriptor currentEdgeReduced;
        typename boost::graph_traits<Graph<T>>::edge_iterator i_edg, i_edg_end;
        typename std::vector<EdgeDescriptor>::iterator i_edg_red;
        VertexDescriptor<T> vertexSourceReduced, vertexTargetReduced, vertexReduced;
        std::size_t compSourceIndex, compTargetIndex, currentEdgeReducedIndex;
        std::vector<T> currentGain(boost::num_edges(this->reducedGraph));
        //std::map<std::pair<std::size_t, std::size_t>, T> gainMap;
        std::priority_queue<Orderedpair<T>, std::vector<Orderedpair<T>>, lessOrderedPair<T>> gainQueue;
        T gain; // the gain obtained by removing the border corresponding to the edge in the reduced graph
        for (boost::tie(i_edg,i_edg_end) = boost::edges(this->reducedGraph); i_edg !=  i_edg_end; ++i_edg)
        {
            //a first pass go trhough all the edges in the reduced graph and comlpute the gain obtained by
            //mergeing the corresponding nodes
            currentEdgeReduced      = *i_edg;
            currentEdgeReducedIndex = reduced_edge_attribute_map(currentEdgeReduced).index;
            vertexSourceReduced = boost::source(currentEdgeReduced, this->reducedGraph);
            vertexTargetReduced = boost::target(currentEdgeReduced, this->reducedGraph);
            //std::cout << currentEdgeReducedIndex << " : (" << vertexSourceReduced << ", " << vertexTargetReduced << ")" << std::endl;
            //std::cout << reduced_edge_attribute_map(currentEdgeReduced).weight
            //<< " : (" << reduced_vertex_attribute_map(vertexSourceReduced).weight
            //<< " , "  << reduced_vertex_attribute_map(vertexTargetReduced).weight << ")" << std::endl;
            gain    = reduced_edge_attribute_map(currentEdgeReduced).weight
                    * this->parameter.reg_strenth;
            std::pair<std::vector<T>, T> merge_gain = compute_merge_gain(vertexSourceReduced, vertexTargetReduced);
            // compute the value obtained by mergeing the two connected components
            gain = gain + merge_gain.second;
            compSourceIndex = vIndexMapReduced(vertexSourceReduced);
            compTargetIndex = vIndexMapReduced(vertexTargetReduced);
            Orderedpair<T> compPair(compSourceIndex, compTargetIndex, currentEdgeReducedIndex, gain);
            compPair.mergedValue = merge_gain.first;
//            std::cout << gain << " : " << reduced_edge_attribute_map(currentEdgeReduced).weight
//                         * this->parameter.reg_strenth << " + "
//                         << merge_gain.second << std::endl;
            if (gain>0)
            {
                gainQueue.push(compPair);
                currentGain.at(currentEdgeReducedIndex) = gain;
            }
        }
        std::vector<bool> isMerged(this->components.size());
        std::fill(isMerged.begin(), isMerged.end(), false);
        std::vector<bool> toDestroy(this->components.size());
        std::fill(toDestroy.begin(), toDestroy.end(), false);
        while(gainQueue.size()>0)
        {   //loop through the optential mergeing and accept the ones that decrease the energy
            Orderedpair<T> compPair = gainQueue.top();
            if (compPair.value<=0)
            {
                break;
            }
            gainQueue.pop();
            if (isMerged.at(compPair.comp1) || isMerged.at(compPair.comp2))
            {
                //at least one of the vertices have already been merged
                continue;
            }
            //std::cout << "mergeing comp " << compPair.comp1 << " and " << compPair.comp2 <<  " for a gain of "
            //          << compPair.value << std::endl;
            //std::cout << "(" <<
            this->components[compPair.comp1].insert( this->components[compPair.comp1].end()
                ,components[compPair.comp2].begin(), this->components[compPair.comp2].end());
            this->saturatedComponent[compPair.comp1] = false;
            reduced_vertex_attribute_map(compPair.comp1).weight
                           += reduced_vertex_attribute_map(vertexTargetReduced).weight;
            reduced_vertex_attribute_map(compPair.comp1).value  = compPair.mergedValue;
            for (i_edg_red = this->borders.at(compPair.index).begin();
                i_edg_red != this->borders.at(compPair.index).end() ; ++i_edg_red)
            {
                 edge_attribute_map(*i_edg_red).isActive = false;
            }

            isMerged.at(compPair.comp1)  = true;
            isMerged.at(compPair.comp2)  = true;
            toDestroy.at(compPair.comp2) = true;
        }
        //we now rebuild the vectors components, rootComponents and saturatedComponent
        std::vector<std::vector<VertexDescriptor<T>>> componentsNew;
        std::vector<VertexDescriptor<T>> rootVertexNew; //a root for each connected components
        std::vector<bool> saturatedComponentNew;
        for (std::size_t i_com = 0; i_com < this->components.size(); i_com++)
        {
            if (toDestroy.at(i_com))
            {
                continue;
            }
            componentsNew.push_back(this->components.at(i_com));
            rootVertexNew.push_back(this->rootVertex.at(i_com));
            saturatedComponentNew.push_back(saturatedComponent.at(i_com));
            vertexReduced = boost::vertex(i_com, this->reducedGraph);
            if (isMerged.at(i_com))
            {
                for (VertexComponentIterator<T> i_ver = this->components[i_com].begin();
                     i_ver != this->components[i_com].end(); ++i_ver)
                {
                    vertex_attribute_map(*i_ver).value       = reduced_vertex_attribute_map(vertexReduced).value;
                    vertex_attribute_map(*i_ver).inComponent = i_com;
                }
            }
        }
        this->components         = componentsNew;
        this->rootVertex         = rootVertexNew;
        this->saturatedComponent = saturatedComponentNew;
    }
//===============================================================================================
//==========================saturateComponent====================================================
//===============================================================================================
    inline void saturateComponent(std::size_t i_com)
    {
        EdgeAttributeMap<T> edge_attribute_map
                                    = boost::get(boost::edge_bundle, this->mainGraph);
        //std::cout << "SATURATING " << i_com << " = " << this->components[i_com].size() << std::endl;
        this->saturatedComponent[i_com] = true;
        for (int i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
        {
            VertexDescriptor<T> desc_v = this->components[i_com][i_ver];
            // because of the adjacency structure NEVER access edge (source,v) directly!
            EdgeDescriptor desc_v2source = boost::edge(desc_v, this->source,this->mainGraph).first;
            EdgeDescriptor desc_source2v = edge_attribute_map(desc_v2source).reverseEdge; //use reverseEdge instead
            EdgeDescriptor desc_v2sink   = boost::edge(desc_v, this->sink,this->mainGraph).first;
            // we set the capcities to zero
            edge_attribute_map(desc_source2v).capacity = 0;
            edge_attribute_map(desc_v2sink  ).capacity = 0.;
        }
    }
};
}
