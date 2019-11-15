#include <iostream>
#include <cstdio>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <../include/API.h>
//#include <../include/connected_components.h>

namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Custom_tuple;

struct VecToArray
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};

struct VecToArray_float
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<float> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(float));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

struct to_py_tuple
{//converts output to a python tuple
    static PyObject* convert(const Custom_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

struct to_py_tuple_list
{//converts output to a python list of tuples
    static PyObject* convert(const std::vector <Custom_tuple > & c_tuple_vec){
        int n_hierarchy = c_tuple_vec.size();
        bp::list values;
        //add all c_tuple items to "values" list
        for (int i_hierarchy = 0; i_hierarchy < n_hierarchy; i_hierarchy++)
        {
            PyObject * tuple_pyo = to_py_tuple::convert(c_tuple_vec[i_hierarchy]);
            values.append(bp::handle<>(bp::borrowed(tuple_pyo)));
        }
        return bp::incref(values.ptr());
    }
};

PyObject * cutpursuit(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,
                      float lambda, const int cutoff, const int spatial, float weight_decay)
{//read data and run the L0-cut pursuit partition algorithm
    srand(0);

    const uint32_t n_ver = bp::len(obs);
    const uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    std::vector<float> solution(n_ver *n_obs);
    //float solution [n_ver * n_obs];
    std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    if (spatial == 0)
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  1.f, 4.f, weight_decay, 0.f);
    }
    else
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  2.f, 4.f, weight_decay, 0.f);
    }
    return to_py_tuple::convert(Custom_tuple(components, in_component));
}

PyObject * cutpursuit_hierarchy(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target, const bpn::ndarray & edge_weight,
                      const bpn::ndarray & lambda, const bpn::ndarray & cutoff, const int spatial, float weight_decay)
{//read data and run the L0-cut pursuit partition algorithm
    srand(0);
    uint32_t n_ver = bp::len(obs);
    uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const uint32_t n_hierarchy = bp::len(lambda);
    float * obs_data = reinterpret_cast<float*>(obs.get_data());
    uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    const float * lambda_data = reinterpret_cast<float*>(lambda.get_data());
    const uint32_t * cutoff_data = reinterpret_cast<uint32_t*>(cutoff.get_data());
    std::vector<float> node_weight(n_ver, 1.0f);
    //prepare outputs
    std::vector<float> solution(n_ver *n_obs);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    //prepare hierarchical intermediaries
    uint32_t n_nodes_red;
    uint32_t n_edges_red;
    std::vector<uint32_t> Eu_red(1,0);
    std::vector<uint32_t> Ev_red(1,0);
    std::vector< float > edgeWeight_red(1,0);
    std::vector< float > nodeWeight_red(1,0);
    //prepare list of outputs
    std::vector< Custom_tuple > output_hierarchy(n_hierarchy);
    for (int ite_hierarchy = 0; ite_hierarchy < n_hierarchy; ite_hierarchy++)
    {
        if (spatial == 0)
        {
            CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                     , solution.data(), in_component, components
                     , n_nodes_red, n_edges_red, Eu_red, Ev_red, edgeWeight_red, nodeWeight_red
                     , lambda_data[ite_hierarchy], (uint32_t)cutoff_data[ite_hierarchy],  1.f, 4.f, weight_decay, 0.f);
        }
        else
        {
            CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                     , solution.data(), in_component, components
                     , n_nodes_red, n_edges_red, Eu_red, Ev_red, edgeWeight_red, nodeWeight_red
                     , lambda_data[ite_hierarchy], (uint32_t)cutoff_data[ite_hierarchy],  2.f, 4.f, weight_decay, 0.f);
        }
        n_ver = n_nodes_red;
        n_edg = n_edges_red;
        source_data = Eu_red.data();
        target_data = Ev_red.data();
        edge_weight_data = edgeWeight_red.data();
        node_weight = nodeWeight_red;
        uint32_t ind_sol = 0;
        int index = 0;
        for(uint32_t ind_comp = 0; ind_comp < n_ver; ind_comp++ )
        {
            for(uint32_t i_obs=0; i_obs < n_obs; i_obs++)
            {
                obs_data[index] = solution[components[ind_comp][0] * n_obs + i_obs];
                index++;
            }
        }
        solution = std::vector<float> (n_ver * n_obs,0.);
        output_hierarchy[ite_hierarchy] = Custom_tuple(components, in_component);
    }

    return to_py_tuple_list::convert(output_hierarchy);
}

PyObject * cutpursuit2(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,
                       const bpn::ndarray & node_weight, float lambda)
{//read data and run the L0-cut pursuit partition algorithm
    srand(0);
    const uint32_t n_ver = bp::len(obs);
    const uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    const float * node_weight_data = reinterpret_cast<float*>(node_weight.get_data());
    std::vector<float> solution(n_ver *n_obs);
    //float solution [n_ver * n_obs];
    //std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, node_weight_data
                 , solution.data(), in_component, components, lambda, (uint32_t)0,  2.f, 4.f, 1.f, 1.f);
    return to_py_tuple::convert(Custom_tuple(components, in_component));
}

BOOST_PYTHON_MODULE(libcp)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();
    
    def("cutpursuit", cutpursuit);
    def("cutpursuit", cutpursuit, (bp::args("cutoff")=0, bp::args("spatial")=0, bp::args("weight_decay")=1));
    def("cutpursuit_hierarchy", cutpursuit_hierarchy, (bp::args("cutoff")=0, bp::args("spatial")=0, bp::args("weight_decay")=1));
    def("cutpursuit2", cutpursuit2);
}

