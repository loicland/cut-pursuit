#include <iostream>
#include <cstdio>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include <../include/API.h>

namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

typedef std::pair< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Custom_pair;

struct PairToList
{
    static PyObject* convert(const Custom_pair & c_pair)
    {
        boost::python::list* l1 = new boost::python::list();
        for(size_t i = 0; i < c_pair.first.size(); i++)
        {
            boost::python::list* l2 = new boost::python::list();
            for(size_t j = 0; j < c_pair.first.at(i).size(); j++)
            {
                l2->append((uint32_t) c_pair.first.at(i).at(j));
            }
            l1->append((l2, l2[0]));
        }
        boost::python::list* l3 = new boost::python::list();
        for(size_t i = 0; i < c_pair.second.size(); i++) {
            l3->append((uint32_t) c_pair.second[i]);
        }
        boost::python::list* l4 = new boost::python::list();
        l4->append((l1, l1[0]));
        l4->append((l3, l3[0]));
        return l4->ptr();
    }
};

PyObject * cutpursuit(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,  float lambda)
{
    const uint32_t n_ver = boost::python::len(obs);
    const uint32_t n_edg = boost::python::len(source);
    const uint32_t n_obs = boost::python::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    //std::vector< std::vector<float> > solution(n_ver, std::vector<float>(n_obs));
    float solution [n_ver * n_obs];
    float node_weight[n_ver];
    for (uint32_t i_ver = 0; i_ver < n_ver; i_ver++)
    {
        node_weight[i_ver] = 1.f;
    }
    //memset(node_weight, 1.f, n_ver);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0));
    CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, node_weight
              , solution, in_component, components, lambda, 1.f, 2.f, 2.f);
    return PairToList::convert(Custom_pair(components, in_component));
}
BOOST_PYTHON_MODULE(libcp)
{
    _import_array();
    Py_Initialize();
    boost::python::numpy::initialize();
    //bp::to_python_converter<std::vector<uint32_t, std::allocator<uint32_t> >, VecToArray<uint32_t> >();
    //to_python_converter<std::vector<std::vector<uint32_t>, std::allocator<std::vector<uint32_t> > >, VecvecToList<uint32_t> >();
    //bp::to_python_converter<std::vector<float, std::allocator<float> >, VecToArray<float> >();
    //to_python_converter<std::vector<int, std::allocator<int> >, VecToList<int> >();
    //to_python_converter<std::vector<std::vector<int>, std::allocator<std::vector<int> > >, VecvecToList<int> >();
    //to_python_converter<PyObject *, make_pair<PyObject *, PyObject *> >();
    bp::to_python_converter<Custom_pair, PairToList >();
    def("cutpursuit", cutpursuit);
}

