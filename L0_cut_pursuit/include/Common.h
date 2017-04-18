#pragma once
#include <string>
#include <sstream>
#include <ctime>
#include <functional>
#include<stdio.h>


#ifndef COMMON_H
#define COMMON_H

#endif // COMMON_H

namespace patch
{
template < typename T > std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}
}

enum fidelityType {L2, linear, KL, loglinear};

typedef std::pair<std::string, float> NameScale_t;

class GenericParameter
{
    public:
    std::string in_name, out_name, base_name, extension;
    int natureOfData;
    fidelityType fidelity;
    //std::vector< NameScale_t > v_coord_name_scale, v_attrib_name_scale;
    GenericParameter(std::string inName = "in_name", double reg_strength = 0, double fidelity = 0)
    {
        this->in_name  = inName;
        char buffer [inName.size() + 10];
        std::string extension = inName.substr(inName.find_last_of(".") + 1);
        this->extension  = extension;
        std::string baseName  = inName.substr(0, inName.size() - extension.size() - 1);
        this->base_name  = baseName;
        sprintf(buffer, "%s_out_%1.0f_%.0f.%s",baseName.c_str(),fidelity,reg_strength*1000, extension.c_str());
        this->out_name = std::string(buffer);
        this->natureOfData = 0;
        this->fidelity = L2;
    }
    virtual ~GenericParameter() {}
};

class TimeStack
{
    clock_t lastTime;
public:
    TimeStack(){}
    void tic() {
        this->lastTime = clock();
    }

    std::string toc() {
        std::ostringstream stm ;
        stm << ((double)(clock() - this->lastTime)) / CLOCKS_PER_SEC;
        return stm.str();
    }

double tocDouble() {
      std::ostringstream stm ;
      double x =  ((double)(clock() - this->lastTime)) / CLOCKS_PER_SEC;
      return x;
}
};

template<typename T>
class Orderedpair
{
public:
    std::size_t comp1, comp2, index;
    T value;
    std::vector<T> mergedValue;

    Orderedpair(std::size_t c1, std::size_t c2, std::size_t ind = 0, T val = 0.)
    {
        this->comp1 = c1;
        this->comp2 = c2;
        this->index = ind;
        this->value = val;
    }
};

template<typename T>
struct lessOrderedPair: public std::binary_function<Orderedpair<T>, Orderedpair<T>, bool>
{
    bool operator()(const Orderedpair<T> lhs, const Orderedpair<T> rhs) const
    {
        return lhs.value < rhs.value;
    }
};


template<typename T>
class Point3D
{
public:
    T x,y,z;
    Point3D(T x = 0., T y = 0., T z = 0.)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

template<typename T>
struct lessPoint3D: public std::binary_function<Point3D<T>, Point3D<T>, bool>
{
    bool operator()(const Point3D<T> lhs, const Point3D<T> rhs) const
    {
        if (lhs.x != rhs.x)
        {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y)
        {
            return lhs.y < rhs.y;
        }
        if (lhs.z > rhs.z)
        {
            return lhs.z < rhs.z;
        }
        return true;
    }
};
