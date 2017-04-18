# L0 cut-pursuit
C++ implementation for the L0-cut pursuit algorithm, with Matlab/python interfaces, described in:

Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
Landrieu, Loic and Obozinski, Guillaume,2016.

A working-set / greedy approach to solve the (generalized) optimal partition problem structured by a graph G=(V,E,w), with w the edge weight:

argmin_{x \in \Omega} f(x)+\sum_{(i,j)}{w_{i,j} g(\abs{x_i - x_j})}

\Omega being the search space

f the fidelity function

g a graph-structured regularizer (g = Id for TV, g = 1-kroenecker(.) for potts penalty)

To compile the MATLAB mex file for L0-cut pursuit type the following in MATLAB in the workspace containing the L0_Cut_Pursuit folder:

```
mkdir ./L0_Cut_Pursuit/bin
addpath('./L0_Cut_Pursuit/bin/')
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" L0_Cut_Pursuit/api/L0_cut_pursuit_mex.cpp ...
    -output L0_Cut_Pursuit/bin/L0_cut_pursuit_mex
```
