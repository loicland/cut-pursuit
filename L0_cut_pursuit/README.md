# L0-cut pursuit

L0 cut pursuit is a greedy algorithm to compute a piecewise constant approximation of a function on a weighted graph G=(V,E,w).
This corresponds to solving the following optimization problem:

x^* = \argmin_{x \in \Omega^V} \sum_{v \in V}f_v(x_v) + \sum_{u,v \in E}w_{u,v} \delta(x_u != x_v)

\Omega being the space of the value associated with each node. Implementation works for \Omega = R, R^V and simplex values
f is the fidelity function. Current implementation supports 
  - quadratic fidelity: f_v(x_v) = ||x_v - y_v||² with y an observed value associated with node v
  - linear fidelity: f_v(x_v) = - <x_v, d_v> with d_v a weight associated with node v
  - Kullback leibler fidelity f_v(x_v) = KL(x_v, p_v) with p_v an probability associated with node v. Only applu whenn \Omega is a simplex

### Reference
if using this algorithm with \Omega other than R, one must also cite:

A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds.
Loic Landrieu, Hugo Raguet , Bruno Vallet , Clément Mallet, Martin Weinmann

### Compilation
MATLAB
```
mkdir ./L0_cut_pursuit/bin
addpath('./L0_Cut_Pursuit/bin/')
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" L0_cut_pursuit/api/L0_cut_pursuit_mex.cpp ...
    -output L0_cut_pursuit/bin/L0_cut_pursuit_mex
```    
C++
```
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -fopenmp -O3 -Wall -std=c++11")'
```

Note: the parallelisation is not yet implmented, so fopenmp can be foregone for now.

 ### Example
 
minimal example on a chain graph.
```
rng(1)
n_nodes = 100;
y = rand(3,n_nodes);
Eu = 0:(n_nodes-2);
Ev = 1:(n_nodes-1);
edge_weight = ones(numel(Eu),1);
node_weight = ones(n_nodes,1);

solution = L0_cut_pursuit_mex(single(y), int32(Eu), int32(Ev), single(.2)...
    , single(edge_weight), single(node_weight), 1, 2, 2);

subplot(2,1,1)
imagesc(repmat(y, [1 1 1]))
subplot(2,1,2)
imagesc(solution)
```
