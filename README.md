# cut pursuit: a working-set aprpoach to computing piecewise constant functions on graphs
C/C++ implementation fot the cut pursuit algorithms with Matlab interfaces.

Cut pursuit is a strategy to minimize functions regularized by graph-structured regularizers. For G=(V,E,w) a graph with edges weighted by w, the problem writes:

x^* = argmin_ {x \in R^V} f(x) + \sum_{u, v \in E) w_{u,v} \phi(x_u - x_v)

We distinguish two different cases for, \phi, for which the implementation are different:
- cut_pursuit : the convex case, \phi(x) = |x| and the regularizer is the total variation.
- L0_cut_pursuit : the non-convex case, \phi(x) = \delta(x != 0) and the regularizer is the weight of the cut between the adjacent constant components.

# Reference:

Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
Landrieu, Loic and Obozinski, Guillaume,2016.

if using the implementation in the convex case, one must also cite:

`Cut-pursuit` algorithm for convex nonsmooth functionals regularized by graph total variation, H. Raguet and L. Landrieu, in preparation. 
