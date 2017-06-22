# cut pursuit: a working-set strategy to compute piecewise constant functions on graphs
C/C++ implementation fot the cut pursuit algorithms with Matlab interfaces.

Cut pursuit is a graph cut-based working set strategy to minimize functions regularized by graph-structured regularizers. For G=(V,E,w) a graph with edges weighted by w, the problem writes:

x^* = argmin_ {x \in \Omega^V} f(x) + \sum_{u, v \in E) w_{u,v} \phi(x_u - x_v)

with \Omega the space in which the value associated with each node belongs.

We distinguish two different cases  for \phi, for which the implementations are different:
- \phi(t) = |t|  : the convex case,  the regularizer is the total variation. Folder `cut_pursuit`. Code by Hugo Raguet
- \phi(t) = \delta(t != 0) = 1 - kroenecker(t):  the non-convex case, the regularizer is the weight of the cut between the adjacent constant components. Folder `L0_cut_pursuit`. Code by Loic Landrieu

### References:

Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
L. Landrieu and G. Obozinski, 2016.

if using the implementation in the convex case, one must also cite:

Cut-pursuit algorithm for convex nonsmooth functionals regularized by graph total variation, H. Raguet and L. Landrieu, in preparation. 
