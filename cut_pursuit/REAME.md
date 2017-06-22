# Cut Pursuit: minimizing convex nonsmooth functionals regularized by graph total variation

A working set strategy for minizing a convex nonsmooth function on a weighted graph G=(V,E,w) when regularized by the total variation:
x^* = \argmin_{x \in \Omega^V} f(x) + \sum_{u,v}w_{u,v} ||x_u - x_v||

with \Omega the space of the values associated to each node. Current implementation only support \Omega = R

f must be convex, and its non-smooth part g must be separable over V: g(x) = \sum_{v \in V} g_v(x_v).
Current implementation supports f under the form:

f(x) = ||A x - y ||^2 + \sum_{v \in V}\mu_v |x| + \iota(x > 0)

with A a linear operator, \mu_v a vector of l1 weights and iota the indicator function

in particular, this allows to solve inverse problems regularized by the total variation or the fused LASSO penalty.
