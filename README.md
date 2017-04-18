# cut-pursuit
C++ implementation for the cut-pursuit algorithm, with Matlab/python interfaces 
Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
Landrieu, Loic and Obozinski, Guillaume,2016.

A working-set / greedy approach to solve optimization problems structured by a graph G=(V,E,w), with w the edge weight:
min_{x \in \Omega} f(x)+\sum_{(i,j)}{w_{i,j} g(\abs{x_i - x_j})}
\Omega being the search space
f the fidelity function
g a graph structured regularizer (g = Id for TV, g = 1-kronecker for potts penalty)
