function [Cv, rX, CP_it, Time, Obj, Dif] = CP_PFDR_graph_quadratic_d1_bounds_AtA_mex(AtY, AtA, Eu, Ev, La_d1, Bnd, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
%
%        [Cv, rX, CP_it, Time, Obj, Dif] = CP_PFDR_graph_quadratic_d1_bounds_AtA_mex(AtY, AtA, Eu, Ev, La_d1, Bnd, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
% 
% minimize functional over a graph G = (V, E)
%
%       F(x) = 1/2 ||y - A x||^2 + ||x||_{d1,La_d1}  + i_{[m, M]}(x)
%
% where y in R^N, x in R^V, A in R^{N-by-|V|}
%      ||x||_{d1,La_d1} = sum_{uv in E} la_d1_uv |x_u - x_v|,
%      i_{[m, M]}(x) = 0          if for all v in V, m <= x_v <= M,
%                      +infinity  otherwise
% 
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm, with premultiplication by A^t (see INPUTS).
% 
% INPUTS: (warning: real numeric type is either single or double, not both)
% AtY        - correlation of A with the observations (A^t Y),
%              array of length V (real)
% AtA        - matrix (A^t A) V-by-V array (real)
% Eu         - for each edge, index of one vertex, array of length E (int32)
% Ev         - for each edge, index of the other vertex, array of length E (int32)
% La_d1      - d1 penalization coefficients, array of length E (real)
% Bnd        - lower and upper bounds constraints, array of length 2 (real)
%              set to [-inf inf] for no bounds
% 
% [CP]
% difTol     - stopping criterion on iterate evolution. Algorithm stops if
%              relative changes of X (in Euclidean norm) is less than difTol.
%              1e-2 is a conservative value; 1e-3 or less can give better
%              precision but with longer computational time
% itMax      - maximum number of iterations (graph cut and subproblem)
%              10 cuts solve accurately most problems
% 
% [PFDR]
% rho        - relaxation parameter, 0 < rho < 2
%              1 is a conservative value; 1.5 often speeds up convergence
% condMin    - small positive parameter ensuring stability of preconditioning
%              0 < condMin =< 1; 1e-3 is a conservative value; 1e-6 might 
%              enhance preconditioning
% difRcd     - reconditioning criterion on iterate evolution. A reconditioning
%              is performed if relative changes of X (in Euclidean norm) is less
%              than difRcd. It is then divided by 10.
%              0 (no reconditioning) is a conservative value, 10*difTol or 
%              1e2*difTol might speed up convergence. reconditioning might 
%              temporarily draw minimizer away from solution, and give bad
%              subproblem solution
% difTol     - stopping criterion on iterate evolution. Algorithm stops if
%              relative changes of X (in Euclidean norm) is less than difTol.
%              1e-3*CP_difTol is a conservative value.
% itMax      - maximum number of iterations
%              1e4 iterations provides enough precision for most subproblems
% 
% verbose    - if nonzero, display information on the progress, every 'verbose'
%              iterations
% OUTPUTS:
% Cv    - assignement of each vertex of the minimizer to an homogeneous connected
%         component of the graph, numeroted from 0 to (rV - 1)
%         array of length V (int)
% rX    - values of each homogeneous connected components of the minimizer, 
%         array of length rV (real)
%         The actual minimizer is then reconstructed as X = rX(Cv+1);
% CP_it - actual number of iterations (cuts) performed
% Time  - if requested, the elapsed time along iterations (itMax + 1 values)
% Obj   - if requested, the values of the objective functional along 
%         iterations (itMax + 1 values)
% Dif   - if requested, the iterate evolution along iterations (see difTol)
%
% Parallel implementation with OpenMP API.
%
% Typical compilation command (UNIX):
% mex CXXFLAGS="\$CXXFLAGS -DMEX -fopenmp -DNDEBUG" ...
%     LDFLAGS="\$LDFLAGS -fopenmp" ...
%     api/CP_PFDR_graph_quadratic_d1_bounds_AtA_mex.cpp ...
%     src/CP_PFDR_graph_quadratic_d1_bounds.cpp ...
%     src/PFDR_graph_quadratic_d1_bounds.cpp ...
%     src/graph.cpp src/maxflow.cpp src/operator_norm_matrix.cpp ...
%     -output bin/CP_PFDR_graph_quadratic_d1_bounds_AtA_mex
%
% Reference: H. Raguet and L. Landrieu, `Cut-pursuit` algorithm for convex
% nonsmooth functionals regularized by graph total variation, in preparation.
%
% Hugo Raguet 2016
