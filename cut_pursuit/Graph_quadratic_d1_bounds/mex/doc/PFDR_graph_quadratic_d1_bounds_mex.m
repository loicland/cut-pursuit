function [X, it, Obj, Dif] = PFDR_graph_quadratic_d1_bounds_mex(Y, A, Eu, Ev, La_d1, Bnd, L, rho, condMin, difRcd, difTol, itMax, verbose)
%
%        [X, it, Obj, Dif] = PFDR_graph_quadratic_d1_bounds_mex(Y, A, Eu, Ev, La_d1, Bnd, L, rho, condMin, difRcd, difTol, itMax, verbose)
%
% minimize functional over a graph G = (V, E)
%
%       F(x) = 1/2 ||y - x||_{l2,La_l2}^2 + ||x||_{d1,La_d1}  + i_{[m, M]}(x)
%
% where x, y in R^V,
%      ||x||_{l2,La_l2}^2 = sum_{v in V} la_l2_v (y_v - x_v)^2,
%      ||x||_{d1,La_d1} = sum_{uv in E} la_d1_uv |x_u - x_v|,
%      i_{[m, M]}(x) = 0          if for all v in V, m <= x_v <= M,
%                      +infinity  otherwise
%
% using preconditioned forward-Douglas-Rachford splitting algorithm.
%
% INPUTS: (warning: real numeric type is either single or double, not both)
% Y          - observations, array of length N (real)
% A          - matrix, N-by-V array (real)
% Eu         - for each edge, index of one vertex, array of length E (int32)
% Ev         - for each edge, index of the other vertex, array of length E (int32)
%              Every vertex should belong to at least one edge. If it is not the
%              case, a workaround is to add an edge from the vertex to itself
%              with a nonzero penalization coefficient.
% La_d1      - d1 penalization coefficients, array of length E (real)
% Bnd        - lower and upper bounds constraints, array of length 2 (real)
%              set to [-inf inf] for no bounds
% L          - information on Lipschitzianity of the operator A^* A.
%              either a scalar satisfying 0 < L <= ||A^* A||,
%              or a diagonal matrix (array of length V (real)) satisfying
%               0 < L and ||L^(-1/2) A^* A L^(-1/2)|| <= 1
% rho        - relaxation parameter, 0 < rho < 2
%              1 is a conservative value; 1.5 often speeds up convergence
% condMin    - positive parameter ensuring stability of preconditioning
%              0 < condMin =< 1; 1e-3 is a conservative value; 1e-6 might 
%              enhance preconditioning
% difRcd     - reconditioning criterion on iterate evolution. A reconditioning
%              is performed if relative changes of X (in Euclidean norm) is less
%              than difRcd. It is then divided by 10.
%              10*difTol is a conservative value, 1e2*difTol or 1e3*difTol might
%              speed up convergence. reconditioning might temporarily draw 
%              minimizer away from solution; it is advised to monitor objective
%              value when using reconditioning
% difTol     - stopping criterion on iterate evolution. Algorithm stops if
%              relative changes of X (in Euclidean norm) is less than difTol.
% itMax      - maximum number of iterations
% verbose    - if nonzero, display information on the progress, every 'verbose'
%              iterations
% OUTPUTS:
% X   - final minimizer, array of length V (real)
% it  - actual number of iterations performed
% Obj - if requested, the values of the objective functional 
%       along iterations (it+1 values)
% Dif - if requested, the iterate evolution along iterations (see difTol)
% 
% Parallel implementation with OpenMP API.
%
% Typical compilation command (UNIX):
% mex CXXFLAGS="\$CXXFLAGS -DMEX -fopenmp -DNDEBUG" ...
%     LDFLAGS="\$LDFLAGS -fopenmp" ...
%     api/PFDR_graph_quadratic_d1_bounds_mex.cpp ...
%     src/PFDR_graph_quadratic_d1_bounds.cpp ...
%     -output bin/PFDR_graph_quadratic_d1_bounds_mex
%
% Reference: H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for
% Monotone Inclusion and Convex Optimization, to appear.
%
% Hugo Raguet 2016
