/*==================================================================
 * minimize functional over a graph G = (V, E)
 *
 *        F(x) = 1/2 ||y - A x||^2 + ||x||_{d1,La_d1} + ||x||_{l1,La_l1}
 *
 * where y in R^N, x in R^V, A in R^{N-by-|V|}
 *       ||x||_{d1,La_d1} = sum_{uv in E} la_d1_uv |x_u - x_v|,
 *       ||x||_{l1,La_l1} = sum_{v  in V} la_l1_v |x_v|,
 *
 * with the possibility of adding a positivity constraint on the coordinates of x,
 *        F(x) + i_{x >= 0}
 *
 * using preconditioned forward-Douglas-Rachford splitting algorithm.
 *
 * It is easy to introduce a SDP metric weighting the squared l2-norm
 * between y and A x. Indeed, if M is the matrix of such a SDP metric,
 *   ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
 * Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
 * Moreover, when A is the identity and M is diagonal (weighted l2 distance
 * between x and y), one should call on the precomposed version 
 * (N set to zero, see below) with Y <- DDy = My and A <- D2 = M.
 *
 * Parallel implementation with OpenMP API.
 *
 * Reference: H. Raguet,  A Note on the Forward-Douglas-Rachford Splitting for
 * Monotone Inclusion and Convex Optimization, to appear.
 *
 * Hugo Raguet 2016
 *================================================================*/
#ifndef PFDR_GRAPH_QUADRATIC_D1_L1_H
#define PFDR_GRAPH_QUADRATIC_D1_L1_H

typedef enum {SCAL, DIAG} Lipschtype;

template <typename real>
void PFDR_graph_quadratic_d1_l1(const int V, const int E, const int N, \
    real *X, const real *Y, const real *A, const int *Eu, const int *Ev, \
    const real *La_d1, const real *La_l1, const int positivity, \
    const Lipschtype Ltype, const real *L, const real rho, const real condMin, \
    real difRcd, const real difTol, const int itMax, int *it, \
    real *Obj, real *Dif, const int verbose);
/* 22 arguments:
 * V, E       - number of vertices and of (undirected) edges
 * N          - number of observations.
 *              if nonpositive, matricial information is precomputed,
 *              that is, argument A is (A^t A) and argument Y is (A^t Y)
 *              if exactly zero, A is a diagonal matrix and only the diagonal
 *              of (A^t A) = A^2 is given
 * X          - minimizer, array of length V
 *              must be initialized (usually all zeros)
 * Y          - if N is positive, observations, array of length N
 *              otherwise, correlation of A with the observations (A^t Y), 
 *              array of length V
 * A          - if N is positive, linear operator explaining observation from 
 *              the coefficients, N-by-V array, column major format
 *              if N is negative, matrix (A^t A), V-by-V array, column major format
 *              if N is zero, diagonal of (A^t A) = A^2, array of length V
 *              set to NULL (and N to zero) for identity matrix
 * Eu         - for each edge, index of one vertex, array of length E
 * Ev         - for each edge, index of the other vertex, array of length E
 *              Every vertex should belong to at least one edge. If it is not the
 *              case, a workaround is to add an edge from the vertex to itself
 *              with a nonzero penalization coefficient.
 * La_d1      - d1 penalization coefficients, array of length E
 * La_l1      - l1 penalization coefficients, array of length V
 *              set to NULL for no l1 penalization
 * positivity - if nonzero, the positivity constraint is added
 * Ltype      - tells if Lipschitz information L is scalar (SCAL, 0)
 *              or diagonal (DIAG, 1)
 * L          - information on Lipschitzianity of the operator A^t A.
 *              if Ltype is SCAL, then it is a scalar satisfying
 *              0 < L <= ||A^t A||,
 *              if Ltype is DIAG, then it represents a diagonal matrix
 *              (array of length V) satisfying
 *              0 < L and ||L^(-1/2) A^t A L^(-1/2)|| <= 1
 * rho        - relaxation parameter, 0 < rho < 2
 *              1 is a conservative value; 1.5 often speeds up convergence
 * condMin    - parameter ensuring stability of preconditioning
 *              0 < condMin =< 1; 1e-3 is a conservative value; 1e-6 might 
 *              enhance preconditioning
 * difRcd     - reconditioning criterion on iterate evolution. A reconditioning
 *              is performed if relative changes of X (in Euclidean norm) is less
 *              than difRcd. It is then divided by 10.
 *              10*difTol is a conservative value, 1e2*difTol or 1e3*difTol might
 *              speed up convergence. reconditioning might temporarily draw 
 *              minimizer away from solution; it is advised to monitor objective
 *              value when using reconditioning
 * difTol     - stopping criterion on iterate evolution. Algorithm stops if
 *              relative changes of X (in Euclidean norm) is less than difTol.
 * itMax      - maximum number of iterations
 * it         - adress of an integer keeping track of iteration number
 * Obj        - if not NULL, records the values of the objective
 *              functional, array of length itMax
 *              if N is nonpositive, objective is computed up to a contant 1/2 ||Y||^2
 * Dif        - if not NULL, records the iterate evolution (see difTol),
 *              array of length itMax
 * verbose    - if nonzero, display information on the progress, every 'verbose'
 *              iterations */
#endif
