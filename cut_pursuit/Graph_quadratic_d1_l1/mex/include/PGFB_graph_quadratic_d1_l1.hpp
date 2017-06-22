/*==================================================================
 * minimize functional over a graph G = (V, E) of the form:
 *
 *        F(x) = 1/2 ||y - A*x||^2 + ||x||_{d1,La_d1} + ||x||_{l1,La_l1}
 *
 * where ||x||_{d1,La_d1} = sum_{uv in E} la_d1_uv |x_u - x_v|,
 *       ||x||_{l1,La_l1} = sum_{v  in V} la_l1_v |x_v|,
 *
 * with the possibility of adding a positivity constraint on the coordinates of x,
 *        F(x) + i_{x >= 0}
 *
 * using preconditioned forward-Douglas-Rachford splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * Hugo Raguet 2016
 *================================================================*/
#ifndef PFDR_GRAPH_QUADRATIC_D1_L1_H
#define PFDR_GRAPH_QUADRATIC_D1_L1_H

typedef enum {SCAL, DIAG} Lipschtype;

template <typename real>
void PFDR_graph_quadratic_d1_l1(const int V, const int E, const int N, \
    const real *Y, const real *A, const int *Eu, const int *Ev, \
    const real *La_d1, const real *La_l1, const int positivity, \
    real *X, real *Yl1, \
    const Lipschtype Ltype, const real *L, const real rho, const real condMin, \
    real difRcd, const real difTol, const int itMax, int *it, \
    real *Obj, real *Dif, const int verbose);
/* 23 arguments:
 * V, E       - number of vertices and of (undirected) edges
 * N          - number of observations.
 *              if set to zero, matricial information is precomputed,
 *              that is, argument A is (A^t A) and argument Y is (A^t Y)
 * Y          - if N is nonzero, observations, array of length N
 *              if N is set to zero, correlation of A with the observations, array of length V
 * A          - if N is nonzero, linear operator explaining observation from 
 *              the coefficients, N-by-V array, column major format
 *              if N is set to zero, matrix (A^t A) V-by-V array, column major format
 * Eu         - for each edge, index of one vertex, array of length E
 * Ev         - for each edge, index of the other vertex, array of length E
 *              Every vertex should belong to at least one edge. If it is not the
 *              case, a workaround is to add an edge from the vertex to itself
 *              with a nonzero penalization coefficient.
 * La_d1      - d1 penalization coefficients, array of length E
 * La_l1      - l1 penalization coefficients, array of length V
 * positivity - if nonzero, the positivity constraint is added
 * X          - best retrieved coefficients, array of length V
 *              must be initialized (all zeros is good)
 * Yl1        - if not set to NULL, the final subgradient of the l1-norm (and
 *              positive constraint, if any) is given in Yl1
 * Ltype      - tells if Lipschitz information L is scalar (SCAL, 0)
 *              or diagonal (DIAG, 1)
 * L          - information on Lipschitzianity of the operator A^* A.
 *              if Ltype is SCAL, then it is a scalar satisfying
 *              0 < L <= ||A^* A||,
 *              if Ltype is DIAG, then it represents a diagonal matrix
 *              (array of length V) satisfying
 *              0 < L and ||L^(-1/2) A^* A L^(-1/2)|| <= 1
 * rho        - relaxation parameter, 0 < rho < 2
 * condMin    - small positive parameter ensuring stability of preconditioning
 * difRcd     - reconditioning criterion on iterate evolution. A reconditioning
 *              is performed if relative changes of X (in Euclidean norm) is less
 *              than difRcd. It is then divided by 10.
 * difTol     - stopping criterion on iterate evolution. Algorithm stops if
 *              relative changes of X (in Euclidean norm) is less than difTol.
 * itMax      - maximum number of iterations
 * it         - adress of an integer keeping track of iteration number
 * objRec     - if objRec is nonzero and N is nonzero, the values of the objective
 *              functional is recorded in Obj
 * Obj        - if not NULL and N is nonzero, the records the values of the objective
 *              functional, array of length itMax
 * Dif        - if not NULL, records the iterate evolution (see difTol),
 *              array of length itMax 
 * verbose    - if nonzero, display information on the progress, every 'verbose'
 *              iterations */
#endif
