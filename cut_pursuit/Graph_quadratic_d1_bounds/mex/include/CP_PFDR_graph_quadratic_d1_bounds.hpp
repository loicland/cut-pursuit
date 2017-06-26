/*==================================================================
 * minimize functional over a graph G = (V, E)
 *
 *       F(x) = 1/2 ||y - x||_{l2,La_l2}^2 + ||x||_{d1,La_d1}  + i_{[m, M]}(x)
 *
 * where x, y in R^V,
 *      ||x||_{l2,La_l2}^2 = sum_{v in V} la_l2_v (y_v - x_v)^2,
 *      ||x||_{d1,La_d1} = sum_{uv in E} la_d1_uv |x_u - x_v|,
 *      i_{[m, M]}(x) = 0          if for all v in V, m <= x_v <= M,
 *                      +infinity  otherwise
 *
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
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
 * Reference: H. Raguet and L. Landrieu, `Cut-pursuit` algorithm for convex
 * nonsmooth functionals regularized by graph total variation, in preparation.
 *
 * Hugo Raguet 2016
 *================================================================*/
#ifndef CP_PFDR_GRAPH_QUADRATIC_D1_BOUNDS_H
#define CP_PFDR_GRAPH_QUADRATIC_D1_BOUNDS_H

template <typename real> struct CPqb_Restart;
/* structure for "warm restart"
 * for more information, see implemention of CP_PFDR_graph_quadratic_d1_bounds
 * G   - graph structure with V nodes and E edges
 * Vc  - list of vertices within each connected component
 * rVc - cumulative sum of the components sizes (in an array of length (rV + 1))
 * R   - if N > 0 (see below), residual, array of length N
 *       set to NULL otherwise */

template <typename real> struct CPqb_Restart<real>*
create_CPqb_Restart(const int V, const int E, const int N, \
                    int *rV, int *Cv, real **rX, const real *Y, const real *A, \
                    const int *Eu, const int *Ev, const real min, const real max);

template <typename real>
void free_CPqb_Restart(struct CPqb_Restart<real> *CPqb_restart);

template <typename real>
void CP_PFDR_graph_quadratic_d1_bounds(const int V, const int E, const int N, \
    int *rV, int *Cv, real **rX, const real *Y, const real *A, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real min, const real max, \
    const real CP_difTol, const int CP_itMax, int *CP_it, \
    const real PFDR_rho, const real PFDR_condMin, \
    const real PFDR_difRcd, const real PFDR_difTol, const int PFDR_itMax, \
    double *Time, real *Obj, real *Dif, const int verbose, \
    struct CPqb_Restart<real> *CP_restart);
/* 26 arguments:
 * V, E       - number of vertices and of (undirected) edges
 * N          - number of observations.
 *              if nonpositive, matricial information is precomputed,
 *              that is, argument A is (A^t A) and argument Y is (A^t Y)
 *              if exactly zero, A is a diagonal matrix and only the diagonal
 *              of (A^t A) = A^2 is given
 * rV         - adress of an integer keeping track of the number of homogeneous
 *              connected components of the minimizer
 * Cv         - assignement of each vertex of the minimizer to an homogeneous connected
 *              component of the graph, array of length V
 * rX         - adress of a pointer keeping track of the values of each
 *              homogeneous connected components of the minimizer (in an array
 *              of length rV)
 * Y          - if N is positive, observations, array of length N
 *              otherwise, correlation of A with the observations (A^t Y), 
 *              array of length V
 * A          - if N is positive, matrix, N-by-V array, column major format
 *              if N is negative, matrix (A^t A), V-by-V array, column major format
 *              if N is zero, diagonal of (A^t A) = A^2, array of length V
 *              set to NULL (and N to zero) for identity matrix
 * Eu         - for each edge, index of one vertex, array of length E
 * Ev         - for each edge, index of the other vertex, array of length E
 * La_d1      - d1 penalization coefficients, array of length E
 * min        - lower bound constraint, set to -inf for no bound
 * max        - upper bound constraint, set to inf for no bound
 *              (uses HUGE_VAL or HUGE_VALF from math.h)
 *
 * [CP]
 * difTol     - stopping criterion on iterate evolution. Algorithm stops if
 *              relative changes of X (in Euclidean norm) is less than difTol.
 *              1e-2 is a conservative value; 1e-3 or less can give better
 *              precision but with longer computational time
 * itMax      - maximum number of iterations (graph cut and subproblem)
 *              10 cuts solve accurately most problems
 * it         - adress of an integer keeping track of iteration (cut) number
 *
 * [PFDR]
 * rho        - relaxation parameter, 0 < rho < 2
 *              1 is a conservative value; 1.5 often speeds up convergence
 * condMin    - small positive parameter ensuring stability of preconditioning
 *              0 < condMin =< 1; 1e-3 is a conservative value; 1e-6 might 
 *              enhance preconditioning
 * difRcd     - reconditioning criterion on iterate evolution. A reconditioning
 *              is performed if relative changes of X (in Euclidean norm) is less
 *              than difRcd. It is then divided by 10.
 *              0 (no reconditioning) is a conservative value, 10*difTol or 
 *              1e2*difTol might speed up convergence. reconditioning might 
 *              temporarily draw minimizer away from solution, and give bad
 *              subproblem solution
 * difTol     - stopping criterion on iterate evolution. Algorithm stops if
 *              relative changes of X (in Euclidean norm) is less than difTol.
 *              1e-3*CP_difTol is a conservative value.
 * itMax      - maximum number of iterations
 *              1e4 iterations provides enough precision for most subproblems
 *
 * Time       - if not NULL, records elapsed time, array of length (CP_itMax + 1)
 * Obj        - if not NULL and N is nonzero, records the values of the objective
 *              functional, array of length (CP_itMax + 1)
 * Dif        - if not NULL, records the iterate evolution (see CP_difTol),
 *              array of length CP_itMax
 * verbose    - if nonzero, display information on the progress, every 'verbose'
 *              iterations
 * CP_restart - pointer to structure (see above) for "warm restart"; rV, Cv 
 *              and rX should be initialized in coherence with rVc, Vc; if the
 *              objective is monitored, the first value is not computed (Obj[0]
 *              not written). set to NULL for no warm restart
 *              THIS FUNCTIONALITY HAS NOT BEEN TESTED YET */
#endif
