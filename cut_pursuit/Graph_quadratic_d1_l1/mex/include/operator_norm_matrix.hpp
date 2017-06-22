/*==================================================================
 * compute the squared operator norm (squared greatest eigenvalue) of a real 
 * matrix using power method.
 *
 * Hugo Raguet 2016
 *================================================================*/
#ifndef OPERATOR_NORM_MATRIX
#define OPERATOR_NORM_MATRIX

template <typename real>
real operator_norm_matrix(int M, int N, const real *A, \
                          const real nTol, const int itMax, int nbInit, const int verbose);
/* 7 arguments:
 * M, N        - matrix dimensions; set M or N to zero for symmetrized version.
 * A           - if M and N are nonzero, A is an M-by-N array, column major format
 *               if M or N is zero, then A is actually (A^t A) or (A A^t),
 *               M-by-M or N-by-N (whichever is nonzero) array, column major format.
 * nTol        - stopping criterion on relative norm evolution.
 * itMax       - maximum number of iterations
 * nbInit      - number of random initializations
 * verbose     - if nonzero, display information on the progress
 * returns the square operator norm of the matrix */

#endif
