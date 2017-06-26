/*==================================================================
 * compute the squared operator norm (squared greatest eigenvalue) of a real 
 * matrix using power method.
 *
 * Hugo Raguet 2016
 *================================================================*/
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <time.h>
#include <cmath>
#ifdef MEX
    #include "mex.h"
    #define FLUSH mexEvalString("drawnow expose")
#else
    #define FLUSH fflush(stdout)
#endif

/* constants of the correct type */
#define ZERO ((real) 0.)

static const int HALF_RAND_MAX = (RAND_MAX/2 + 1);

/* minimum problem size each thread should take care of within parallel regions */
#define CHUNKSIZE 1000

static int compute_num_threads(const int size)
{
#ifdef _OPENMP
    if (size > omp_get_num_procs()*CHUNKSIZE){
        return omp_get_num_procs();
    }else{
        return 1 + (size - CHUNKSIZE)/CHUNKSIZE;
    }
#else
    return 1;
#endif
}

template <typename real>
static inline real compute_norm(const real *X, const int N)
{
    int n;
    real b = ZERO;
    for (n = 0; n < N; n++){ b += X[n]*X[n]; }
    return sqrt(b);
}

template <typename real>
static inline void normalize_and_apply_matrix(const real *A, real *X, real *AX, \
                            const real b, const int symmetrized, const int M, const int N)
{
    int m, n, p;
    real a;
    const real *An;
    if (symmetrized){
        for (n = 0; n < N; n++){ AX[n] = X[n]/b; }
    }else{
        for (n = 0; n < N; n++){ X[n] /= b; }
        /* apply A */
        for (m = 0; m < M; m++){
            a = ZERO;
            p = m;
            for (n = 0; n < N; n++){
                a += A[p]*X[n];
                p += M;
            }
            AX[m] = a;
        }
    }
    /* apply A^t or AA */
    An = A;
    for (n = 0; n < N; n++){
        a = ZERO;
        for (m = 0; m < M; m++){ a += An[m]*AX[m]; }
        X[n] = a;
        An += M;
    }
}

template <typename real>
real operator_norm_matrix(int M, int N, const real *A, \
                          const real nTol, const int itMax, int nbInit, const int verbose)
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
{

    /* initialize general purpose variables */
    int i, m, n, p, it;
    real a, b, n2, *X, *AX, *AA = NULL;
    const real *An;
    int symmetrized = 0;

    /* pseudo-random generator */
    unsigned int seed;

    /* preprocessing */
    const int P = (M < N) ? M : N;
    i = nbInit*itMax;
    if (P == 0){
        symmetrized = 1;
        M = (M > N) ? M : N;
        N = M;
    }else if (2*M*N*i > (M*N*P + P*P*i)){
        symmetrized = 1;
        const int ntMNP = compute_num_threads(M*N*P/2);
        const int ntP = compute_num_threads(P);
        /* compute symmetrization */
        real *Ap;
        const real *Am;
        AA = (real*) calloc(P*P, sizeof(real));
        if (P == M){ /* case A A^t */
            /* upper triangular part */
            #pragma omp parallel for private(m, n, p, Am, An, Ap, a) num_threads(ntMNP)
            for (p = 0; p < P; p++){
                Am = A;
                An = A + p;
                Ap = AA + P*p;
                for (n = 0; n < N; n++){
                    a = *An;
                    for (m = 0; m <= p; m++){ Ap[m] += a*Am[m]; }
                    An += M;
                    Am += M;
                }
            }
        }else{ /* case A^t A */
            /* upper triangular part */
            #pragma omp parallel for private(m, n, p, Am, An, Ap, a) num_threads(ntMNP)
            for (p = 0; p < P; p++){
                Am = A;
                An = A + M*p;
                Ap = AA + P*p;
                for (n = 0; n <= p; n++){
                    a = ZERO;
                    for (m = 0; m < M; m++){ a += (*(Am++))*An[m]; }
                    Ap[n] = a;
                }
            }
        }
        /* copy in lower triangular part */
        #pragma omp parallel for private(m, n, p, Ap) num_threads(ntP)
        for (p = 0; p < P-1; p++){
            Ap = AA + P*p;
            m = P + (P+1)*p;
            for (n = p+1; n < P; n++){
                Ap[n] = AA[m];
                m += P;
            }
        }
        M = P;
        N = P;
        A = AA;
    }

    /* power method */
    const int ntI = omp_get_num_procs();
    nbInit = (1 + (nbInit - 1)/ntI)*ntI;
    if (verbose){
        printf("compute matrix operator norm on %d random initializations over %d parallel threads... ", nbInit, ntI);
        FLUSH;
    }

    n2 = ZERO;
    #pragma omp parallel private(X, AX, i, n, a, b, seed) reduction(max:n2) num_threads(ntI)
    {
    seed = time(NULL) + omp_get_thread_num(); 
    X = (real*) alloca(N*sizeof(real));
    AX = (real*) alloca(M*sizeof(real));
    #pragma omp for
    for (i = 0; i < nbInit; i++){
        /* random initialization */
        for (n = 0; n < N; n++){
            /* very crude uniform distribution on [-1,1] */
            X[n] = ((real) (rand_r(&seed) - HALF_RAND_MAX))/((real) HALF_RAND_MAX);
        }
        b = compute_norm(X, N);
        normalize_and_apply_matrix(A, X, AX, b, symmetrized, M, N);
        b = compute_norm(X, N);
        /* iterate */
        if (b > ZERO){
            for (it = 0; it < itMax; it++){
                normalize_and_apply_matrix(A, X, AX, b, symmetrized, M, N);
                a = compute_norm(X, N);
                if ((a - b)/b < nTol){ break; }
                b = a;
            }
        }
        if (b > n2){ n2 = b; }
    }
    }
    if (verbose){ printf("done.\n"); FLUSH; }
    free(X);
    free(AX);
    free(AA);
    return n2;
}

/* instantiate for compilation */
template float operator_norm_matrix<float>(int, int, const float*, \
                                const float, const int, int, const int);

template double operator_norm_matrix<double>(int, int, const double*, \
                                const double, const int, int, const int);
