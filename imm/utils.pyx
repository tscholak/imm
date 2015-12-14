# -*- coding: utf-8 -*-
# cython: boundscheck = False, nonecheck = False, wraparound = False

cimport cython
from cpython cimport bool
from libc.math cimport sqrt as csqrt
from libc.math cimport hypot
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport numpy as np
import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import gammaln, multigammaln


_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] _chol(np.float64_t[:,::1] a,
        bool clean=True):
    """
    Return the Cholesky decomposition,
    ```
        a = l l',
    ```
    where `l` is a lower triangular matrix with real and positive diagonal
    entries.

    This function is a thin wrapper for the `dpotrf` LAPACK function. Expect
    significant speedups only for small matrices.

    Parameters
    ----------
    a : ndarray
        Matrix to be decomposed
    clean : boolean, optional
        Whether to zero strictly upper part of `l`

    Returns
    -------
    l : ndarray
        Lower-triangular Cholesky factor of `a`
    """

    cdef:
        int i, j, info
        int inc = 1
        int n = a.shape[0]
        int nn = n*n
        np.float64_t[::1,:] l = np.empty((n,n), order='F')

    with nogil:
        blas.dcopy(&nn, &a[0,0], &inc, &l[0,0], &inc)
        lapack.dpotrf('L', &n, &l[0,0], &n, &info)

    if info > 0:
        raise LinAlgError("%d-th leading minor not positive definite" % info)

    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potrf'
                % -info)

    if clean:
        for i in range(n):
            for j in range(i+1,n):
                l[i,j] = 0.0

    return np.array(l, order='C')


@cython.cdivision(True)
cpdef void _chol_update(int dim, np.float64_t[:,::1] l, np.float64_t[::1] x):
    """
    Update the lower triangular Cholesky factor `l` with the rank-1 addition
    implied by `x` such that:
    ```
        l_' l_ = l' l + outer(x, x),
    ```
    where `l_` is the lower triangular Cholesky factor `l` after updating.

    Parameters
    ----------
    dim : int
        Dimension of the Cholesky factor to be updated
    l : ndarray
        Lower-triangular Cholesky factor. Will be updated in-place
    x : ndarray
        Update vector. Will be destroyed in the process
    """

    cdef:
        int i, j
        np.float64_t r, c, s

    with nogil:
        for j in range(dim):
            r = hypot(l[j, j], x[j])
            c = r / l[j, j]
            s = x[j] / l[j, j]
            l[j, j] = r
            for i in range(j+1, dim):
                l[i, j] = (l[i, j] + s*x[i]) / c
                x[i] = c*x[i] - s*l[i, j]


# TODO: Revisit
# NOTE: Source: https://github.com/mattjj/pylds/blob/master/pylds/util.pxd
# NOTE: See also: https://gist.github.com/mattjj/ad5f57e396eaf0553298
# cpdef void _chol_update2(int dim, np.float64_t[:,::1] l,
#         np.float64_t[::1] x):
#     cdef:
#         int k
#         int inc = 1
#         np.float64_t a, b, c, s
#     with nogil:
#         for k in range(dim):
#             a, b = l[k,k], x[k]
#             blas.drotg(&a, &b, &c, &s)
#             blas.drot(&dim, &l[k,0], &inc, &x[0], &inc, &c, &s)


@cython.cdivision(True)
cpdef void _chol_downdate(int dim, np.float64_t[:,::1] l, np.float64_t[::1] x):
    """
    Downdate the lower triangular Cholesky factor `l` with the rank-1
    subtraction implied by `x` such that
    ```
        l_' l_ = l' l - outer(x, x),
    ```
    where `l_` is the lower triangular Cholesky factor `l` after downdating.

    Parameters
    ----------
    dim : int
        Dimension of the Cholesky factor to be downdated
    l : ndarray
        Lower-triangular Cholesky factor. Will be downdated in-place
    x : ndarray
        Update vector. Will be destroyed in the process
    """

    cdef:
        int i, j
        np.float64_t r, c, s

    with nogil:
        for j in range(dim):
            r = csqrt((l[j, j] - x[j]) * (l[j, j] + x[j]))
            c = r / l[j, j]
            s = x[j] / l[j, j]
            l[j, j] = r
            for i in range(j+1, dim):
                l[i, j] = (l[i, j] - s*x[i]) / c
                x[i] = c*x[i] - s*l[i, j]


# TODO: Revisit
# NOTE: Source: https://github.com/mattjj/pylds/blob/master/pylds/util.pxd
# NOTE: See also: https://gist.github.com/mattjj/ad5f57e396eaf0553298
# @cython.cdivision(True)
# cpdef void _chol_downdate2(int dim, np.float64_t[:,::1] l,
#         np.float64_t[::1] x):
#     cdef:
#         int k, j
#         np.float64_t rbar
#     with nogil:
#         for k in range(dim):
#             rbar = csqrt((l[k,k] - x[k]) * (l[k,k] + x[k]))
#             for j in range(k+1,dim):
#                 l[k,j] = (l[k,k]*l[k,j] - x[k]*x[j]) / rbar
#                 x[j] = (rbar*x[j] - x[k]*l[k,j]) / l[k,k]
#             l[k,k] = rbar


cpdef np.float64_t _chol_logdet(np.float64_t[:,::1] l):
    """
    Return the log-determinant of a matrix given its Cholesky factor `l`.

    Parameters
    ----------
    l : ndarray
        Lower-triangular Cholesky factor of a matrix

    Returns
    -------
    logdet : float
        Log-determinant of the matrix
    """

    ret = 2.0 * np.sum(np.log(np.diagonal(l)))

    return ret


# TODO: Add `np.ndarray[np.float64_t, ndim=1, mode='c']`
cpdef _chol_solve(np.float64_t[:,::1] l, np.float64_t[::1] b):
    """
    Solve the linear equation
    ```
        l l' x = b
    ```
    given the lower triangular Cholesky factor `l` and the right-hand side
    vector `b`.

    Parameters
    ----------
    l : ndarray
        Lower triangular Cholesky factor
    b : ndarray
        Right-hand side vector

    Returns
    -------
    x : ndarray
        Solution
    """

    cdef:
        int info, inc = 1
        int n = l.shape[0]
        int nrhs = 1
        np.float64_t[::1] x = np.empty((n,), order='F')

    with nogil:
        blas.dcopy(&n, &b[0], &inc, &x[0], &inc)
        # We have to say 'U' here because of row-major ('C') order of `l`
        lapack.dpotrs('U', &n, &nrhs, &l[0,0], &n, &x[0], &n, &info)

    if info != 0:
        raise ValueError('illegal value in %d-th argument of internal potrs'
                % -info)

    return np.array(x, order='C')


# TODO: Add `np.ndarray[np.float64_t, ndim=1, mode='c']`
cpdef _solve_triangular_vector(np.float64_t[:,::1] a, np.float64_t[::1] b):
    """
    Solve the equation
    ```
        a' x = b
    ```
    for `x`, assuming `a` is a lower-triangular matrix and `b` a vector.

    Parameters
    ----------
    a : nparray
        A lower-triangular matrix
    b : nparray
        Right-hand side vector

    Returns
    -------
    x : ndarray
        Solution
    """

    cdef:
        int inc = 1
        int n = a.shape[0]
        int nrhs = 1
        int lda = n
        int ldx = n
        int info = 1
        np.float64_t[::1] x = np.empty((n,), order='F')

    with nogil:
        blas.dcopy(&n, &b[0], &inc, &x[0], &inc)
        # We have to say 'U', 'N' here because of row-major ('C') order of
        # `l`. Otherwise, we would say 'L', 'T'
        lapack.dtrtrs('U', 'N', 'N', &n, &nrhs, &a[0,0], &lda, &x[0], &ldx,
                &info)

    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %s"
                % (info-1))
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal trtrs'
                % -info)

    return np.array(x, order='C')


# TODO: Add `np.ndarray[np.float64_t, ndim=2, mode='c']`
cpdef _solve_triangular_matrix(np.float64_t[:,::1] a, np.float64_t[:,::1] b):
    """
    Solve the equation
    ```
        a' x = b
    ```
    for `x`, assuming `a` is a lower-triangular matrix and `b` a matrix.

    Parameters
    ----------
    a : nparray
        A lower-triangular matrix
    b : nparray
        Right-hand side matrix

    Returns
    -------
    x : ndarray
        Solution
    """

    cdef:
        int inc = 1
        int n = a.shape[1]
        int nrhs = b.shape[1]
        int nnrhs = n*nrhs
        int lda = b.shape[0]
        int ldx = b.shape[0]
        int info = 1
        np.float64_t[::1,:] x = np.empty((n,n), order='F')

    with nogil:
        blas.dcopy(&nnrhs, &b[0,0], &inc, &x[0,0], &inc)
        # We have to say 'U', 'N' here because of row-major ('C') order of
        # `l`. Otherwise, we would say 'L', 'T'
        lapack.dtrtrs('U', 'N', 'N', &n, &nrhs, &a[0,0], &lda, &x[0,0], &ldx,
                &info)

    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %s"
                % (info-1))
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal trtrs'
                % -info)

    return np.array(x, order='C')


def _normal_logpdf(x, dim, mean, prec_chol, prec_logdet):
    """
    Logarithm of the multivariate normal probability density function.

    Parameters
    ----------
    x : ndarray
        Point at which to evaluate the logarithm of the probability density
        function
    dim : int
        Dimension of the location and the scale matrix
    mean : ndarray
        Location of the distribution
    prec_chol : ndarray
        Lower triangular Cholesky factor of the precision matrix
    prec_logdet : float
        Log-determinant of the precision matrix

    Returns
    -------
    logpdf : float
        Logarithm of the probability density function evaluated at `x`
    """

    # Compute the squared Mahalanobis distance
    #     (x - mean)' prec (x - mean)
    dev = x - mean
    # TODO: Use only the lower triangular part
    maha = np.sum(np.square(np.dot(dev, prec_chol)), axis=-1)

    ret = -0.5 * (dim * _LOG_2PI - prec_logdet + maha)

    return ret


def _normal_rvs(dim, mean, prec_chol, random_state):
    """
    Draw a random sample from a multivariate normal distribution.

    Parameters
    ----------
    dim : int
        Dimension of the location and the scale matrix
    mean : ndarray
        Location of the distribution
    prec_chol : ndarray
        Lower triangular Cholesky factor of the precision matrix
    random_state : np.random.RandomState instance
        Used for drawing the random variates

    Returns
    -------
    rvs : ndarray or scalar
        Random variate of size `dim`, where `dim` is the dimension of the
        mean vector and the precision matrix
    """

    x = random_state.normal(loc=0.0, scale=1.0, size=dim)
    x = _solve_triangular_vector(prec_chol, x)
    x += mean

    return x


def _t_logpdf(x, dim, loc, df, scale_chol, scale_logdet):
    """
    Logarithm of the multivariate t probability density function.

    Parameters
    ----------
    x : ndarray
        Point at which to evaluate the logarithm of the probability density
        function
    dim : int
        Dimension of the location and the scale matrix
    loc : ndarray
        Location of the distribution
    df : float
        Degrees of freedom
    scale_chol : ndarray
        Lower triangular Cholesky factor of the scale matrix
    scale_logdet : float
        Logarithm of the determinant of the scale matrix

    Returns
    -------
    logpdf : float
        Logarithm of the probability density function evaluated at `x`
    """

    # Compute the squared Mahalanobis distance
    #     (x - loc)' scale^(-1) (x - loc)
    diff = (x - loc).T
    chol_solved = _chol_solve(scale_chol, diff)
    maha = np.sum(diff * chol_solved, axis=0).reshape(-1,1).squeeze()

    ret = -0.5 * dim *(np.log(df) + _LOG_PI)
    ret += gammaln(0.5 * (df + dim)) - gammaln(0.5 * df)
    ret -= 0.5 * scale_logdet
    ret -= 0.5 * (df + dim) * np.log(1.0 + maha / df)

    return ret


def _t_rvs(dim, loc, df, scale_chol, random_state):
    """
    Draw a sample from a multivariate t distribution.

    Parameters
    ----------
    dim : int
        Dimension of the location and the scale matrix
    loc : ndarray
        Location of the distribution
    df : float
        Degrees of freedom
    scale_chol : ndarray
        Lower triangular Cholesky factor of the scale matrix
    random_state : np.random.RandomState instance
        Used for drawing the random variates

    Returns
    -------
    rvs : ndarray or scalar
        Random variate of size `dim`, where `dim` is the dimension of the
        location vector and the scale matrix
    """

    g = random_state.gamma(df/2., 2./df)
    norm = random_state.normal(size=dim)

    Z = np.dot(scale_chol, norm)
    x = loc + Z / np.sqrt(g)

    return x


def _wishart_logpdf(x_chol, x_logdet, dim, df, invscale_chol,
        invscale_logdet):
    """
    Logarithm of the Wishart probability density function.

    Parameters
    ----------
    x_chol : ndarray
        Lower triangular Cholesky factor of the matrix at which to evaluate
        the logarithm of the probability density function
    x_logdet : float
        Log-determinant of the matrix at which to evaluate the logarithm of
        the probability density function
    dim : int
        Dimension of `x` and `scale`
    df : int
        Degrees of freedom
    invscale_chol : ndarray
        Lower triangular Cholesky factor of the inverse scale matrix
    invscale_logdet : float
        Log-determinant of the inverse scale matrix

    Returns
    -------
    logpdf : float
        Logarithm of the probability density function evaluated at `x`
    """

    # Compute Tr[scale^(-1) x]
    # TODO: Make better use of lower triangular shapes
    invscale_x_tr = np.sum(np.square(np.dot(invscale_chol.T, x_chol)))

    ret = 0.5 * (df - dim - 1) * x_logdet
    ret -= 0.5 * invscale_x_tr
    ret -= 0.5 * df * dim * _LOG_2
    ret += 0.5 * df * invscale_logdet
    ret -= multigammaln(0.5 * df, dim)

    return ret


def _wishart_rvs(dim, df, invscale_chol, random_state):
    """
    Draw a random sample from a Wishart distribution.

    Parameters
    ----------
    dim : int
        Dimension of the inverse scale matrix
    df : int
        Degrees of freedom
    invscale_chol : ndarray
        Lower triangular Cholesky factor of the inverse scale matrix
    random_state : np.random.RandomState instance
        Used for drawing the random variates

    Returns
    -------
    rvs : ndarray or scalar
        Random variate of shape (`dim`, `dim`), where `dim` is the dimension
        of the scale matrix
    """

    a = np.zeros((dim, dim))

    n_tril = dim * (dim-1) // 2
    covariances = random_state.normal(size=n_tril).reshape((n_tril,))
    tril_idx = np.tril_indices(dim, k=-1)
    a[tril_idx] = covariances

    diag_idx = np.diag_indices(dim)
    variances = np.array([random_state.chisquare(df-(i+1)+1)**0.5
            for i in range(dim)])
    a[diag_idx] = variances

    a = _solve_triangular_matrix(invscale_chol, a)

    a = np.dot(a, a.T)

    return a
