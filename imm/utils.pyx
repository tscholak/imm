# cython: boundscheck = False, nonecheck = False, wraparound = False

cimport cython
from cpython cimport bool
from libc.math cimport (log as clog, sqrt as csqrt, lgamma as cgammaln, hypot,
        exp as cexp)
from numpy.math cimport INFINITY
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport numpy as np
import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import multigammaln


cdef:
    np.float64_t _LOG_2PI = 1.8378770664093453
    np.float64_t _LOG_2 = 0.69314718055994529
    np.float64_t _LOG_PI = 1.1447298858494002


cpdef np.float64_t _logsumexp(Py_ssize_t dim, np.float64_t[::1] a):
    """
    Compute the logarithm of the sum of exponentials of input elements.

    Parameters
    ----------
    dim : int
        Dimension of input vector
    a : ndarray
        Input vector

    Returns
    -------
    res : float
        The result
    """

    cdef:
        Py_ssize_t i
        np.float64_t res = 0.0
        np.float64_t a_max = -INFINITY

    for i in range(dim):
        if a[i] > a_max:
            a_max = a[i]

    for i in range(dim):
        res += cexp(a[i] - a_max)

    res = a_max + clog(res)

    return res


cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] _chol(Py_ssize_t dim,
        np.float64_t[:,::1] a, bool clean=False):
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
    dim : int
        Dimension of input matrix
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
        Py_ssize_t i, j
        int info, inc = 1
        int n = dim
        int nn = n*n
        np.float64_t[::1,:] l = np.empty((dim,dim), order='F')

    blas.dcopy(&nn, &a[0,0], &inc, &l[0,0], &inc)
    lapack.dpotrf('L', &n, &l[0,0], &n, &info)

    if info > 0:
        raise LinAlgError("%d-th leading minor not positive definite" % info)

    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potrf'
                % -info)

    if clean:
        for i in range(dim):
            for j in range(i+1,dim):
                l[i,j] = 0.0

    return np.asarray(l, order='C')


@cython.cdivision(True)
cpdef void _chol_update(Py_ssize_t dim, np.float64_t[:,::1] l,
        np.float64_t[::1] x):
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
        Py_ssize_t i, j
        np.float64_t r, c, s

    with nogil:
        for j in range(dim):
            r = hypot(l[j,j], x[j])
            c = r / l[j,j]
            s = x[j] / l[j,j]
            l[j,j] = r
            for i in range(j+1, dim):
                l[i,j] = (l[i,j] + s*x[i]) / c
                x[i] = c*x[i] - s*l[i,j]


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
cpdef void _chol_downdate(Py_ssize_t dim, np.float64_t[:,::1] l,
        np.float64_t[::1] x):
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
        Py_ssize_t i, j
        np.float64_t r, c, s

    with nogil:
        for j in range(dim):
            r = csqrt((l[j,j] - x[j]) * (l[j,j] + x[j]))
            c = r / l[j,j]
            s = x[j] / l[j,j]
            l[j,j] = r
            for i in range(j+1, dim):
                l[i,j] = (l[i,j] - s*x[i]) / c
                x[i] = c*x[i] - s*l[i,j]


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


cpdef np.float64_t _chol_logdet(Py_ssize_t dim, np.float64_t[:,::1] l):
    """
    Return the log-determinant of a matrix given its Cholesky factor `l`.

    Parameters
    ----------
    dim : int
        Dimension of the matrix
    l : ndarray
        Lower-triangular Cholesky factor of the matrix

    Returns
    -------
    logdet : float
        Log-determinant of the matrix
    """

    cdef:
        Py_ssize_t i
        np.float64_t logdet = 0.0

    for i in range(dim):
        logdet += clog(l[i,i])
    logdet *= 2.0

    return logdet


cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] _chol_solve(Py_ssize_t dim,
        np.float64_t[:,::1] l, np.float64_t[::1] b):
    """
    Solve the linear equation
    ```
        l l' x = b
    ```
    given the lower triangular Cholesky factor `l` and the right-hand side
    vector `b`.

    Parameters
    ----------
    dim : int
        Dimension of the Cholesky factor and the right-hand side vector
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
        int n = dim
        int nrhs = 1
        np.float64_t[::1] x = np.empty((dim,), order='F')

    blas.dcopy(&n, &b[0], &inc, &x[0], &inc)
    # We have to say 'U' here because of row-major ('C') order of `l`
    lapack.dpotrs('U', &n, &nrhs, &l[0,0], &n, &x[0], &n, &info)

    if info != 0:
        raise ValueError('illegal value in %d-th argument of internal potrs'
                % -info)

    return np.asarray(x, order='C')


cpdef np.float64_t _normal_logpdf(np.float64_t[::1] x, Py_ssize_t dim,
        np.float64_t[::1] mean, np.float64_t[:,::1] prec_chol,
        np.float64_t prec_logdet):
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

    cdef:
        Py_ssize_t i, j
        int n = dim
        int inc = 1
        np.float64_t alpha = -1.0
        np.float64_t[::1] dev = np.empty((dim,), order='F')
        np.float64_t inner, maha = 0.0

    # Compute the squared Mahalanobis distance
    #     (x - mean)' prec (x - mean)
    blas.dcopy(&n, &x[0], &inc, &dev[0], &inc)
    blas.daxpy(&n, &alpha, &mean[0], &inc, &dev[0], &inc)
    for j in range(dim):
        inner = 0.0
        for i in range(j,dim):
            inner += dev[i] * prec_chol[i, j]
        maha += inner*inner

    return -0.5 * (<np.float64_t>dim * _LOG_2PI - prec_logdet + maha)


cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] _normal_rvs(Py_ssize_t dim,
        np.float64_t[::1] mean, np.float64_t[:,::1] prec_chol, random_state):
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

    cdef:
        Py_ssize_t i
        int n = dim
        int nrhs = 1
        int lda = dim
        int ldx = dim
        int info = 1
        np.float64_t[::1] x = random_state.normal(loc=0.0, scale=1.0,
                size=dim)

    # We have to say 'U', 'N' here because of row-major ('C') order of
    # `prec_chol`. Otherwise, we would say 'L', 'T'
    lapack.dtrtrs('U', 'N', 'N', &n, &nrhs, &prec_chol[0,0], &lda, &x[0],
            &ldx, &info)

    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %s"
                % (info-1))
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal trtrs'
                % -info)

    for i in range(dim):
        x[i] += mean[i]

    return np.asarray(x, order='C')


@cython.cdivision(True)
cpdef np.float64_t _t_logpdf(np.float64_t[::1] x, Py_ssize_t dim,
        np.float64_t[::1] loc, np.float64_t df,
        np.float64_t[:,::1] scale_chol, np.float64_t scale_logdet):
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

    cdef:
        Py_ssize_t i
        int info
        int n = dim
        int nrhs = 1
        int lda = dim
        int inc = 1
        np.float64_t alpha = -1.0
        np.float64_t logpdf, maha = 0.0
        np.float64_t[::1] dev = np.empty((dim,), order='F')

    # Compute the squared Mahalanobis distance
    #     (x - loc)' scale^(-1) (x - loc)
    blas.dcopy(&n, &x[0], &inc, &dev[0], &inc)
    blas.daxpy(&n, &alpha, &loc[0], &inc, &dev[0], &inc)
    # We have to say 'U', 'T' here because of row-major ('C') order of
    # `invscale_chol`. Otherwise, we would say 'L', 'N'
    lapack.dtrtrs('U', 'T', 'N', &n, &nrhs, &scale_chol[0,0], &lda, &dev[0],
            &lda, &info)
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %s"
                % (info-1))
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal trtrs'
                % -info)
    for i in range(dim):
        maha += dev[i]*dev[i]

    logpdf = -0.5 * (<np.float64_t>dim * (clog(df) + _LOG_PI) + scale_logdet +
            (df + <np.float64_t>dim) * clog(1.0 + maha / df)) - \
            cgammaln(0.5 * df) + cgammaln(0.5 * (df + <np.float64_t>dim))

    return logpdf


@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] _t_rvs(Py_ssize_t dim,
        np.float64_t[::1] loc, np.float64_t df,
        np.float64_t[:,::1] scale_chol, random_state):
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
    rvs : ndarray
        Random variate of size `dim`, where `dim` is the dimension of the
        location vector and the scale matrix
    """

    cdef:
        Py_ssize_t i
        int inc = 1
        int n = dim
        int lda = dim
        np.float64_t sqrtg = csqrt(random_state.gamma(df/2.0, 2.0/df))
        np.float64_t[::1] x = random_state.normal(size=dim)

    # We have to say 'U', 'T' here because of row-major ('C') order of
    # `scale_chol`. Otherwise, we would say 'L', 'N'
    blas.dtrmv('U', 'T', 'N', &n, &scale_chol[0,0], &lda, &x[0], &inc)

    # Cannot use daxpy here, because that would overwrite `loc`
    for i in range(dim):
        x[i] = loc[i] + x[i] / sqrtg

    return np.asarray(x, order='C')


cpdef np.float64_t _wishart_logpdf(np.float64_t[:,:] x_chol,
        np.float64_t x_logdet, Py_ssize_t dim, np.float64_t df,
        np.float64_t[:,:] invscale_chol, np.float64_t invscale_logdet):
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
    df : float
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

    cdef:
        Py_ssize_t i, j, k
        np.float64_t inner, ret
        np.float64_t invscalex_tr = 0.0

    # Compute Tr[scale^(-1) x]
    for i in range(dim):
        for j in range(dim):
            inner = 0.0
            for k in range(max(i,j), dim):
                inner += invscale_chol[k, i] * x_chol[k, j]
            invscalex_tr += inner*inner

    ret = (df - <np.float64_t>dim - 1.0) * x_logdet - invscalex_tr + \
            df * (invscale_logdet - <np.float64_t>dim * _LOG_2)
    ret *= 0.5
    ret -= multigammaln(0.5 * df, <np.float64_t>dim)

    return ret


cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] _wishart_rvs(Py_ssize_t dim,
        np.float64_t df, np.float64_t[:,::1] invscale_chol, random_state):
    """
    Draw a random sample from a Wishart distribution.

    Parameters
    ----------
    dim : int
        Dimension of the inverse scale matrix
    df : float
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

    cdef:
        Py_ssize_t i, j
        np.float64_t[::1,:] a = np.empty((dim,dim), order='F')
        np.float64_t[::1,:] x = np.empty((dim,dim), order='F')
        int info
        int n = dim
        int nrhs = dim
        int k = dim
        np.float64_t alpha = 1.0
        int lda = dim
        np.float64_t beta = 0.0

    for i in range(dim):
        for j in range(0,i):
            a[i,j] = random_state.normal()
            a[j,i] = 0.0
        a[i,i] = csqrt(random_state.chisquare(df-(i+1)+1))

    # We have to say 'U', 'N' here because of row-major ('C') order of
    # `invscale_chol`. Otherwise, we would say 'L', 'T'
    lapack.dtrtrs('U', 'N', 'N', &n, &nrhs, &invscale_chol[0,0], &lda,
            &a[0,0], &lda, &info)

    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %s"
                % (info-1))
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal trtrs'
                % -info)

    blas.dsyrk('L', 'N', &n, &k, &alpha, &a[0,0], &lda, &beta, &x[0,0], &n)

    for i in range(dim):
        for j in range(0,i):
            x[j,i] = x[i,j]

    return np.asarray(x, order='C')
