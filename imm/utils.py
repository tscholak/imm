# -*- coding: utf-8 -*-

"""
Extra distributions to complement scipy.stats
"""

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.stats._multivariate import (multi_rv_generic, multi_rv_frozen,
        _squeeze_output, _LOG_PI)
from scipy.special import gammaln


# TODO: Replace with cython version for a ~ 10-fold speedup
def _chol_downdate(dim, L, x):
    """
    Downdate the lower triangular Cholesky factor L with the rank-1
    subtraction implied by x such that:
        L_' L_ = L' L - outer(x, x)
    where L_ is the lower triangular Cholesky factor L after updating.
    """

    for k in range(dim):
        r = np.sqrt(L[k,k]**2 - x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k+1:dim, k] = (L[k+1:dim, k] - s*x[k+1:dim]) / c
        x[k+1:dim] = c*x[k+1:dim] - s*L[k+1:dim, k]


# TODO: Replace with cython version for a ~ 10-fold speedup
def _chol_update(dim, L, x):
    """
    Update the lower triangular Cholesky factor L with the rank-1 addition
    implied by x such that:
        L_' L_ = L' L + outer(x, x)
    where L_ is the lower triangular Cholesky factor L after updating.
    """

    for k in range(dim):
        r = np.sqrt(L[k,k]**2 + x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k+1:dim, k] = (L[k+1:dim, k] + s*x[k+1:dim]) / c
        x[k+1:dim] = c*x[k+1:dim] - s*L[k+1:dim, k]


class multivariate_t_gen(multi_rv_generic):
    """
    A multivariate t random variable.

    The `loc` keyword specifies the location. The `scale` keyword specifies
    the scale matrix. The `df` keyword refers to the number of degrees of
    freedom.

    Methods
    -------
    ``pdf(x, loc, scale, df)``
        Probability density function.
    ``logpdf(x, loc, scale, df)``
        Logarithm of the probability density function.
    ``rvs(loc, scale, df, size=1, random_state=None)``
        Draw random samples from a multivariate t distribution.

    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the logarithm of the probability
        density function

    Alternatively, the object may be called (as a function) to fix the
    location, the scale, and degrees of freedom parameters, returning a
    "frozen" multivariate t random variable:

    rv = multivariate_t(loc=None, scale=1, df=1)
        - Frozen object with the same methods but holding the given location,
          scale matrix, and number of degrees of freedom fixed.

    Notes
    -----
    The scale matrix `scale` must be a (symmetric) positive semi-definite
    matrix. The determinant and inverse of `scale` are computed as the
    pseudo-determinant and pseudo-inverse, respectively, so that `scale` does
    not need to have full rank.
    """

    def __init__(self, seed=None):
        super(multivariate_t_gen, self).__init__(seed)

    def __call__(self, loc=None, scale=None, df=None, seed=None):
        """
        Create a frozen multivariate t distribution.

        See `multivariate_t_frozen` for more information.
        """

        return multivariate_t_frozen(loc, scale, df, seed=seed)

    @staticmethod
    def _process_parameters(dim, loc, scale, df):
        # Try to infer dimensionality
        if dim is None:
            if loc is None:
                if scale is None:
                    dim = 1
                else:
                    scale = np.asarray(scale, dtype=float)
                    if scale.ndim < 2:
                        dim = 1
                    else:
                        dim = scale.shape[0]
            else:
                loc = np.asarray(loc, dtype=float)
                dim = loc.size
        else:
            if not np.isscalar(dim):
                msg = ("Dimension of random variable must be a scalar.")
                raise ValueError(msg)

        # Check input sizes and return full arrays for loc and scale if
        # necessary
        if loc is None:
            loc = np.zeros(dim)
        loc = np.asarray(loc, dtype=float)

        if scale is None:
            scale = 1.0
        scale = np.asarray(scale, dtype=float)

        if dim == 1:
            loc.shape = (1,)
            scale.shape = (1, 1)

        if loc.ndim != 1 or loc.shape[0] != dim:
            msg = ("Array 'loc' must be a vector of length %d." % dim)
            raise ValueError(msg)

        if scale.ndim == 0:
            scale = scale * np.eye(dim)
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and scale.shape != (dim, dim):
            rows, cols = scale.shape
            if rows != cols:
                msg = ("Array 'scale' must be square if it is two-dimensional"
                       ", but scale.shape = %s." % str(scale.shape))
            else:
                msg = ("Dimension mismatch: array 'scale' is of shape %s,"
                       " but 'loc' is a vector of length %d.")
                msg = msg % (str(scale.shape), len(loc))
            raise ValueError(msg)
        elif scale.ndim > 2:
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)

        if df is None:
            df = dim
        elif not np.isscalar(df):
            raise ValueError("Degrees of freedom must be a scalar.")
        elif df <= 0.0:
            raise ValueError("Degrees of freedom must be larger than zero,"
                             " but df = %f" % df)

        return dim, loc, scale, df

    @staticmethod
    def _process_quantiles(x, dim):
        """
        Adjust quantiles array such that the last axis of `x` denotes the
        components of each data point.
        """

        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis, np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        elif x.ndim > 2:
            raise ValueError("Quantiles must be at most one-dimensional with"
                             " an additional dimension for multiple"
                             " components, but x.ndim = %d" % x.ndim)

        if not x.shape[-1] == dim:
            raise ValueError('Quantiles have incompatible dimensions: should'
                             ' be %s, got %s.' % (dim, x.shape[-1]))

        return x

    @staticmethod
    def _process_size(size):
        size = np.asarray(size)

        if size.ndim == 0:
            size = size[np.newaxis]
        elif size.ndim > 1:
            raise ValueError('Size must be an integer or tuple of integers;'
                             ' thus must have dimension <= 1.'
                             ' Got size.ndim = %s' % str(tuple(size)))
        n = size.prod()
        shape = tuple(size)

        return n, shape

    def _logpdf(self, x, dim, loc, scale_chol, log_det_scale, df):
        """
        Logarithm of the multivariate t probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the logarithm of the probability
            density function
        dim : int
            Dimension of the location and the scale matrix
        loc : ndarray
            Location of the distribution
        scale_chol : ndarray
            Lower triangular Cholesky factor of the scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        df : float
            Degrees of freedom

        Returns
        -------
        logpdf : ndarray
            Logarithm of the probability density function evaluated at `x`

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly. Use 'logpdf' instead.
        """

        # TODO: Replace with cython version

        # Retrieve the squared Mahalanobis distance
        # (x - loc)' scale^{-1} (x - loc)
        diff = (x - loc).T
        maha = np.sum(diff * scipy.linalg.cho_solve((scale_chol, True), diff),
                axis=0).reshape(-1, 1).squeeze()

        # Log PDF
        out = (-0.5 * dim *(np.log(df) + _LOG_PI) + gammaln(0.5 * (df + dim))
               - gammaln(0.5 * df) - 0.5 * log_det_scale
               - 0.5 * (df + dim) * np.log(1.0 + maha / df))

        return out

    def logpdf(self, x, loc, scale, df):
        """
        Logarithm of the multivariate t probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the logarithm of the probability
            density function
        loc : ndarray
            Location of the distribution
        scale : ndarray
            Scale matrix
        df : float
            Degrees of freedom

        Returns
        -------
        logpdf : ndarray
            Logarithm of the probability density function evaluated at `x`
        """

        dim, loc, scale, df = self._process_parameters(
                None, loc, scale, df)

        x = self._process_quantiles(x, dim)

        scale_chol = scipy.linalg.cholesky(scale, lower=True)
        log_scale_det = 2. * np.sum(np.log(np.diagonal(scale_chol)))

        out = self._logpdf(x, dim, loc, scale_chol, log_scale_det, df)

        return _squeeze_output(out)

    def pdf(self, x, loc, scale, df):
        """
        Multivariate t probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the probability density function
        loc : ndarray
            Location of the distribution
        scale : ndarray
            Scale matrix
        df : float
            Degrees of freedom

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`
        """

        out = np.exp(self.logpdf(x, loc, scale, df))

        return _squeeze_output(out)

    def _rvs(self, n, shape, dim, loc, scale_chol, df, random_state):
        """
        Draw random samples from a multivariate t distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the location and the scale matrix
        loc : ndarray
            Location of the distribution
        scale_chol : ndarray
            Lower triangular Cholesky factor of the scale matrix
        df : float
            Degrees of freedom
        random_state : None or int or np.random.RandomState instance
            Used for drawing the random variates

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly. Use 'rvs' instead.
        """

        # TODO: Replace with cython version

        random_state = self._get_random_state(random_state)

        g = np.tile(random_state.gamma(df/2., 2./df, n),
                 (dim, 1)).T.reshape(shape+(dim,))
        norm = random_state.normal(size=n*dim).reshape(shape+(dim,))

        x = np.zeros(shape + (dim,))
        for index in np.ndindex(shape):
            Z = np.dot(scale_chol, norm[index]).T
            x[index] = loc + Z / np.sqrt(g[index])

        return x

    def rvs(self, loc, scale, df, size=1, random_state=None):
        """
        Draw random samples from a multivariate t distribution.

        Parameters
        ----------
        loc : ndarray
            Location of the distribution
        scale: ndarray
            Scale matrix
        df: float
            Degrees of freedom
        size : integer, optional
            Number of samples to draw (default 1)
        random state: None or int or np.random.RandomState instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None (or np.random), the global np.random state is used.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """

        n, shape = self._process_size(size)
        dim, loc, scale, df = self._process_parameters(None, loc, scale, df)

        random_state = self._get_random_state(random_state)

        scale_chol = scipy.linalg.cholesky(scale, lower=True)

        out = self._rvs(n, shape, dim, loc, scale_chol, df, random_state)

        return _squeeze_output(out)


multivariate_t = multivariate_t_gen()


class multivariate_t_frozen(multi_rv_frozen):
    """
    Create a frozen multivariate t distribution.

    Parameters
    ----------
    loc : array_like, optional
        Location of the distribution (default zero)
    scale : array_like, optional
        Scale matrix of the distribution (default one)
    df : int, optional
        Degrees of freedom, must be greater than or equal to dimension of
        the scale matrix (default one)
    seed : None or int or np.random.RandomState instance, optional
        This parameter defines the RandomState object to use for drawing
        random variates.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local RandomState instance
        Default is None.
    """

    def __init__(self, loc=None, scale=1, df=1, seed=None):
        self._dist = multivariate_t_gen(seed)

        self.dim, self.loc, self.shape, self.df = \
            self._dist._process_parameters(None, loc, shape, df)

        # Obtain the lower-triangular Cholesky factor of the scale matrix
        self.scale_chol = scipy.linalg.cholesky(self.scale, lower=True)

        self.log_scale_det = 2. * np.sum(np.log(np.diagonal(self.scale_chol)))

    @staticmethod
    def _process_rank1(x):
        """
        Adjust the `x` array such that the last axis of `x` denotes the
        components.
        """

        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if self.dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        if not x.shape == (self.dim, ):
            raise ValueError('The array has incompatible dimensions: should'
                             ' be %s, got %s.' % ((self.dim,), x.shape))

        return x

    def logpdf(self, x):
        """
        Logarithm of the multivariate t probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the logarithm of the probability
            density function

        Returns
        -------
        logpdf : ndarray
            Logarithm of the probability density function evaluated at `x`
        """

        x = self._dist._process_quantiles(x, self.dim)

        out = self._dist._logpdf(x, self.dim, self.loc,
                self.scale_chol, self.log_scale_det, self.df)

        return _squeeze_output(out)

    def pdf(self, x):
        """
        Multivariate t probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the probability density function

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`
        """

        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from a multivariate t distribution.

        Parameters
        ----------
        size : integer, optional
            Number of samples to draw (default 1)
        random state: None or int or np.random.RandomState instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None (or np.random), the global np.random state is used.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the dimension
            of the random variable.
        """

        n, shape = self._dist._process_size(size)

        random_state = self._get_random_state(random_state)

        return self._dist._rvs(n, shape,
                self.dim, self.loc, self.scale_chol, self.df, random_state)

    def scale_chol_downdate(self, x, downdate_scale=True):
        """
        Downdate the lower triangular Cholesky factor of the scale matrix with
        the rank-1 subtraction implied by x.

        Parameters
        ----------
        x : array-like
            Downdate vector
        downdate_scale : bool, optional
            If true, then not only the Cholesky factors of the scale matrix,
            but also the scale matrix itself will be downdated (default true)
        """

        x = self._process_rank1(x)

        _chol_downdate(self.dim, self.scale_chol, x)

        self.log_scale_det = 2. * np.sum(np.log(np.diagonal(self.scale_chol)))

        if downdate_scale:
            self.scale = np.dot(self.scale_chol, self.scale_chol.T)

    def scale_chol_update(self, x, update_scale=True):
        """
        Update the lower triangular Cholesky factor of the scale matrix with
        the rank-1 addition implied by x.

        Parameters
        ----------
        x : array-like
            Update vector
        update_scale : bool, optional
            If true, then not only the Cholesky factors of the scale matrix,
            but also the scale matrix itself will be updated (default true)
        """

        x = self._process_rank1(x)

        _chol_update(self.dim, self.scale_chol, x)

        self.log_scale_det = 2. * np.sum(np.log(np.diagonal(self.scale_chol)))

        if update_scale:
            self.scale = np.dot(self.scale_chol, self.scale_chol.T)
