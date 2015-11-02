# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import cholesky
from scipy.stats._multivariate import _squeeze_output, invwishart

from . import MMBase
from ..utils import _chol_downdate, _chol_update, multivariate_t


class CDPGMM(MMBase):
    """
    Collapsed Dirichlet process Gaussian mixture model.

    Parameters
    ----------
    alpha: float
        alpha prior
    xi: array-like
        xi prior
    rho: float
        rho prior
    beta: float
        beta prior
    W: array-like
        W prior
    random state: None or int or np.random.RandomState instance, optional
        If int or RandomState, use it for drawing the random variates.
        If None (or np.random), the global np.random state is used.
        Default is None.
    """

    @staticmethod
    def _process_parameters(dim, alpha, xi, rho, beta, W):
        # Try to infer dimensionality
        if dim is None:
            if xi is None:
                if W is None:
                    dim = 1
                else:
                    W = np.asarray(W, dtype=float)
                    if W.ndim < 2:
                        dim = 1
                    else:
                        dim = W.shape[0]
            else:
                xi = np.asarray(xi, dtype=float)
                dim = xi.size
        else:
            if not np.isscalar(dim):
                msg = ("Dimension of random variable must be a scalar.")
                raise ValueError(msg)

        # Check input sizes and return full arrays for xi and W if necessary
        if xi is None:
            xi = np.zeros(dim, dtype=float)
        xi = np.asarray(xi, dtype=float)

        if W is None:
            W = 1.0
        W = np.asarray(W, dtype=float)

        if dim == 1:
            xi.shape = (1,)
            W.shape = (1, 1)

        if xi.ndim != 1 or xi.shape[0] != dim:
            msg = ("Array 'xi' must be a vector of length %d." % dim)
            raise ValueError(msg)

        if W.ndim == 0:
            W = W * np.eye(dim)
        elif W.ndim == 1:
            W = np.diag(scale)
        elif W.ndim == 2 and W.shape != (dim, dim):
            rows, cols = W.shape
            if rows != cols:
                msg = ("Array 'W' must be square if it is two-dimensional,"
                       " but W.shape = %s." % str(W.shape))
            else:
                msg = ("Dimension mismatch: array 'W' is of shape %s,"
                       " but 'xi' is a vector of length %d.")
                msg = msg % (str(W.shape), len(xi))
            raise ValueError(msg)
        elif W.ndim > 2:
            raise ValueError("Array 'W' must be at most two-dimensional,"
                             " but W.ndim = %d" % W.ndim)

        if rho is None:
            rho = 1.0
        elif not np.isscalar(rho):
            raise ValueError("Float 'rho' must be a scalar.")
        elif rho <= 0.0:
            raise ValueError("Float 'rho' must be larger than zero, but"
                             " rho = %f" % rho)

        if beta is None:
            beta = dim
        elif not np.isscalar(beta):
            raise ValueError("Float 'beta' must be a scalar.")
        elif beta <= dim - 1:
            raise ValueError("Float 'beta' must be larger than the dimension"
                             " minus one, but beta = %f" % beta)

        if alpha is None:
            alpha = 1.0
        elif not np.isscalar(alpha):
            raise ValueError("Float 'alpha' must be a scalar.")
        elif alpha <= 0.0:
            raise ValueError("Float 'alpha' must be larger than zero, but"
                             " alpha = %f" % alpha)

        return dim, alpha, xi, rho, beta, W

    def __init__(self, alpha=1.0, xi=None, rho=1.0, beta=1.0, W=1.0,
            sampler=None, seed=None):
        from ..samplers import CollapsedGibbsSampler
        self.default_sampler = CollapsedGibbsSampler

        super(CDPGMM, self).__init__(sampler, seed)

        self.dim, self.alpha, self.xi, self.rho, self.beta, self.W = \
                self._process_parameters(None, alpha, xi, rho, beta, W)

        self.beta_W_chol = cholesky(self.beta*self.W, lower=True)

    def sample(self, size=1, random_state=None):
        """
        Sample from infinite Gaussian mixture distribution.

        Generates a synthetic data set from a Dirichlet process mixture model
        with Gaussian mixture components. The mixture components are drawn
        from a symmetric Dirichlet prior.

        Parameters
        ----------
        size : int
            Number of samples
        random_state : None or int or np.random.RandomState instance
            Used for drawing the random variates

        Returns
        -------
        X : array
            Samples from the infinite Gaussian mixture model
        z : array
            Component indicator variables
        """

        n, m, shape = self._process_size(size)

        random_state = self._get_random_state(random_state)

        # Maximum number of components is number of examples
        c_max = n

        # Array of vectors of component indicator variables
        c_n = np.zeros(m*n, dtype=int).reshape((shape+(n,)))

        # Array of examples
        x_n = np.zeros((m*n*self.dim), dtype=float).reshape(
                (shape+(n,self.dim)))

        # TODO: Make all of this model-agnostic!
        for index in np.ndindex(shape):
            cov = np.zeros(c_max*self.dim*self.dim, dtype=float).reshape(
                    (c_max,self.dim,self.dim))
            mean = np.zeros(c_max*self.dim, dtype=float).reshape(
                    (c_max,self.dim))

            for i in range(n):
                # Draw a component k for example i from the Chinese restaurant
                # process with concentration parameter alpha
                dist = np.bincount(c_n[index]).astype(float)
                dist[0] = self.alpha

                # TODO: Define a scipy-style categorical distribution object
                #       and sample from it!
                cdf = np.cumsum(dist)
                r = random_state.uniform(size=1) * cdf[-1]
                [k] = cdf.searchsorted(r)
                k = len(dist) if k == 0 else k
                c_n[index+(i,)] = k

                # If k is a new component, instantiate it by drawing its
                # parameters
                if k == len(dist):
                    # Draw from the inverted Wishart distribution
                    cov[k-1] += _squeeze_output(
                            invwishart._rvs(1, (1,), self.dim, self.beta,
                                    self.beta_W_chol, random_state))

                    # Draw from the multivariate normal distribution
                    mean[k-1] += _squeeze_output(
                            random_state.multivariate_normal(
                                    self.xi, cov[k-1] / self.rho, 1))

                x_n[index+(i,)] = _squeeze_output(
                        random_state.multivariate_normal(
                                mean[k-1], cov[k-1], 1))

        c_n = c_n - 1

        return _squeeze_output(x_n), _squeeze_output(c_n)

    class Param(object):
        def __init__(self, mixture_model):
            self.mixture_model = mixture_model

            self._rho_c = None
            self._beta_c = None
            self._xsum_c = None
            self._xi_c = None
            self._beta_W_chol_c = None

        @property
        def rho_c(self):
            if self._rho_c is None:
                return self.mixture_model.rho
            else:
                return self._rho_c

        @property
        def beta_c(self):
            if self._beta_c is None:
                return self.mixture_model.beta
            else:
                return self._beta_c

        @property
        def xsum_c(self):
            if self._xsum_c is None:
                return np.zeros(self.mixture_model.dim, dtype=float)
            else:
                return self._xsum_c

        @property
        def xi_c(self):
            if self._xi_c is None:
                return np.array(self.mixture_model.xi, copy=True)
            else:
                return self._xi_c

        @property
        def beta_W_chol_c(self):
            if self._beta_W_chol_c is None:
                return np.array(self.mixture_model.beta_W_chol, copy=True)
            else:
                return self._beta_W_chol_c

        def update(self, x):
            mm = self.mixture_model

            if self._rho_c is None:
                self._rho_c = mm.rho + 1.0
            else:
                self._rho_c += 1.0

            if self._beta_c is None:
                self._beta_c = mm.beta + 1.0
            else:
                self._beta_c += 1.0

            if self._xsum_c is None:
                self._xsum_c = np.array(x, copy=True)
            else:
                self._xsum_c += x

            self._xi_c = (mm.rho * mm.xi + self._xsum_c) / self._rho_c

            if self._beta_W_chol_c is None:
                self._beta_W_chol_c = np.array(mm.beta_W_chol, copy=True)
            _chol_update(mm.dim, self._beta_W_chol_c,
                    np.sqrt(self._rho_c/(self._rho_c-1.0)) * (x - self._xi_c))

            return self

        def downdate(self, x):
            mm = self.mixture_model

            if self._rho_c is None:
                raise ValueError('rho_c must be updated before it can be'
                                 ' downdated')
            else:
                self._rho_c -= 1.0

            if self._beta_c is None:
                raise ValueError('beta_c must be updated before it can be'
                                 ' downdated')
            else:
                self._beta_c -= 1.0

            if self._xsum_c is None:
                raise ValueError('xsum_c must be updated before it can be'
                                 ' downdated')
            else:
                self._xsum_c -= x

            self._xi_c = (mm.rho * mm.xi + self._xsum_c) / self._rho_c

            if self._beta_W_chol_c is None:
                raise ValueError('beta_W_chol_c must be updated before it can'
                                 ' be downdated')
            _chol_downdate(mm.dim, self._beta_W_chol_c,
                    np.sqrt(self._rho_c/(self._rho_c+1.0)) * (x - self._xi_c))

            return self

    def log_likelihood(self, x, param):
        df = param.beta_c - self.dim + 1.0
        scale_chol = np.sqrt((param.rho_c+1)/(param.rho_c*df)) \
                * param.beta_W_chol_c
        log_det_scale = 2. * np.sum(np.log(np.diagonal(scale_chol)))

        return _squeeze_output(multivariate_t._logpdf(
                multivariate_t._process_quantiles(x, self.dim),
                self.dim, param.xi_c, scale_chol, log_det_scale, df))
