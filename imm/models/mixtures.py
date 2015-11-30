# -*- coding: utf-8 -*-

import abc
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import (_squeeze_output, _process_quantiles,
        multivariate_normal, wishart, invwishart)

from ..utils import _chol_downdate, _chol_update, multivariate_t


class GenericMixture(object):
    """
    Class which encapsulates common functionality between all mixture models.
    """

    __metaclass__  = abc.ABCMeta

    def __init__(self):

        super(GenericMixture, self).__init__()

    class Param(object):

        __metaclass__  = abc.ABCMeta

        def __init__(self, random_state):

            self._random_state = check_random_state(random_state)

        @abc.abstractmethod
        def dump(self):

            raise NotImplementedError

    class DrawParam(Param):

        __metaclass__  = abc.ABCMeta

        @abc.abstractmethod
        def draw_x_n(self):

            raise NotImplementedError

    class InferParam(Param):

        __metaclass__  = abc.ABCMeta

        @abc.abstractmethod
        def update(self, x):

            raise NotImplementedError

        @abc.abstractmethod
        def downdate(self, x):

            raise NotImplementedError

        @abc.abstractmethod
        def iterate(self):

            raise NotImplementedError

        @abc.abstractmethod
        def log_likelihood(self, x):

            raise NotImplementedError


class ConjugateGaussianMixture(GenericMixture):
    """
    Conjugate Gaussian mixture model. Parametrization according to
    Görür and Rasmussen (2010), DOI 10.1007/s11390-010-1051-1.

    Parameters
    ----------
    xi : None or array-like, optional
        xi hyperparameter
    rho : None or float, optional
        rho hyperparameter
    beta : None or float, optional
        beta hyperparameter. Must be larger than the dimension of xi minus one
    W : None or array-like, optional
        W hyperparameter
    """

    def __init__(self, xi=None, rho=1.0, beta=1.0, W=1.0):

        super(ConjugateGaussianMixture, self).__init__()

        self.dim, self.xi, self.rho, self.beta, self.W = \
                self._check_parameters(None, xi, rho, beta, W)
        self._rho_xi = self.rho * self.xi
        self._beta_W_chol = cholesky(self.beta*self.W, lower=True)

    @staticmethod
    def _check_parameters(dim, xi, rho, beta, W):

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
            W = np.diag(W)
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

        return dim, xi, rho, beta, W

    @classmethod
    def _check_mixture_model(cls, mixture_model):

        if not isinstance(mixture_model, cls):
            raise ValueError("'mixture_model' must be a conjugate"
                             " Gaussian mixture model."
                             " Got mixture_model = %r" % mixture_model)

        return mixture_model

    class DrawParam(GenericMixture.DrawParam):

        def __init__(self, mixture_model, random_state):

            super(ConjugateGaussianMixture.DrawParam, self).__init__(
                    random_state)

            self._mixture_model = \
                    ConjugateGaussianMixture._check_mixture_model(
                            mixture_model)

            self._cov = None
            self._mean = None

        @property
        def cov(self):

            if self._cov is None:

                mm = self._mixture_model

                self._cov = _squeeze_output(invwishart._rvs(1, (1,), mm.dim,
                        mm.beta, mm._beta_W_chol, self._random_state))

            return self._cov

        @property
        def mean(self):

            if self._mean is None:

                self._mean = _squeeze_output(
                        self._random_state.multivariate_normal(
                                self._mixture_model.xi,
                                self.cov / self._mixture_model.rho,
                                1))

            return self._mean

        def draw_x_n(self):

            return _squeeze_output(self._random_state.multivariate_normal(
                    self.mean, self.cov, 1))

        def dump(self):

            return self._mean, self._cov

    class InferParam(GenericMixture.InferParam):

        def __init__(self, mixture_model, random_state):

            super(ConjugateGaussianMixture.InferParam, self).__init__(
                    random_state)

            self._mixture_model = \
                    ConjugateGaussianMixture._check_mixture_model(
                            mixture_model)

            self._rho_c = None
            self._beta_c = None
            self._xsum_c = None
            self._xi_c = None
            self._beta_W_c_chol = None
            self._df = None
            self._scale_chol = None
            self._scale_logdet = None

        @property
        def rho_c(self):

            if self._rho_c is None:
                return self._mixture_model.rho
            else:
                return self._rho_c

        @property
        def beta_c(self):

            if self._beta_c is None:
                return self._mixture_model.beta
            else:
                return self._beta_c

        @property
        def xi_c(self):

            if self._xi_c is None:
                return self._mixture_model.xi
            else:
                return self._xi_c

        @property
        def beta_W_c_chol(self):

            if self._beta_W_c_chol is None:
                return self._mixture_model._beta_W_chol
            else:
                return self._beta_W_c_chol

        @property
        def df(self):

            if self._df is None:
                self._df = self.beta_c - self._mixture_model.dim + 1.0
            return self._df

        @property
        def scale_chol(self):

            if self._scale_chol is None:
                self._scale_chol = np.sqrt((self.rho_c+1.0) / \
                        (self.rho_c*self.df)) * self.beta_W_c_chol
            return self._scale_chol

        @property
        def scale_logdet(self):

            if self._scale_logdet is None:
                self._scale_logdet = 2.0 * np.sum(np.log(np.diagonal(
                        self.scale_chol)))
            return self._scale_logdet

        def update(self, x):

            mm = self._mixture_model

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

            self._xi_c = (mm._rho_xi + self._xsum_c) / self._rho_c

            if self._beta_W_c_chol is None:
                self._beta_W_c_chol = np.array(mm._beta_W_chol, copy=True)
            _chol_update(mm.dim, self._beta_W_c_chol,
                    np.sqrt(self._rho_c/(self._rho_c-1.0)) * (x-self._xi_c))

            self._df = None
            self._scale_chol = None
            self._scale_logdet = None

            return self

        def downdate(self, x):

            mm = self._mixture_model

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

            self._xi_c = (mm._rho_xi + self._xsum_c) / self._rho_c

            if self._beta_W_c_chol is None:
                raise ValueError('beta_W_c_chol must be updated before it can'
                                 ' be downdated')
            _chol_downdate(mm.dim, self._beta_W_c_chol,
                    np.sqrt(self._rho_c/(self._rho_c+1.0)) * (x-self._xi_c))

            self._df = None
            self._scale_chol = None
            self._scale_logdet = None

            return self

        def iterate(self):

            pass

        def log_likelihood(self, x):

            dim = self._mixture_model.dim

            return _squeeze_output(multivariate_t._logpdf(
                    multivariate_t._process_quantiles(x, dim), dim, self.xi_c,
                    self.scale_chol, self.scale_logdet, self.df))

        def dump(self):

            return self._rho_c, self._beta_c, self._xsum_c, self._xi_c, \
                    self._beta_W_c_chol, self._df, self._scale_chol, \
                    self._scale_logdet


class NonconjugateGaussianMixture(GenericMixture):
    """
    Conditionally conjugate Gaussian mixture model. Parametrization according
    to Görür and Rasmussen (2010), DOI 10.1007/s11390-010-1051-1.

    Parameters
    ----------
    xi : None or array-like, optional
        xi hyperparameter
    R : None or array-like, optional
        R hyperparameter
    beta : None or float, optional
        beta hyperparameter
    W : None or array-like, optional
        W hyperparameter
    """

    def __init__(self, xi=None, R=1.0, beta=1.0, W=1.0):

        super(NonconjugateGaussianMixture, self).__init__()

        self.dim, self.xi, self.R, self.beta, self.W = \
                self._check_parameters(None, xi, R, beta, W)
        self._R_xi = np.dot(self.R, self.xi)
        self._R_chol = cholesky(self.R, lower=True)
        self._beta_W_chol = cholesky(self.beta*self.W, lower=True)

    @staticmethod
    def _check_parameters(dim, xi, R, beta, W):

        if dim is None:
            if xi is None:
                if R is None:
                    if W is None:
                        dim = 1
                    else:
                        W = np.asarray(W, dtype=float)
                        if W.ndim < 2:
                            dim = 1
                        else:
                            dim = W.shape[0]
                else:
                    R = np.asarray(R, dtype=float)
                    if R.ndim < 2:
                        dim = 1
                    else:
                        dim = R.shape[0]
            else:
                xi = np.asarray(xi, dtype=float)
                dim = xi.size
        else:
            if not np.isscalar(dim):
                msg = ("Dimension of random variable must be a scalar.")
                raise ValueError(msg)

        if xi is None:
            xi = np.zeros(dim, dtype=float)
        xi = np.asarray(xi, dtype=float)

        if R is None:
            R = 1.0
        R = np.asarray(R, dtype=float)

        if W is None:
            W = 1.0
        W = np.asarray(W, dtype=float)

        if dim == 1:
            xi.shape = (1,)
            R.shape = (1, 1)
            W.shape = (1, 1)

        if xi.ndim != 1 or xi.shape[0] != dim:
            msg = ("Array 'xi' must be a vector of length %d." % dim)
            raise ValueError(msg)

        if R.ndim == 0:
            R = R * np.eye(dim)
        elif R.ndim == 1:
            R = np.diag(R)
        elif R.ndim == 2 and R.shape != (dim, dim):
            rows, cols = R.shape
            if rows != cols:
                msg = ("Array 'R' must be square if it is two-dimensional,"
                       " but R.shape = %s." % str(R.shape))
            else:
                msg = ("Dimension mismatch: array 'R' is of shape %s,"
                       " but 'xi' is a vector of length %d.")
                msg = msg % (str(R.shape), len(xi))
            raise ValueError(msg)
        elif R.ndim > 2:
            raise ValueError("Array 'R' must be at most two-dimensional,"
                             " but R.ndim = %d" % R.ndim)

        if W.ndim == 0:
            W = W * np.eye(dim)
        elif W.ndim == 1:
            W = np.diag(W)
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

        if beta is None:
            beta = dim
        elif not np.isscalar(beta):
            raise ValueError("Float 'beta' must be a scalar.")
        elif beta <= dim - 1:
            raise ValueError("Float 'beta' must be larger than the dimension"
                             " minus one, but beta = %f" % beta)

        return dim, xi, R, beta, W

    @classmethod
    def _check_mixture_model(cls, mixture_model):

        if not isinstance(mixture_model, cls):
            raise ValueError("'mixture_model' must be a non-conjugate"
                             " Gaussian mixture model."
                             " Got mixture_model = %r" % mixture_model)

        return mixture_model

    class DrawParam(GenericMixture.DrawParam):

        def __init__(self, mixture_model, random_state):

            super(NonconjugateGaussianMixture.DrawParam, self).__init__(
                    random_state)

            self._mixture_model = \
                    NonconjugateGaussianMixture._check_mixture_model(
                            mixture_model)

            self._precision_chol = None
            self._mean = None

        @property
        def precision_chol(self):

            if self._precision_chol is None:

                mm = self._mixture_model

                precision = _squeeze_output(wishart._rvs(1, (1,), mm.dim,
                        mm.beta, mm._beta_W_chol, self._random_state))

                self._precision_chol = cholesky(precision, lower=True)

            return self._precision_chol

        @property
        def mean(self):

            if self._mean is None:

                self._mean = self._random_state.normal(loc=0.0, scale=1.0,
                        size=self._mixture_model.dim)
                self._mean = solve_triangular(self._mixture_model._R_chol,
                        self._mean, lower=True, trans='T', overwrite_b=True,
                        check_finite=False)
                self._mean += self._mixture_model.xi

            return self._mean

        def draw_x_n(self):

            x_n = self._random_state.normal(loc=0.0, scale=1.0,
                    size=self._mixture_model.dim)
            x_n = solve_triangular(self.precision_chol, x_n, lower=True,
                    trans='T', overwrite_b=True, check_finite=False)
            x_n += self.mean

            return x_n

        def dump(self):

            return self._mean, self._precision_chol

    class InferParam(GenericMixture.InferParam):

        def __init__(self, mixture_model, random_state):

            super(NonconjugateGaussianMixture.InferParam, self).__init__(
                    random_state)

            self._mixture_model = \
                    NonconjugateGaussianMixture._check_mixture_model(
                            mixture_model)

            self._n_c = None
            self._beta_c = None
            self._xsum_c = None
            self._beta_W_help_c_chol = None
            self._mu_c = None
            self._S_c = None

        @property
        def n_c(self):

            if self._n_c is None:
                return 0
            else:
                return self._n_c

        @property
        def beta_c(self):

            if self._beta_c is None:
                return self._mixture_model.beta
            else:
                return self._beta_c

        @property
        def xsum_c(self):

            if self._xsum_c is None:
                return np.zeros(self._mixture_model.dim, dtype=float)
            else:
                return self._xsum_c

        @property
        def beta_W_help_c_chol(self):

            if self._beta_W_help_c_chol is None:
                return self._mixture_model._beta_W_chol
            else:
                return self._beta_W_help_c_chol

        @property
        def mu_c(self):

            if self._mu_c is None:
                mm = self._mixture_model
                self._mu_c = self._draw_mu_c(mm.dim, self.n_c, self.xsum_c,
                        self._S_c, mm.R, mm._R_chol, mm.xi, mm._R_xi,
                        self._random_state)
            return self._mu_c

        @staticmethod
        def _draw_mu_c(dim, n_c, xsum_c, S_c, R, R_chol, xi, R_xi,
                random_state):

            if n_c > 0:
                if S_c is None:
                    raise ValueError
                S_c, _, _ = S_c
                R_c_chol = cholesky(R + n_c*S_c, lower=True)
                xi_c = cho_solve((R_c_chol, True), np.dot(S_c, xsum_c) + R_xi,
                        overwrite_b=True, check_finite=False)
            else:
                R_c_chol = R_chol
                xi_c = xi

            mu_c = random_state.normal(loc=0.0, scale=1.0, size=dim)
            mu_c = solve_triangular(R_c_chol, mu_c, lower=True, trans='T',
                    overwrite_b=True, check_finite=False)
            mu_c += xi_c

            return mu_c

        @property
        def S_c(self):

            if self._S_c is None:
                self._S_c = self._draw_S_c(self._mixture_model.dim, self.n_c,
                        self.beta_W_help_c_chol, self.xsum_c, self._mu_c,
                        self.beta_c, self._random_state)
            return self._S_c

        @staticmethod
        def _draw_S_c(dim, n_c, beta_W_help_c_chol, xsum_c, mu_c, beta_c,
                random_state):

            if n_c > 0:
                if mu_c is None:
                    raise ValueError
                beta_W_c_chol = np.array(beta_W_help_c_chol, copy=True)
                _chol_update(dim, beta_W_c_chol, np.sqrt(n_c) * \
                        (xsum_c/n_c - mu_c))
            else:
                beta_W_c_chol = beta_W_help_c_chol

            S_c = _squeeze_output(wishart._standard_rvs(1, (1,), dim, beta_c,
                    random_state))
            S_c = solve_triangular(beta_W_c_chol, S_c, trans='T', lower=True,
                    unit_diagonal=False, overwrite_b=True, check_finite=False)
            S_c = np.dot(S_c, S_c.T)
            # S_c = _squeeze_output(wishart._rvs(1, (1,), dim, beta_c,
            #         cholesky(np.linalg.inv(np.dot(beta_W_c_chol,
            #                 beta_W_c_chol.T), lower=True), random_state))
            S_c_chol = cholesky(S_c, lower=True)
            S_c_logdet = 2.0 * np.sum(np.log(np.diagonal(S_c_chol)))

            return S_c, S_c_chol, S_c_logdet

        def update(self, x):

            mm = self._mixture_model

            if self._beta_W_help_c_chol is None:
                self._beta_W_help_c_chol = np.array(mm._beta_W_chol,
                        copy=True)
            else:
                _chol_update(mm.dim, self._beta_W_help_c_chol,
                        np.sqrt(self._n_c / float(self._n_c+1)) * \
                                (x - self._xsum_c/self._n_c))

            if self._n_c is None:
                self._n_c = 1
            else:
                self._n_c += 1

            if self._xsum_c is None:
                self._xsum_c = np.array(x, copy=True)
            else:
                self._xsum_c += x

            if self._beta_c is None:
                self._beta_c = mm.beta + 1.0
            else:
                self._beta_c += 1.0

            return self

        def downdate(self, x):

            mm = self._mixture_model

            if self._beta_W_help_c_chol is None:
                raise ValueError('beta_W_help_c must be updated before it can'
                                 ' be downdated')
            elif self._n_c > 1:
                _chol_downdate(mm.dim, self._beta_W_help_c_chol,
                        np.sqrt(self._n_c / float(self._n_c-1)) * \
                                (x - self._xsum_c/self._n_c))
            else:
                self._beta_W_help_c_chol = None

            if self._n_c is None:
                raise ValueError('n_c must be updated before it can be'
                                 ' downdated')
            elif self._n_c > 1:
                self._n_c -= 1
            else:
                self._n_c = None

            if self._xsum_c is None:
                raise ValueError('xsum_c must be updated before it can be'
                                 ' downdated')
            elif self._n_c is not None:
                self._xsum_c -= x
            else:
                self._xsum_c = None

            if self._beta_c is None:
                raise ValueError('beta_c must be updated before it can be'
                                 ' downdated')
            elif self._n_c is not None:
                self._beta_c -= 1.0
            else:
                self._beta_c = None

            return self

        def iterate(self):

            mm = self._mixture_model
            dim = mm.dim

            self._mu_c = self._draw_mu_c(dim, self.n_c, self.xsum_c,
                    self._S_c, mm.R, mm._R_chol, mm.xi, mm._R_xi,
                    self._random_state)
            self._S_c = self._draw_S_c(dim, self.n_c, self.beta_W_help_c_chol,
                    self.xsum_c, self._mu_c, self.beta_c, self._random_state)

        def log_likelihood(self, x):

            dim = self._mixture_model.dim
            _, S_c_chol, S_c_logdet = self.S_c

            return _squeeze_output(multivariate_normal._logpdf(
                    _process_quantiles(x, dim), self.mu_c,
                    S_c_chol, -S_c_logdet, dim))

        def dump(self):

            return self._n_c, self._R_c_chol, self._beta_c, self._xsum_c, \
                    self._xi_c, self._beta_W_help_c_chol, self._mu_c, \
                    self._S_c
