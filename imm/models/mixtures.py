# -*- coding: utf-8 -*-

import abc
import numpy as np
from scipy.linalg import cholesky
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _squeeze_output, invwishart

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

class ConjugateGaussianMixture(GenericMixture):
    """
    Conjugate Gaussian mixture model.

    Parameters
    ----------
    xi : None or array-like, optional
        xi prior
    rho : None or float, optional
        rho prior
    beta : None or float, optional
        beta prior
    W : None or array-like, optional
        W prior
    """

    @classmethod
    def _check_parameters(cls, dim, xi, rho, beta, W):

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

        return dim, xi, rho, beta, W

    def __init__(self, xi=None, rho=1.0, beta=1.0, W=1.0):

        super(ConjugateGaussianMixture, self).__init__()

        self.dim, self.xi, self.rho, self.beta, self.W = \
                self._check_parameters(None, xi, rho, beta, W)
        self.beta_W_chol = cholesky(self.beta*self.W, lower=True)

    class Param(GenericMixture.Param):

        def __init__(self, mixture_model, random_state):

            super(ConjugateGaussianMixture.Param, self).__init__(random_state)

            if isinstance(mixture_model, ConjugateGaussianMixture):
                self._mixture_model = mixture_model
            else:
                raise ValueError("'mixture_model' must be a conjugate"
                                 " Gaussian mixture model."
                                 " Got mixture_model = %r" % mixture_model)

    class DrawParam(Param):

        def __init__(self, mixture_model, random_state):

            super(ConjugateGaussianMixture.DrawParam, self).__init__(
                    mixture_model, random_state)

            self._cov = None
            self._mean = None

        @property
        def cov(self):

            if self._cov is None:

                self._cov = _squeeze_output(invwishart._rvs(
                        1, (1,),
                        self._mixture_model.dim,
                        self._mixture_model.beta,
                        self._mixture_model.beta_W_chol,
                        self._random_state))

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

    class InferParam(Param):

        def __init__(self, mixture_model, random_state):

            super(ConjugateGaussianMixture.InferParam, self).__init__(
                    mixture_model, random_state)

            self._rho_c = None
            self._beta_c = None
            self._xsum_c = None
            self._xi_c = None
            self._beta_W_chol_c = None

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
        def xsum_c(self):

            if self._xsum_c is None:
                return np.zeros(self._mixture_model.dim, dtype=float)
            else:
                return self._xsum_c

        @property
        def xi_c(self):

            if self._xi_c is None:
                return np.array(self._mixture_model.xi, copy=True)
            else:
                return self._xi_c

        @property
        def beta_W_chol_c(self):

            if self._beta_W_chol_c is None:
                return np.array(self._mixture_model.beta_W_chol, copy=True)
            else:
                return self._beta_W_chol_c

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

            self._xi_c = (mm.rho * mm.xi + self._xsum_c) / self._rho_c

            if self._beta_W_chol_c is None:
                self._beta_W_chol_c = np.array(mm.beta_W_chol, copy=True)
            _chol_update(mm.dim, self._beta_W_chol_c,
                    np.sqrt(self._rho_c/(self._rho_c-1.0)) * (x - self._xi_c))

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

            self._xi_c = (mm.rho * mm.xi + self._xsum_c) / self._rho_c

            if self._beta_W_chol_c is None:
                raise ValueError('beta_W_chol_c must be updated before it can'
                                 ' be downdated')
            _chol_downdate(mm.dim, self._beta_W_chol_c,
                    np.sqrt(self._rho_c/(self._rho_c+1.0)) * (x - self._xi_c))

            return self

        def log_likelihood(self, x):

            mm = self._mixture_model

            df = self.beta_c - mm.dim + 1.0
            scale_chol = np.sqrt((self.rho_c+1)/(self.rho_c*df)) * \
                    self.beta_W_chol_c
            log_det_scale = 2. * np.sum(np.log(np.diagonal(scale_chol)))

            return _squeeze_output(multivariate_t._logpdf(
                    multivariate_t._process_quantiles(x, mm.dim),
                    mm.dim, self.xi_c, scale_chol, log_det_scale, df))

        def dump(self):

            return (self._rho_c, self._beta_c, self._xsum_c, self._xi_c,
                    self._beta_W_chol_c)
