# -*- coding: utf-8 -*-

import abc
import numpy as np
from scipy.misc import factorial
from scipy.special import poch, gammaln
from scipy.stats import poisson
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _squeeze_output

from ..utils import _logsumexp


class GenericProcess(object):
    """
    Class which encapsulates common functionality between all process models.
    """

    __metaclass__  = abc.ABCMeta

    default_sampler = None

    def __init__(self, mixture_model, sampler=None, seed=None):

        super(GenericProcess, self).__init__()

        self._mixture_model = self._check_mixture_model(mixture_model)
        self._sampler = self._check_sampler(sampler)
        self._random_state = check_random_state(seed)

    def _check_mixture_model(self, mixture_model):

        from .mixtures import GenericMixture

        if issubclass(mixture_model, GenericMixture):
            return mixture_model()

        if isinstance(mixture_model, GenericMixture):
            return mixture_model

        raise ValueError("'mixture_model' must be a mixture model."
                         " Got mixture_model = %r" % mixture_model)

    @property
    def mixture_model(self):

        return self._mixture_model

    def _check_sampler(self, sampler):

        if sampler is None:
            return self.default_sampler(self)

        from ..samplers import GenericSampler

        if issubclass(sampler, GenericSampler):
            return sampler(self)

        if isinstance(sampler, GenericSampler):
            if not sampler.process_model == self:
                raise ValueError('%r was initialized with a different process'
                                 ' model and is thus incompatible.' % sampler)
            return sampler

        raise ValueError("'sampler' must be a compatible sampler."
                         " Got sampler = %r" % sampler)

    @property
    def random_state(self):
        """
        Get or set the RandomState object for generating random variates.
        This can be either None or an existing RandomState object.
        If None (or np.random), use the RandomState singleton used by
        np.random. If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):

        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):

        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state

    @property
    def sampler(self):

        return self._sampler

    @sampler.setter
    def sampler(self, sampler):

        self._sampler = self._check_sampler(sampler)

    def _get_sampler(self, sampler):

        if sampler is not None:
            return self._check_sampler(sampler)
        else:
            return self._sampler

    @staticmethod
    def _process_size(size):

        size = np.asarray(size, dtype=int)

        if size.ndim == 0:
            n = np.asscalar(size)
            size = np.asarray([1])
        elif size.ndim == 1:
            n = size[0]
            if len(size) == 1:
                size = np.asarray([1])
            elif len(size) > 1:
                size = size[1:]
        elif size.ndim > 1:
            raise ValueError('Size must be an integer or tuple of integers;'
                             ' thus must have dimension <= 1.'
                             ' Got size.ndim = %s' % str(tuple(size)))

        m = size.prod()
        shape = tuple(size)

        return n, m, shape

    class _Param(object):

        __metaclass__  = abc.ABCMeta

        def __init__(self, random_state):

            self._random_state = check_random_state(random_state)

        @abc.abstractmethod
        def dump(self):

            raise NotImplementedError

    class DrawParam(_Param):

        __metaclass__  = abc.ABCMeta

        @abc.abstractmethod
        def log_prior(self, n, n_c):

            raise NotImplementedError

    class InferParam(_Param):

        __metaclass__  = abc.ABCMeta

        @abc.abstractmethod
        def log_prior(self, n, n_c, m):

            raise NotImplementedError

        @abc.abstractmethod
        def iterate(self, n, k):

            raise NotImplementedError

    def draw(self, size, random_state=None):
        """
        Draw from the Chinese restaurant process.
        """

        mm = self._mixture_model

        n, m, shape = self._process_size(size)

        random_state = self._get_random_state(random_state)

        # Array of vectors of component indicator variables. In the beginning,
        # assign every example to the component with indicator value 0
        c_n = np.empty(m*n, dtype=int).reshape((shape+(n,)))

        # Maximum number of components is number of examples
        c_max = n

        # Array of examples
        # TODO: Make this truly model-agnostic, i.e. get rid of mm.dim and
        #       dtype=float
        x_n = np.empty((m*n*mm.dim), dtype=float).reshape((shape+(n,mm.dim)))

        for index in np.ndindex(shape):
            process_param = self.DrawParam(self, random_state)

            n_c = np.zeros(c_max, dtype=int)

            active_components = set()
            inactive_components = set(range(c_max))

            # Lazily instantiate the components
            mixture_params = [mm.DrawParam(mm, random_state)
                    for _ in range(c_max)]

            for i in range(n):
                # Draw a component k for example i from the Chinese restaurant
                # process

                # Get a new component from the stack
                prop_k = inactive_components.pop()
                active_components.add(prop_k)

                # Initialize and populate the log probability accumulator
                log_dist = np.empty(c_max, dtype=float)
                log_dist.fill(-np.inf)
                for k in active_components:
                    # Calculate the process prior
                    log_dist[k] = process_param.log_prior(i+1, n_c[k])

                # Sample from log_dist. Normalization is required
                log_dist -= _logsumexp(c_max, log_dist)
                next_k = random_state.choice(a=c_max, p=np.exp(log_dist))

                # cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
                # r = random_state.uniform(size=1) * cdf[-1]
                # [next_k] = cdf.searchsorted(r)

                c_n[index+(i,)] = next_k

                # Update component counter
                n_c[next_k] += 1

                # New components are instantiated automatically when needed
                x_n[index+(i,)] = mixture_params[next_k].draw_x_n()

                # Cleanup
                if next_k != prop_k:
                    active_components.remove(prop_k)
                    inactive_components.add(prop_k)

        # TODO: Make it possible to return the component parameters

        return _squeeze_output(x_n), _squeeze_output(c_n)

    def infer(self, *args, **kwargs):

        sampler = self._get_sampler(kwargs.pop('sampler', None))

        c_n, phi_c = sampler.infer(*args, **kwargs)

        return _squeeze_output(c_n), phi_c


class DP(GenericProcess):
    """
    Dirichlet process.

    Parameters
    ----------
    alpha : None or float, optional
        alpha hyperparameter
    a : None or float, optional
        shape of Gamma hyperprior on alpha
    b : None or float, optional
        scale of Gamma hyperprior on alpha
    random_state : None or int or np.random.RandomState instance, optional
        If int or RandomState, use it for drawing the random variates.
        If None (or np.random), the global np.random state is used.
        Default is None.
    """

    def __init__(self, mixture_model, alpha=1.0, a=None, b=None, sampler=None,
            seed=None):

        from ..samplers import GibbsSampler
        self.default_sampler = GibbsSampler

        super(DP, self).__init__(mixture_model, sampler, seed)

        self.alpha, self.a, self.b = self._check_parameters(alpha, a, b)

    @staticmethod
    def _check_parameters(alpha, a, b):

        for (var, varname) in zip([alpha, a, b], ['alpha', 'a', 'b']):
            if var is not None:
                if not np.isscalar(var):
                    raise ValueError("Float '%s' must be a scalar." % varname)
                var = float(var)
                if var <= 0.0:
                    raise ValueError("Float '%s' must be larger than zero,"
                                     " but %s = %f" % (varname, varname, var))

        if alpha is None:
            if a is None or b is None:
                alpha = 1.0

        return alpha, a, b

    @classmethod
    def _check_process_model(cls, process_model):

        if not isinstance(process_model, cls):
            raise ValueError("'process_model' must be a Dirichlet"
                             " process (DP) model."
                             " Got process_model = %r" % process_model)

        return process_model

    def _ms_log_prior_pre(self, n, t, tp, process_param):
        """
        First term in the logarithm of the prior appearing in the M-H
        acceptance ratio used by the merge-split samplers.
        """

        ret = (t-tp) * np.log(process_param.alpha)

        return ret

    def _ms_log_prior_post(self, nc, process_param):
        """
        Second term in the logarithm of the prior appearing in the M-H
        acceptance ratio used by the merge-split samplers.
        """

        ret = gammaln(nc)

        return ret

    class DrawParam(GenericProcess.DrawParam):

        def __init__(self, process_model, random_state):

            super(DP.DrawParam, self).__init__(random_state)

            self._process_model = DP._check_process_model(process_model)

            self._alpha = None

        @property
        def alpha(self):

            if self._alpha is None:
                pm = self._process_model

                # TODO: Use try-except instead?
                if pm.a is not None and pm.b is not None:
                    self._alpha = self._random_state.gamma(pm.a, pm.b)
                else:
                    self._alpha = pm.alpha

            return self._alpha

        def log_prior(self, n, n_c):
            """
            Return the logarithm of the conditional prior.
            """

            if n_c == 0:
                return np.log(self.alpha)
            elif n_c > 0:
                return np.log(n_c)
            else:
                raise ValueError("'n_c' must be non-negative.")

        def dump(self):

            return self._alpha

    class InferParam(GenericProcess.InferParam):

        def __init__(self, process_model, random_state):

            super(DP.InferParam, self).__init__(random_state)

            self._process_model = DP._check_process_model(process_model)

            self._alpha = None

        @property
        def alpha(self):

            if self._alpha is None:
                return self._process_model.alpha
            else:
                return self._alpha

        def log_prior(self, n, n_c, m):
            """
            Return the logarithm of the conditional prior.
            """

            if n_c == 0:
                # The conditional prior of a new component is proportional to
                # alpha
                return np.log(self.alpha) - np.log(m)
            elif n_c > 0:
                # The conditional priors of all existing components are
                # proportional to the number of examples assigned to them
                return np.log(n_c)
            else:
                raise ValueError("'n_c' must be non-negative.")

        def iterate(self, n, k):
            """
            Sample a new value for alpha.
            """

            pm = self._process_model

            if pm.a is not None and pm.b is not None:

                x = self._random_state.beta(alpha+1.0, n)

                shape = pm.a + k - 1.0
                scale = pm.b - np.log(x)

                pi_x = shape / (shape + n * scale)

                alpha = pi_x * self._random_state.gamma(shape+1.0, scale)
                alpha += (1-pi_x) * self._random_state.gamma(shape, scale)

                self._alpha = alpha

        def dump(self):

            return self._alpha


class MFM(GenericProcess):
    """
    Mixture of finite mixtures.

    Parameters
    ----------
    gamma : None or float, optional
        gamma hyperparameter
    mu : None or float, optional
        parameter of the Poisson distribution for the number of components
    random_state : None or int or np.random.RandomState instance, optional
        If int or RandomState, use it for drawing the random variates.
        If None (or np.random), the global np.random state is used.
        Default is None.
    """

    def __init__(self, mixture_model, gamma=1.0, mu=1.0, sampler=None,
            seed=None):

        from ..samplers import GibbsSampler
        self.default_sampler = GibbsSampler

        super(MFM, self).__init__(mixture_model, sampler, seed)

        self.gamma, self.mu = self._check_parameters(gamma, mu)

    @staticmethod
    def _check_parameters(gamma, mu):

        if gamma is not None:
            if not np.isscalar(gamma):
                raise ValueError("Float 'gamma' must be a scalar.")
            elif gamma <= 0.0:
                raise ValueError("Float 'gamma' must be larger than zero, but"
                                 " gamma = %f" % b)
        else:
            gamma = 1.0

        if mu is not None:
            if not np.isscalar(mu):
                raise ValueError("Float 'mu' must be a scalar.")
            elif mu <= 0.0:
                raise ValueError("Float 'mu' must be larger than zero, but"
                                 " mu = %f" % b)
        else:
            mu = 1.0

        return gamma, mu

    @classmethod
    def _check_process_model(cls, process_model):

        if not isinstance(process_model, cls):
            raise ValueError("'process_model' must be a mixture of"
                             " finite mixtures (MFM) model."
                             " Got process_model = %r" % process_model)

        return process_model

    @staticmethod
    def _log_v_quotient(n, t, tp, gamma, mu, k_max=1000, diff=50.0, memo={}):
        # TODO: Make it possible to use a custom pmf p_K, e.g.
        #       - Geometric: p_K(k) = (1-r)^(k-1) * r
        #       - Poisson: p_K(k) = mu^(k-1)/(k-1)! * exp(-mu)

        try:

            return memo[(n, t, tp)]

        except KeyError:

            def help(s):
                comp = gammaln(n) - gammaln(n+s*gamma) - diff
                ret = np.empty(k_max, dtype=float)
                for k in range(k_max):
                    ret[k] = gammaln(n) + k*np.log(mu) - gammaln(1.0+k) - \
                            gammaln(1.0+s*gamma) - gammaln(n+(k+s)*gamma) + \
                            gammaln(1.0+(k+s)*gamma)
                    if ret[k] < comp:
                        break
                ret = s*gamma*np.log(n) + _logsumexp(k+1, ret[:k+1])
                return ret

            ret = help(t) - help(tp)
            ret += (t-tp) * (np.log(mu) - gamma*np.log(n))
            ret += gammaln(1.0 + t*gamma) - gammaln(1.0 + tp*gamma)

            memo[(n, t, tp)] = ret

            return ret

    def _ms_log_prior_pre(self, n, t, tp, process_param):
        """
        First term in the logarithm of the prior appearing in the M-H
        acceptance ratio used by the merge-split samplers.
        """

        ret = self._log_v_quotient(n, t, tp, process_param.gamma,
                process_param.mu, memo=process_param._log_v_quotient_memo)

        return ret

    def _ms_log_prior_post(self, nc, process_param):
        """
        Second term in the logarithm of the prior appearing in the M-H
        acceptance ratio used by the merge-split samplers.
        """

        ret = gammaln(process_param.gamma + nc) - gammaln(process_param.gamma)

        return ret

    class DrawParam(GenericProcess.DrawParam):

        def __init__(self, process_model, random_state):

            super(MFM.DrawParam, self).__init__(random_state)

            self._process_model = MFM._check_process_model(process_model)

            self._gamma = None
            self._mu = None
            self._log_v_quotient_memo = {}

        @property
        def gamma(self):

            if self._gamma is None:
                return self._process_model.gamma
            else:
                return self._gamma

        @property
        def mu(self):

            if self._mu is None:
                return self._process_model.mu
            else:
                return self._mu

        def log_prior(self, n, n_c):
            """
            Return the logarithm of the conditional prior.
            """

            if n_c == 0:
                pm = self._process_model
                ret = np.log(self.gamma) + pm._log_v_quotient(n, n_c+1, n_c,
                        self.gamma, self.mu, memo=self._log_v_quotient_memo)
                return ret
            elif n_c > 0:
                return np.log(n_c + self.gamma)
            else:
                raise ValueError("'n_c' must be non-negative.")

        def dump(self):

            return self._gamma, self._mu

    class InferParam(GenericProcess.InferParam):

        def __init__(self, process_model, random_state):

            super(MFM.InferParam, self).__init__(random_state)

            self._process_model = MFM._check_process_model(process_model)

            self._gamma = None
            self._mu = None
            self._log_v_quotient_memo = {}

        @property
        def gamma(self):

            if self._gamma is None:
                return self._process_model.gamma
            else:
                return self._gamma

        @property
        def mu(self):

            if self._mu is None:
                return self._process_model.mu
            else:
                return self._mu

        def log_prior(self, n, n_c, m):
            """
            Return the logarithm of the conditional prior.
            """

            if n_c == 0:
                # a new component

                pm = self._process_model

                ret = np.log(self.gamma)
                ret += pm._log_v_quotient(n, n_c+1, n_c, self.gamma, self.mu,
                        memo=self._log_v_quotient_memo)
                # TODO: Check that!
                ret -= np.log(m)

                return ret

            elif n_c > 0:
                # an existing Component

                return np.log(n_c + self.gamma)

            else:
                raise ValueError("'n_c' must be non-negative.")

        def iterate(self, n, k):

            pass

        def dump(self):

            return self._gamma, self._mu
