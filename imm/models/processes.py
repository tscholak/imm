# -*- coding: utf-8 -*-

import abc
import numpy as np
from scipy.misc import factorial, logsumexp
from scipy.special import poch, gammaln
from scipy.stats import poisson
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _squeeze_output


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

    class Param(object):

        __metaclass__  = abc.ABCMeta

        def __init__(self, random_state):

            self._random_state = check_random_state(random_state)

        @abc.abstractmethod
        def dump(self):

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

                # Sample from log_dist. Normalization is not required
                # TODO: Find a better way to sample
                cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
                r = random_state.uniform(size=1) * cdf[-1]
                [next_k] = cdf.searchsorted(r)

                c_n[index+(i,)] = next_k

                # Update component counter
                n_c[next_k] += 1

                # New components are instantiated automatically when needed
                x_n[index+(i,)] = mixture_params[next_k].draw_x_n()

                # Cleanup
                if next_k != prop_k:
                    active_components.remove(prop_k)
                    inactive_components.add(prop_k)

        return _squeeze_output(x_n), _squeeze_output(c_n)

    def infer(self, x_n, c_n=None, sampler=None, max_iter=None, warmup=None,
            random_state=None):

        random_state = self._get_random_state(random_state)
        sampler = self._get_sampler(sampler)

        c_n = sampler.infer(x_n, c_n, max_iter, warmup, random_state)

        return _squeeze_output(c_n)


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

        from ..samplers import CollapsedGibbsSampler
        self.default_sampler = CollapsedGibbsSampler

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

    class Param(GenericProcess.Param):

        def __init__(self, process_model, random_state):

            super(DP.Param, self).__init__(random_state)

            if isinstance(process_model, DP):
                self._process_model = process_model
            else:
                raise ValueError("'process_model' must be a Dirichlet"
                                 " process (DP) model."
                                 " Got process_model = %r" % process_model)

            self._alpha = None

        @property
        def alpha(self):

            if self._alpha is None:
                return self._process_model.alpha
            else:
                return self._alpha

        def log_prior(self, n, n_c):
            """
            Return the logarithm of the conditional prior.
            """

            if n_c == 0:
                # The conditional prior of a new component is proportional to
                # alpha
                return np.log(self.alpha)
            elif n_c > 0:
                # The conditional priors of all existing components are
                # proportional to the number of examples assigned to them
                return np.log(n_c)
            else:
                raise ValueError("'n_c' must be non-negative.")

        def dump(self):
            return self._alpha

    class DrawParam(Param):

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

    class InferParam(Param):

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

        def log_prior_quotient_pre(self, n, t, tp):
            """
            Evaluate certain expressions used by the split-merge samplers.
            """

            return (t-tp) * np.log(self.alpha)

        def log_prior_quotient_post(self, nc):
            """
            Evaluate certain expressions used by the split-merge samplers.
            """

            return gammaln(nc)

    def draw_poop(self, size=1, random_state=None):

        #pm = self._process_model
        mm = self._mixture_model

        n, m, shape = self._process_size(size)

        random_state = self._get_random_state(random_state)

        # Maximum number of components is number of examples
        c_max = n

        # Array of vectors of component indicator variables
        c_n = np.zeros(m*n, dtype=int).reshape((shape+(n,)))

        # Array of examples
        # TODO: Make this truly model-agnostic, i.e. get rid of mm.dim and
        #       dtype=float
        x_n = np.zeros((m*n*mm.dim), dtype=float).reshape((shape+(n,mm.dim)))

        for index in np.ndindex(shape):
            #process_param = pm.DrawParam(pm, random_state)

            # Lazily instantiate the components
            mixture_params = [mm.DrawParam(mm, random_state)
                    for _ in range(c_max)]

            for i in range(n):
                # Draw a component k for example i from the Chinese restaurant
                # process with concentration parameter alpha
                # TODO: Make process-agnostic and move method to generic
                #       process class!
                dist = np.bincount(c_n[index]).astype(float)
                dist[0] = self.alpha

                # TODO: Define a scipy-style categorical distribution object
                #       and sample from it!
                cdf = np.cumsum(dist)
                r = random_state.uniform(size=1) * cdf[-1]
                [k] = cdf.searchsorted(r)
                k = len(dist) if k == 0 else k
                c_n[index+(i,)] = k

                # New components are instantiated automatically when needed
                x_n[index+(i,)] = mixture_params[k-1].draw_x_n()

        c_n = c_n - 1

        return _squeeze_output(x_n), _squeeze_output(c_n)


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

        from ..samplers import CollapsedGibbsSampler
        self.default_sampler = CollapsedGibbsSampler

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

    @staticmethod
    def _v(n, t, gamma, mu, rtol=1e-08, atol=0, kmax=10000, memo={}):
        # TODO: Make it possible to use a custom pmf p_K, e.g.
        #       - Geometric: p_K(k) = (1-r)^(k-1) * r
        #       - Poisson: p_K(k) = mu^(k-1)/(k-1)! * exp(-mu)

        try:
            ret = memo[(n,t)]
            print "cache hit!"
            return ret
        except KeyError:
            pass

        if t <= n:
            try:
                ret = memo[(n-1,t-1)]/gamma - ((n-1)/gamma+t-1)*memo[(n,t-1)]
                print "cache hit!"
                return ret
            except KeyError:
                pass

        v_cur = 0.0
        k = max(1, t)
        pre = factorial(t)

        while True:

            v_old = v_cur
            v_cur += pre / poch(gamma*k, n) * poisson._pmf(k-1, mu)

            if np.isclose(v_cur, v_old, rtol, atol):
                break

            k += 1

            if k == kmax:
                break

            pre *= k / float(k-t)

        memo[(n,t)] = v_cur

        return v_cur

    @staticmethod
    def _log_v_quotient(n, t, tp, gamma, mu, kmax=1000, memo={}):
        # TODO: Sum up only sufficiently large terms
        # TODO: Make it possible to use a custom pmf p_K, e.g.
        #       - Geometric: p_K(k) = (1-r)^(k-1) * r
        #       - Poisson: p_K(k) = mu^(k-1)/(k-1)! * exp(-mu)

        try:
            return memo[(n, t, tp)]
        except KeyError:
            def help(s):
                tmp = np.array(
                        [(k*np.log(mu) - gammaln(1.0+k)
                          + gammaln(1.0+(k+s)*gamma) - gammaln(1.0+s*gamma)
                          + gammaln(n) - gammaln(n+(k+s)*gamma))
                         for k in range(kmax)])
                return s*gamma*np.log(n) + logsumexp(tmp)

            ret = help(t) - help(tp)
            ret += (t-tp) * (np.log(mu) - gamma*np.log(n))
            ret += gammaln(1.0 + t*gamma) - gammaln(1.0 + tp*gamma)

            memo[(n, t, tp)] = ret

            return ret

    class Param(GenericProcess.Param):

        def __init__(self, process_model, random_state):

            super(MFM.Param, self).__init__(random_state)

            if isinstance(process_model, MFM):
                self._process_model = process_model
            else:
                raise ValueError("'process_model' must be a mixture of"
                                 " finite mixtures (MFM) model."
                                 " Got process_model = %r" % process_model)

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
                # a new component

                pm = self._process_model

                ret = np.log(self.gamma) + pm._log_v_quotient(n, n_c+1, n_c,
                        self.gamma, self.mu, memo=self._log_v_quotient_memo)

                return ret

            elif n_c > 0:
                # an existing Component

                return np.log(n_c + self.gamma)

            else:
                raise ValueError("'n_c' must be non-negative.")

        def dump(self):

            return self._gamma, self._mu

    class DrawParam(Param):

        pass

    class InferParam(Param):

        def iterate(self, n, k):
            pass

        def log_prior_quotient_pre(self, n, t, tp):
            """
            Evaluate certain expressions used by the split-merge samplers.
            """

            pm = self._process_model

            ret = pm._log_v_quotient(n, t, tp, self.gamma, self.mu,
                    memo=self._log_v_quotient_memo)

            return ret

        def log_prior_quotient_post(self, nc):
            """
            Evaluate certain expressions used by the split-merge samplers.
            """

            return gammaln(self.gamma + nc) - gammaln(self.gamma)
