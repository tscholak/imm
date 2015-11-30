# -*- coding: utf-8 -*-

import abc
import numpy as np

from ..models.processes import GenericProcess
from ..models.mixtures import GenericMixture


class GenericSampler(object):
    """
    Class which encapsulates common functionality between all samplers.
    """

    __metaclass__  = abc.ABCMeta

    compatible_process_models = set()

    compatible_mixture_models = set()

    def __init__(self, process_model, max_iter=1000, warmup=None):

        self._process_model = self._check_process_model(process_model)

        self._mixture_model = self._check_mixture_model(
            self._process_model._mixture_model)

        self._max_iter, self._warmup = self._check_max_iter(max_iter, warmup)

    @classmethod
    def _check_process_model(cls, process_model):

        if isinstance(process_model, GenericProcess):

            pm_class = process_model.__class__
            if pm_class in cls.compatible_process_models:
                return process_model

            raise ValueError('A process model of type %r cannot be used with'
                             ' this sampler' % pm_class.__name__)

        raise ValueError("'process_model' must be a compatible process model"
                         " instance. Got process_model = %r" % process_model)

    @property
    def process_model(self):

        return self._process_model

    @process_model.setter
    def process_model(self, process_model):

        self._process_model = self._check_process_model(process_model)

    def _get_process_model(self, process_model):

        if process_model is not None:
            return self._check_process_model(process_model)
        else:
            return self._process_model

    @classmethod
    def _check_mixture_model(cls, mixture_model):

        if isinstance(mixture_model, GenericMixture):

            mm_class = mixture_model.__class__
            if mm_class in cls.compatible_mixture_models:
                return mixture_model

            raise ValueError('A mixture model of type %r cannot be used with'
                             ' this sampler' % mm_class.__name__)

        raise ValueError("'mixture_model' must be a compatible mixture model."
                         " Got mixture_model = %r" % mixture_model)

    @property
    def mixture_model(self):

        return self._mixture_model

    @mixture_model.setter
    def mixture_model(self, mixture_model):

        self._mixture_model = self._check_mixture_model(mixture_model)

    def _get_mixture_model(self, mixture_model):

        if mixture_model is not None:
            return self._check_mixture_model(mixture_model)
        else:
            return self._mixture_model

    @staticmethod
    def _check_max_iter(max_iter, warmup):

        if max_iter is not None:
            if not np.isscalar(max_iter):
                raise ValueError("Integer 'max_iter' must be a scalar.")

            max_iter = int(max_iter)

            if max_iter < 1:
                raise ValueError("Integer 'max_iter' must be larger than"
                                 " zero, but max_iter = %d" % max_iter)
        else:
            max_iter = 1000

        if warmup is not None:
            if not np.isscalar(warmup):
                raise ValueError("Integer 'warmup' must be a scalar.")

            warmup = int(warmup)

            if warmup < 0:
                raise ValueError("Integer 'warmup' must not be smaller than"
                                 " zero, but warmup = %d" % warmup)

            if not warmup < max_iter:
                raise ValueError("Integer 'warmup' must be smaller than"
                                 " 'max_iter', but warmup = %d" % warmup)
        else:
            warmup = max_iter / 2

        return max_iter, warmup

    @property
    def max_iter(self):

        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):

        self._max_iter, _ = self._check_max_iter(max_iter, self._warmup)

    def _get_max_iter(self, max_iter):

        if max_iter is not None:
            max_iter, _ = self._check_max_iter(max_iter, self._warmup)
            return max_iter
        else:
            return self._max_iter

    @property
    def warmup(self):

        return self._warmup

    @warmup.setter
    def warmup(self, warmup):

        _, self._warmup = self._check_max_iter(self._max_iter, warmup)

    def _get_warmup(self, warmup):

        if warmup is not None:
            _, warmup = self._check_max_iter(self._max_iter, warmup)
            return warmup
        else:
            return self._warmup

    @staticmethod
    def _check_examples(x_n):

        # TODO: Make this truly model-agnostic. Get rid of dtype=float
        x_n = np.asarray(x_n, dtype=float)

        if x_n.ndim == 0:
            x_n = x_n[np.newaxis, np.newaxis]
        elif x_n.ndim == 1:
            x_n = x_n[:, np.newaxis]
        elif x_n.ndim > 2:
            raise ValueError("'x_n' must be at most two-dimensional,"
                             " but x_n.ndim = %d" % x_n.ndim)

        return x_n.shape[0], x_n

    @staticmethod
    def _check_components(n, c_n):

        if c_n == None:
            c_n = np.zeros(n, dtype=int)
        else:
            c_n = np.asarray(c_n, dtype=int)
            if c_n.ndim == 0:
                c_n = c_n[np.newaxis]
            elif c_n.ndim > 1:
                raise ValueError("'c_n' must be at most one-dimensional,"
                                 " but c_n.ndim = %d" % c_n.ndim)

        if not c_n.shape == (n, ):
            raise ValueError("'c_n' has incompatible dimensions: should"
                             " be %s, got %s." % ((n,), c_n.shape))

        return c_n

    @staticmethod
    def _gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, m,
            random_state):
        """
        Performs a single iteration of Radford Neal's algorithms 3 or 8, see
        Neal (2000).
        """

        for i in range(n):

            prev_k = c_n[i]

            # Bookkeeping. Note that Neal's algorithms do not need inv_c to
            # work. It is used only in the split & merge algorithms
            if inv_c is not None:
                inv_c[prev_k].remove(i)

            # Downdate component counter
            n_c[prev_k] -= 1

            # Downdate model-dependent parameters
            mixture_params[prev_k].downdate(x_n[i])

            # If the previous component is empty after example i is removed,
            # recycle it and propose it as new component. If it is not empty,
            # we need to get a new component from the inactive_components set
            if n_c[prev_k] == 0:
                proposed_components = set([prev_k])
            else:
                proposed_components = set([inactive_components.pop()])
            for _ in range(1, m):
                proposed_components.add(inactive_components.pop())
            active_components |= proposed_components

            # Make sure the proposed components are not contaminated with
            # obsolete information
            for k in (proposed_components - set([prev_k])):
               mixture_params[k].iterate()

            # Initialize and populate the total log probability accumulator
            log_dist = np.empty(len(n_c), dtype=float)
            log_dist.fill(-np.inf)
            for k in active_components:
                # Calculate the process prior and mixture likelihood
                log_dist[k] = process_param.log_prior(n, n_c[k], m) + \
                        mixture_params[k].log_likelihood(x_n[i])

            # Sample from log_dist. Normalization is not required
            # TODO: Find a better way to sample
            cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
            r = random_state.uniform(size=1) * cdf[-1]
            [next_k] = cdf.searchsorted(r)

            c_n[i] = next_k

            # More bookkeeping
            if inv_c is not None:
                inv_c[next_k].add(i)

            # Update component counter
            n_c[next_k] += 1

            # Update model-dependent parameters
            mixture_params[next_k].update(x_n[i])

            # Cleanup
            proposed_components.discard(next_k)
            active_components -= proposed_components
            inactive_components |= proposed_components

    # TODO: infer has different arguments depending on whether it's a
    #       collapsed or non-collapsed sampler. Do something about that
    @abc.abstractmethod
    def infer(self, x_n, c_n, max_iter=1000, warmup=None, random_state=None):

        raise NotImplementedError
