# -*- coding: utf-8 -*-

import abc
import numpy as np

from ..models.processes import GenericProcess
from ..models.mixtures import GenericMixture


class GenericSampler(object):
    """
    Class which encapsulates common functionality between all samplers.

    Parameters
    ----------
    mixture_model : compatible GenericProcess instance
        Compatible process model
    max_iter : int
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    """

    __metaclass__  = abc.ABCMeta

    compatible_process_models = set()

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

    compatible_mixture_models = set()

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

    def __init__(self, process_model, max_iter=1000, warmup=None):

        self._process_model = self._check_process_model(process_model)

        self._mixture_model = self._check_mixture_model(
            self._process_model._mixture_model)

        self._max_iter, self._warmup = self._check_max_iter(max_iter, warmup)

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

    @abc.abstractmethod
    def infer(self, x_n, c_n, max_iter=1000, warmup=None, random_state=None):

        raise NotImplementedError
