# -*- coding: utf-8 -*-

import abc
import numpy as np

from ..models.mmbase import MMBase


class SamplerBase(object):
    """
    Class which encapsulates common functionality between all samplers.

    Parameters
    ----------
    mixture_model : compatible MMBase instance
        Compatible mixture model
    max_iter : int
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    """
    __metaclass__  = abc.ABCMeta

    compatible_mixture_models = set()

    @classmethod
    def _check_mixture_model(cls, mixture_model):
        if isinstance(mixture_model, MMBase):
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
    def _process_max_iter(max_iter):
        if max_iter is None:
            max_iter = 1000
        else:
            max_iter = int(max_iter)
            if max_iter < 1:
                raise ValueError("Integer 'max_iter' must be larger than"
                                 " zero, but max_iter = %d" % max_iter)
        return max_iter

    def __init__(self, mixture_model, max_iter=None):
        self._mixture_model = self._check_mixture_model(mixture_model)
        self.max_iter = self._process_max_iter(max_iter)

    @staticmethod
    def _process_examples(x_n):
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
    def _process_components(n, c_n):
        if c_n == None:
            c_n = np.zeros(n, dtype=int)
        else:
            c_n = np.asarray(c_n, dtype=int)
            if c_n.ndim == 0:
                c_n = c_n[np.newaxis]
            elif c_n.ndim > 2:
                raise ValueError("'c_n' must be at most one-dimensional,"
                                 " but c_n.ndim = %d" % c_n.ndim)

        if not c_n.shape == (n, ):
            raise ValueError("'c_n' has incompatible dimensions: should"
                             " be %s, got %s." % ((n,), c_n.shape))

        return c_n

    @abc.abstractmethod
    def inference(self, x_n, c_n, random_state):
        raise NotImplementedError
