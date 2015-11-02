# -*- coding: utf-8 -*-

import abc
import numpy as np
from scipy._lib._util import check_random_state


class MMBase(object):
    """
    Class which encapsulates common functionality between all mixture models.
    """
    __metaclass__  = abc.ABCMeta

    default_sampler = None

    def _check_sampler(self, sampler):
        if sampler is None:
            return self.default_sampler(self)
        from ..samplers import SamplerBase
        if issubclass(sampler, SamplerBase):
            return sampler(self)
        if isinstance(sampler, SamplerBase):
            if not sampler.mixture_model == self:
                raise ValueError('%r was initialized with a different mixture'
                                 ' model and is thus incompatible.' % sampler)
            return sampler
        raise ValueError("'sampler' must be a compatible sampler."
                         " Got sampler = %r" % sampler)

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

    def __init__(self, sampler=None, seed=None):
        super(MMBase, self).__init__()
        self._sampler = self._check_sampler(sampler)
        self._random_state = check_random_state(seed)

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

    # TODO: @abc.abstractclass ???
    class Param(object):
        def __init__(self, mixture_model):
            self.mixture_model = mixture_model

        def update(self, x):
            return self

        def downdate(self, x):
            return self

    @abc.abstractmethod
    def sample(self, size=1, random_state=None):
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood(self, x, param):
        raise NotImplementedError

    def inference(self, x_n, c_n=None, sampler=None, random_state=None):
        random_state = self._get_random_state(random_state)

        sampler = self._get_sampler(sampler)

        c_n = sampler.inference(x_n, c_n, random_state)

        return c_n
