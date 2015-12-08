# -*- coding: utf-8 -*-

"""
Collapsed samplers.
"""

import numpy as np

from .generic import (GenericGibbsSampler, GenericRGMSSampler,
        GenericSAMSSampler)
from ..models import ConjugateGaussianMixture
from ..models import DP, MFM


class CollapsedGibbsSampler(GenericGibbsSampler):
    """
    Collapsed Gibbs sampler. The Markov chain for this sampler consists only
    of class indicators c_n on a discrete state space.

    Methods
    -------
    ``infer(x_n, c_n, max_iter, warmup, random_state)``
        Component inference.

    Parameters
    ----------
    process_model : compatible GenericProcess instance
        Compatible process model
    max_iter : None or int, optional
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    warmup : None or int, optional
        The number of warm-up iterations. The algorithm will discard the
        results of all iterations until this many iterations have elapsed.
        This must be non-negative and smaller than max_iter. Default is
        max_iter / 2
    """

    compatible_process_models = set([DP, MFM])

    compatible_mixture_models = set([ConjugateGaussianMixture])

    def __init__(self, process_model, max_iter=1000, warmup=None):

        super(CollapsedGibbsSampler, self).__init__(process_model, m=1,
                max_iter=max_iter, warmup=warmup)

    def infer(self, x_n, c_n=None, max_iter=None, warmup=None,
            random_state=None):
        """
        Component inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like, optional
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        max_iter : None or int, optional
            The maximum number of iterations
        warmup: None or int, optional
            The number of warm-up iterations
        random_state : np.random.RandomState instance, optional
            Used for drawing the random variates

        Returns
        -------
        c_n : ndarray
            Inferred component vectors
        """

        return super(CollapsedGibbsSampler, self).infer(x_n, c_n=c_n, m=1,
                max_iter=max_iter, warmup=warmup, random_state=random_state)


class CollapsedRGMSSampler(GenericRGMSSampler):
    """
    Collapsed restricted Gibbs merge-split sampler.

    Methods
    -------
    ``infer(x_n, c_n, scheme, max_iter, warmup, random_state)``
        Component inference.

    Parameters
    ----------
    process_model : compatible GenericProcess instance
        Compatible process model
    scheme : None or array-like, optional
        Computation scheme. Default is (5,1,1): 5 intermediate scans to reach
        the split launch state, 1 split-merge move per iteration, and 1
        incremental Gibbs scan per iteration
    max_iter : None or int, optional
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    warmup : None or int, optional
        The number of warm-up iterations. The algorithm will discard the
        results of all iterations until this many iterations have elapsed.
        This must be non-negative and smaller than max_iter. Default is
        max_iter / 2
    """

    compatible_process_models = set([DP, MFM])

    compatible_mixture_models = set([ConjugateGaussianMixture])

    def __init__(self, process_model, scheme=None, max_iter=1000,
            warmup=None):

        super(GenericRGMSSampler, self).__init__(process_model, m=1,
                max_iter=max_iter, warmup=warmup)

        self._max_intermediate_scans_split, self._max_split_merge_moves, \
                self._max_gibbs_scans, _ = self._check_scheme(scheme)
        self._max_intermediate_scans_merge = 0

    @property
    def scheme(self):

        return self._max_intermediate_scans_split, \
                self._max_split_merge_moves, self._max_gibbs_scans, \
                self._max_intermediate_scans_merge

    @scheme.setter
    def scheme(self, scheme):

        self._max_intermediate_scans_split, self._max_split_merge_moves, \
                self._max_gibbs_scans, _ = self._check_scheme(scheme)

    def _get_scheme(self, scheme):

        if scheme is not None:
            max_intermediate_scans_split, max_split_merge_moves, \
                    max_gibbs_scans, _ = self._check_scheme(scheme)
            return max_intermediate_scans_split, max_split_merge_moves, \
                    max_gibbs_scans, self._max_intermediate_scans_merge
        else:
            return self._max_intermediate_scans_split, \
                    self._max_split_merge_moves, self._max_gibbs_scans, \
                    self._max_intermediate_scans_merge

    def infer(self, x_n, c_n=None, scheme=None, max_iter=None, warmup=None,
            random_state=None):
        """
        Component inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like, optional
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        scheme: None or array-like, optional
            Computation scheme
        max_iter : None or int, optional
            The maximum number of iterations
        warmup: None or int, optional
            The number of warm-up iterations
        random_state : np.random.RandomState instance, optional
            Used for drawing the random variates

        Returns
        -------
        c_n : ndarray
            Inferred component vectors
        """

        return super(CollapsedRGMSSampler, self).infer(x_n, c_n=c_n, m=1,
                scheme=scheme, max_iter=max_iter, warmup=warmup,
                random_state=random_state)


class CollapsedSAMSSampler(GenericSAMSSampler):
    """
    Collapsed sequentially-allocated merge-split (SAMS) sampler.

    Methods
    -------
    ``infer(x_n, c_n, max_iter, warmup, random_state)``
        Component inference.

    Parameters
    ----------
    process_model : compatible GenericProcess instance
        Compatible process model
    max_iter : None or int, optional
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    warmup : None or int, optional
        The number of warm-up iterations. The algorithm will discard the
        results of all iterations until this many iterations have elapsed.
        This must be non-negative and smaller than max_iter. Default is
        max_iter / 2
    """

    compatible_process_models = set([DP, MFM])

    compatible_mixture_models = set([ConjugateGaussianMixture])

    def __init__(self, process_model, max_iter=1000, warmup=None):

        super(CollapsedSAMSSampler, self).__init__(process_model, m=1,
                max_iter=max_iter, warmup=warmup)

    def infer(self, x_n, c_n=None, max_iter=None, warmup=None,
            random_state=None):
        """
        Component inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like, optional
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        max_iter : None or int, optional
            The maximum number of iterations
        warmup: None or int, optional
            The number of warm-up iterations
        random_state : np.random.RandomState instance, optional
            Used for drawing the random variates

        Returns
        -------
        c_n : ndarray
            Inferred component vectors
        """

        return super(CollapsedSAMSSampler, self).infer(x_n, c_n=c_n, m=1,
                max_iter=max_iter, warmup=warmup, random_state=random_state)
