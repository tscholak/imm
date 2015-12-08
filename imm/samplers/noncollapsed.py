# -*- coding: utf-8 -*-

"""
Non-collapsed samplers.
"""

import numpy as np

from .generic import (GenericGibbsSampler, GenericRGMSSampler,
        GenericSAMSSampler)
from ..models import ConjugateGaussianMixture, NonconjugateGaussianMixture
from ..models import DP, MFM


class GibbsSampler(GenericGibbsSampler):
    """
    Gibbs sampler.

    Methods
    -------
    ``infer(x_n, c_n, m, max_iter, warmup, random_state)``
        Component and latent variable inference.

    Parameters
    ----------
    process_model : compatible GenericProcess instance
        Compatible process model
    m : None or int, optional
        The number of auxiliary components. This must be larger than 0.
        Default is 10
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

    compatible_mixture_models = set([ConjugateGaussianMixture,
                                     NonconjugateGaussianMixture])


class RGMSSampler(GenericRGMSSampler):
    """
    Restricted Gibbs merge-split sampler.

    Methods
    -------
    ``infer(x_n, c_n, m, scheme, max_iter, warmup, random_state)``
        Component inference.

    Parameters
    ----------
    process_model : compatible GenericProcess instance
        Compatible process model
    m : None or int, optional
        The number of auxiliary components. This must be larger than 0.
        Default is 10
    scheme : None or array-like, optional
        Computation scheme. Default is (5,1,1,5): 5 intermediate scans to
        reach the split launch state, 1 split-merge move per iteration, 1
        incremental Gibbs scan per iteration, and 5 intermediate moves to
        reach the merge launch state
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

    compatible_mixture_models = set([ConjugateGaussianMixture,
                                     NonconjugateGaussianMixture])
