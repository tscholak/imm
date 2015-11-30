# -*- coding: utf-8 -*-

"""
Non-collapsed samplers.
"""

import numpy as np
from collections import defaultdict

from .generic import GenericSampler
from ..models import ConjugateGaussianMixture, NonconjugateGaussianMixture
from ..models import DP, MFM


class NoncollapsedSampler(GenericSampler):
    """
    Class which encapsulates common functionality between all non-collapsed
    samplers. The Markov chain for these samplers consists of class indicators
    c_n and the latent class variables phi_n.
    """

    compatible_process_models = set()

    compatible_mixture_models = set()

    def __init__(self, process_model, m=10, max_iter=1000, warmup=None):

        super(NoncollapsedSampler, self).__init__(process_model, max_iter,
                warmup)

        self._m = self._check_m(m)

    @staticmethod
    def _check_m(m):

        if m is not None:
            if not np.isscalar(m):
                raise ValueError("Integer 'm' must be a scalar.")

            m = int(m)

            if m < 1:
                raise ValueError("Integer 'm' must be larger than"
                                 " zero, but m = %d" % m)
        else:
            m = 10

        return m

    def infer(self, x_n, c_n, m=10, max_iter=1000, warmup=None,
            random_state=None):
        """
        Component and latent variable inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        m : None or int, optional
            The number of auxiliary components
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

        max_iter = self._get_max_iter(max_iter)
        warmup = self._get_warmup(warmup)

        pm = self.process_model
        random_state = pm._get_random_state(random_state)
        process_param = pm.InferParam(pm, random_state)

        # TODO: Move into mixture model?
        n, x_n = self._check_examples(x_n)

        c_n = self._check_components(n, c_n)

        # Maximum number of components
        c_max = n + m - 1

        # Inverse mapping from components to examples
        # TODO: Only needed for split and merge samplers
        inv_c = defaultdict(set)
        for i in range(n):
            inv_c[c_n[i]].add(i)

        # Number of examples per component
        n_c = np.bincount(c_n, minlength=c_max)

        # active_components is an unordered set of unique components
        active_components = set(np.unique(c_n))
        # inactive_components is an unordered set of currently unassigned
        # components
        inactive_components = set(range(c_max)) - active_components

        # Initialize model-dependent parameters lazily
        mm = self.mixture_model
        mixture_params = [mm.InferParam(mm, random_state)
                for _ in range(c_max)]
        for k in active_components:
            mixture_params[k].iterate()
            # TODO: Substitute for inv_c?
            for i in inv_c[k]:
                mixture_params[k].update(x_n[i])

        c_n_samples = np.empty((max_iter-warmup)*n, dtype=int).reshape(
                (max_iter-warmup,n))
        # TODO: Use sparse array. scipy csr?
        # phi_c_samples = np.empty((max_iter-warmup)*cmax*???,
        #         dtype=float).reshape((max_iter-warmup,cmax)+???)

        for itn in range(max_iter):

            self._inference_step(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params, m,
                    random_state)

            if not itn-warmup < 0:
                c_n_samples[(itn-warmup,)] = c_n
                # for k in active_components:
                #     phi_c_samples[(itn-warmup,k)] = mixture_params[k].phi_c

        return c_n_samples
        # return c_n_samples, phi_c_samples


class GibbsSampler(NoncollapsedSampler):
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

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, m,
            random_state):

        for k in active_components:
            mixture_params[k].iterate()

        self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params, m,
                random_state)

        process_param.iterate(n, len(active_components))
