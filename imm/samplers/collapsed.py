# -*- coding: utf-8 -*-

"""
Collapsed samplers.
"""

import numpy as np
from collections import defaultdict

from .generic import GenericSampler
from ..models import ConjugateGaussianMixture
from ..models import DP, MFM


class CollapsedSampler(GenericSampler):
    """
    Class which encapsulates common functionality between all collapsed
    samplers. The Markov chain for these samplers consists only of class
    indicators c_n on a discrete state space.
    """

    compatible_process_models = set()

    compatible_mixture_models = set()

    def infer(self, x_n, c_n, max_iter=1000, warmup=None, random_state=None):
        """
        Component inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like
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

        max_iter = self._get_max_iter(max_iter)
        warmup = self._get_warmup(warmup)

        pm = self.process_model
        random_state = pm._get_random_state(random_state)
        process_param = pm.InferParam(pm, random_state)

        # TODO: Move into mixture model?
        n, x_n = self._check_examples(x_n)

        c_n = self._check_components(n, c_n)

        # Maximum number of components
        c_max = n

        # Inverse mapping from components to examples
        # TODO: Only needed for split and merge samplers
        inv_c = defaultdict(set)
        for i in range(n):
            inv_c[c_n[i]].add(i)

        # Number of examples per component
        n_c = np.bincount(c_n, minlength=c_max)

        # Active_components is an unordered set of unique components
        active_components = set(np.unique(c_n))
        # Inactive_components is an unordered set of currently unassigned
        # components
        inactive_components = set(range(c_max)) - active_components

        # Initialize model-dependent parameters lazily
        mm = self.mixture_model
        mixture_params = [mm.InferParam(mm, random_state)
                for _ in range(c_max)]
        for k in active_components:
            # TODO: Substitute for inv_c?
            for i in inv_c[k]:
                mixture_params[k].update(x_n[i])

        c_n_samples = np.empty((max_iter-warmup)*n, dtype=int).reshape(
                (max_iter-warmup,n))

        for itn in range(max_iter):

            self._inference_step(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params,
                    random_state)

            if not itn-warmup < 0:
                c_n_samples[(itn-warmup,)] = c_n

        return c_n_samples


class CollapsedGibbsSampler(CollapsedSampler):
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

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):

        self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params, 1,
                random_state)

        process_param.iterate(n, len(active_components))


class CollapsedMSSampler(CollapsedGibbsSampler):
    """
    Class which encapsulates common functionality between all collapsed
    merge-split samplers.
    """

    compatible_mixture_models = set()

    compatible_mixture_models = set()

    class Launch(object):

        def __init__(self, c, g, x_g, mixture_param):
            # Set the component
            self.c = c

            # The set inv_c will contain all examples that belong to the
            # component c
            self.inv_c = set([g])

            # Number of examples in the component c
            self.n_c = 1

            # Auxiliary, model-dependent parameters
            self.mixture_param = mixture_param.update(x_g)

        def update(self, g, x_g):
            # Add example g to component c
            self.inv_c.add(g)

            # Increment counter
            self.n_c += 1

            # Update model-dependent parameters
            self.mixture_param.update(x_g)

        def downdate(self, g, x_g):
            # Remove example g from component c
            self.inv_c.remove(g)

            # Reduce counter
            self.n_c -= 1

            # Downdate model-dependent parameters
            self.mixture_param.downdate(x_g)

    @staticmethod
    def _select_random_pair(n, random_state):
        """
        Select two distict observations (i.e. examples), i and j, uniformly
        at random
        """

        i, j = random_state.choice(n, 2, replace=False)

        return i, j

    @staticmethod
    def _find_common_components(c_n, inv_c, i, j):
        """
        Define a set of examples, S, that does not contain i or j, but all
        other examples that belong to the same component as i or j
        """

        if c_n[i] == c_n[j]:
            S = inv_c[c_n[i]] - set([i, j])
        else:
            S = (inv_c[c_n[i]] | inv_c[c_n[j]]) - set([i, j])

        return S

    @staticmethod
    def _log_likelihood_quotient(x_n, inv_c, n_c, mixture_param):
        """
        Logarithm of likelihood quotient. Note that the order in which the for
        loop is enumerated does not have an influence on the result.
        """

        llq = 0.0

        for index, l in enumerate(inv_c):
            llq += mixture_param.log_likelihood(x_n[l])
            if index < n_c-1:
                mixture_param.update(x_n[l])

        return llq

    def _attempt_split(self, n, x_n, c_n, inv_c, n_c, acc, launch_i, launch_j,
            active_components, inactive_components, process_param,
            mixture_params, random_state):

        mm = self.mixture_model

        # The probability q that the split proposal will be produced from the
        # launch state is in the denominator of the MH acceptance probability,
        # thus
        acc *= -1.0

        # Logarithm of prior quotient, see Eq. (3.4) in Jain & Neal (2004)
        acc += process_param.log_prior_quotient_pre(n,
                len(active_components), len(active_components)-1)
        acc += process_param.log_prior_quotient_post(launch_i.n_c)
        acc += process_param.log_prior_quotient_post(launch_j.n_c)
        acc -= process_param.log_prior_quotient_post(n_c[launch_j.c])

        # Logarithm of likelihood quotient
        acc += self._log_likelihood_quotient(x_n, launch_i.inv_c,
                launch_i.n_c, mm.InferParam(mm, random_state))
        acc += self._log_likelihood_quotient(x_n, launch_j.inv_c,
                launch_j.n_c, mm.InferParam(mm, random_state))
        acc -= self._log_likelihood_quotient(x_n, inv_c[launch_j.c],
                n_c[launch_j.c], mm.InferParam(mm, random_state))

        # Evaluate the split proposal by the MH acceptance probability
        if np.log(random_state.uniform()) < min(0.0, acc):
            # If the split proposal is accepted, then it becomes the next
            # state. At this point, launch_i.inv_c and launch_j.inv_c contain
            # the split proposal. Therefore, all labels are updated according
            # to the assignments in launch_i.inv_c and launch_j.inv_c
            c_n[list(launch_i.inv_c)] = launch_i.c
            c_n[list(launch_j.inv_c)] = launch_j.c
            # Update assignments in global component-example mapping
            inv_c[launch_i.c] = launch_i.inv_c
            inv_c[launch_j.c] = launch_j.inv_c
            # Update counts
            n_c[launch_i.c] = launch_i.n_c
            n_c[launch_j.c] = launch_j.n_c
            # Update mixture parameters
            for l in launch_i.inv_c:
                mixture_params[launch_i.c].update(x_n[l])
                mixture_params[launch_j.c].downdate(x_n[l])
        else:
            # If the split proposal is rejected, then the old state remains as
            # the next state. Thus, remove launch_i.c from the active
            # components and put it back into the inactive components (if
            # necessary)
            active_components.remove(launch_i.c)
            inactive_components.add(launch_i.c)

    def _attempt_merge(self, n, x_n, c_n, inv_c, n_c, acc, launch_i, launch_j,
            active_components, inactive_components, process_param,
            mixture_params, random_state):

        mm = self.mixture_model

        # Here we finalize the merge proposal by adding all those examples
        # that so far have been assigned to component launch_i.c = c_n[i] to
        # component launch_j.c = c_n[j]. Note that the remaining examples in S
        # are already in launch_j.c
        # TODO: Couldn't we have just done this in further above in
        #       _restricted_gibbs_scans when we instead decided to put every
        #       example l back to where it was in the beginning? Maybe, but
        #       this would have created larger differences between the
        #       implementations of the conjugate and the non-conjugate version
        #       of the algorithm. Anyway, what's done here is equivalent to
        #       what's outlined in Jain & Neal (2004)
        for l in inv_c[launch_i.c]:
            launch_j.update(l, x_n[l])

        # Logarithm of prior quotient, see Eq. (3.5) in Jain & Neal (2004)
        acc += process_param.log_prior_quotient_pre(n,
                len(active_components)-1, len(active_components))
        acc += process_param.log_prior_quotient_post(launch_j.n_c)
        acc -= process_param.log_prior_quotient_post(n_c[launch_i.c])
        acc -= process_param.log_prior_quotient_post(n_c[launch_j.c])

        # Logarithm of likelihood quotient
        acc += self._log_likelihood_quotient(x_n, launch_j.inv_c,
                launch_j.n_c, mm.InferParam(mm, random_state))
        acc -= self._log_likelihood_quotient(x_n, inv_c[launch_i.c],
                n_c[launch_i.c], mm.InferParam(mm, random_state))
        acc -= self._log_likelihood_quotient(x_n, inv_c[launch_j.c],
                n_c[launch_j.c], mm.InferParam(mm, random_state))

        # Evaluate the split proposal by the MH acceptance probability
        if np.log(random_state.uniform()) < min(0.0, acc):
            # If the merge proposal is accepted, then it becomes the next
            # state
            active_components.remove(launch_i.c)
            inactive_components.add(launch_i.c)
            # Assign all examples to component launch_j.c that in the proposal
            # were assigned to launch_j.c
            c_n[list(launch_j.inv_c)] = launch_j.c
            # Remove assignments to launch_i.c from global component-example
            # mapping
            inv_c[launch_i.c].clear()
            # Add assignments to launch_j.c to global component-example
            # mapping
            inv_c[launch_j.c] = launch_j.inv_c
            # Update counts
            n_c[launch_i.c] = 0
            n_c[launch_j.c] = launch_j.n_c
            # Update mixture parameters
            mixture_params[launch_i.c] = mm.InferParam(mm, random_state)
            for l in launch_j.inv_c:
                mixture_params[launch_j.c].update(x_n[l])
        else:
            # There is nothing to do if the merge proposal is rejected
            pass


class CollapsedRGMSSampler(CollapsedMSSampler):
    """
    Collapsed restricted Gibbs merge-split sampler for Dirichlet process
    mixture model.

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
    scheme : None or array-like, optional
        Computation scheme. Default is (5,1,1): 5 intermediate scans to reach
        the split launch state, 1 split-merge move per iteration, and 1
        incremental Gibbs scan per iteration
    """

    compatible_process_models = set([DP, MFM])

    compatible_mixture_models = set([ConjugateGaussianMixture])

    @staticmethod
    def _check_scheme(scheme):

        if scheme is None:
            max_intermediate_scans = 5
            max_split_merge_moves = 1
            max_gibbs_scans = 1
        else:
            scheme = np.asarray(scheme, dtype=int)
            if scheme.ndim == 0:
                max_intermediate_scans = np.asscalar(scheme)
                max_split_merge_moves = 1
                max_gibbs_scans = 1
            elif scheme.ndim == 1:
                max_intermediate_scans = scheme[0]
                try:
                    max_split_merge_moves = scheme[1]
                except:
                    max_split_merge_moves = 1
                try:
                    max_gibbs_scans = scheme[2]
                except:
                    max_gibbs_scans = 1
            elif scheme.ndim > 1:
                raise ValueError('Scheme must be an integer or tuple of'
                                 ' integers; thus must have dimension <= 1.'
                                 ' Got scheme.ndim = %s' % str(tuple(scheme)))

        if max_intermediate_scans < 1:
            raise ValueError('There must be at least one intermediate'
                             ' restricted Gibbs sampling scan; thus must have'
                             ' scheme[0] >= 1. Got scheme[0] ='
                             ' %s' % str(max_intermediate_scans))

        if max_split_merge_moves < 0:
            raise ValueError('The number of split-merge moves per iteration'
                             ' cannot be smaller than zero; thus must have'
                             ' scheme[1] >= 0. Got scheme[1] ='
                             ' %s' % str(max_split_merge_moves))

        if max_gibbs_scans < 0:
            raise ValueError('The number of Gibbs scans per iteration'
                             ' cannot be smaller than zero; thus must have'
                             ' scheme[2] >= 0. Got scheme[2] ='
                             ' %s' % str(max_gibbs_scans))

        return max_intermediate_scans, max_split_merge_moves, max_gibbs_scans

    def __init__(self, process_model, max_iter=1000, warmup=None,
            scheme=None):

        super(CollapsedRGMSSampler, self).__init__(
                process_model, max_iter, warmup)

        self.max_intermediate_scans, self.max_split_merge_moves, \
                self.max_gibbs_scans = self._check_scheme(scheme)

    def _init_launch_state(self, x_n, c_n, i, j, S, active_components,
            inactive_components, random_state):
        """
        Initialize the launch state that will be used to compute the
        restricted Gibbs sampling probabilities
        """

        mm = self.mixture_model
        Launch = self.Launch

        # launch_i.c is the initial launch state component of example i
        if c_n[i] == c_n[j]:
            # This will be a split proposal, so let launch_i.c be a new
            # component
            launch_i = Launch(inactive_components.pop(), i, x_n[i],
                    mm.InferParam(mm, random_state))
            active_components.add(launch_i.c)
        else:
            # This will be a merge proposal, so let launch_i.c be the current
            # component of i
            launch_i = Launch(c_n[i], i, x_n[i],
                    mm.InferParam(mm, random_state))

        # launch_j.c is the initial launch state component of example j
        launch_j = Launch(c_n[j], j, x_n[j], mm.InferParam(mm, random_state))

        # Randomly select the launch state components, independently and with
        # equal probability, for the examples in S
        for l in S:
            if random_state.uniform() < 0.5:
                launch_i.update(l, x_n[l])
            else:
                launch_j.update(l, x_n[l])

        return launch_i, launch_j

    @staticmethod
    def _restricted_gibbs_scans(n, x_n, c_n, i, j, S, launch_i, launch_j,
            process_param, max_intermediate_scans, random_state):
        """
        Modify the initial launch state by performing max_intermediate_scans
        intermediate restricted Gibbs sampling scans. The last scan in this
        loop leads to the proposal state
        """

        # Initialize acceptance probability aggregator
        acc = 0.0

        # Initialize total log probability accumulator. Since there are two
        # possibilities (either component launch_i.c or component launch_j.c),
        # this is a vector of length 2 (i.e., an "unnormalized 2-simplex")
        log_dist = np.empty(2, dtype=float)

        for scan in range(max_intermediate_scans):
            # These scans are restricted to the examples in S. We do not loop
            # over i and j; their launch state is kept fix!
            for l in S:
                # First, remove the current assignment of example l
                if l in launch_i.inv_c:
                    launch_i.downdate(l, x_n[l])
                else:
                    launch_j.downdate(l, x_n[l])

                # Then, calculate the full conditional log-probabilities.
                # First possibility: example l is in component launch_i.c.
                # Second possibility: example l is in component launch_j.c
                for index, launch in enumerate([launch_i, launch_j]):
                    # launch.n_c must never be zero!
                    # TODO: Make sure of that?
                    log_dist[index] = process_param.log_prior(n, launch.n_c,
                            1) + launch.mixture_param.log_likelihood(x_n[l])

                # Normalization
                log_dist_max = log_dist.max()
                log_dist -= log_dist_max + \
                        np.log((np.exp(log_dist - log_dist_max)).sum())

                if scan == max_intermediate_scans - 1 and c_n[i] != c_n[j]:
                    # In this case, launch_i.c = c_n[i]. Remember, we want to
                    # end up with a merge proposal, for which i, j, and all
                    # examples in S end up in the component launch_j.c.
                    # However, that proposal is not finalized here. Instead,
                    # we temporarilly put the examples back to where they were
                    # in the very beginning. Note that, no matter what has
                    # happened to example l during the intermediate restricted
                    # Gibbs sampling scans, in the end, the assignment is
                    # completely *deterministic*.
                    # The only way the intermediate sampling scans play a role
                    # in the merge case is via the acceptance log-probability
                    # acc. That log-probability is the result of all
                    # intermediate scans. This is the only reason we have to
                    # do these scans for the merge proposals
                    coin = 0 if c_n[l] == launch_i.c else 1
                else:
                    # Sample from the categorical log-distribution. There are
                    # only two possibilities (log_dist is an array of length
                    # 2) and thus coin is either 0 or 1
                    # TODO: Find a better way to sample
                    cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
                    r = random_state.uniform(size=1) * cdf[-1]
                    [coin] = cdf.searchsorted(r)

                if coin == 0:
                    launch_i.update(l, x_n[l])
                else:
                    launch_j.update(l, x_n[l])

                # The last iteration of restricted Gibbs sampling leads to the
                # split or merge proposal. We keep the corresponding log-
                # probability log_dist[coin], since it contributes to the
                # proposal density q in the MH acceptance log-probability acc
                if scan == max_intermediate_scans - 1:
                    acc += log_dist[coin]

        return acc

    def _conjugate_split_merge_iterate(self, n, x_n, c_n, inv_c, n_c,
            active_components, inactive_components, process_param,
            mixture_params, random_state):
        """
        Performs a single iteration of the Split-Merge MCMC procedure for the
        conjugate Dirichlet process mixture model, see Jain & Neal (2004).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j = self._init_launch_state(x_n, c_n, i, j, S,
                active_components, inactive_components, random_state)

        acc = self._restricted_gibbs_scans(n, x_n, c_n, i, j, S, launch_i,
                launch_j, process_param, self.max_intermediate_scans,
                random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(n, x_n, c_n, inv_c, n_c, acc, launch_i, launch_j,
                    active_components, inactive_components, process_param,
                    mixture_params, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(n, x_n, c_n, inv_c, n_c, acc, launch_i, launch_j,
                    active_components, inactive_components, process_param,
                    mixture_params, random_state)

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):

        for _ in range(self.max_split_merge_moves):

            self._conjugate_split_merge_iterate(n, x_n, c_n, inv_c, n_c,
                    active_components, inactive_components, process_param,
                    mixture_params, random_state)

        for _ in range(self.max_gibbs_scans):

            self._gibbs_iterate(n, x_n, c_n, inv_c, n_c,
                    active_components, inactive_components, process_param,
                    mixture_params, 1, random_state)

            process_param.iterate(n, len(active_components))


class CollapsedSAMSSampler(CollapsedMSSampler):
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

    def _sequential_allocation(self, n, x_n, c_n, i, j, S, active_components,
            inactive_components, process_param, random_state):
        """
        Proposes splits by sequentially allocating observations to one of two
        split components using allocation probabilities conditional on
        previously allocated data. Returns proposal densities for both splits
        and merges.
        """

        mm = self.mixture_model
        Launch = self.Launch

        if c_n[i] == c_n[j]:
            launch_i = Launch(inactive_components.pop(), i, x_n[i],
                    mm.InferParam(mm, random_state))
            active_components.add(launch_i.c)
        else:
            launch_i = Launch(c_n[i], i, x_n[i],
                    mm.InferParam(mm, random_state))

        launch_j = Launch(c_n[j], j, x_n[j], mm.InferParam(mm, random_state))

        acc = 0.0

        log_dist = np.empty(2, dtype=float)

        for l in random_state.permutation(list(S)):
            for index, launch in enumerate([launch_i, launch_j]):
                # launch.n_c must never be zero!
                # TODO: make sure of that
                log_dist[index] = process_param.log_prior(n, launch.n_c, 1) \
                        + launch.mixture_param.log_likelihood(x_n[l])

            log_dist_max = log_dist.max()
            log_dist -= log_dist_max + \
                    np.log((np.exp(log_dist - log_dist_max)).sum())

            if c_n[i] == c_n[j]:
                # TODO: Find a better way to sample
                cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
                r = random_state.uniform(size=1) * cdf[-1]
                [coin] = cdf.searchsorted(r)
            else:
                coin = 0 if c_n[l] == launch_i.c else 1

            if coin == 0:
                launch_i.update(l, x_n[l])
            else:
                launch_j.update(l, x_n[l])

            acc += log_dist[coin]

        return launch_i, launch_j, acc

    def _sams_iteration(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):
        """
        Performs a single iteration of the Sequentially-Allocated Merge-Split
        procedure for the conjugate Dirichlet process mixture model, see Dahl
        (2003).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j, acc = self._sequential_allocation(n, x_n, c_n, i,
                j, S, active_components, inactive_components, process_param,
                random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(n, x_n, c_n, inv_c, n_c, acc, launch_i,
                    launch_j, active_components, inactive_components,
                    process_param, mixture_params, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(n, x_n, c_n, inv_c, n_c, acc, launch_i,
                    launch_j, active_components, inactive_components,
                    process_param, mixture_params, random_state)

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):

        self._sams_iteration(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params,
                random_state)

        self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params, 1,
                random_state)

        process_param.iterate(n, len(active_components))
