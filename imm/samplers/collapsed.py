# -*- coding: utf-8 -*-

"""
Collapsed samplers.
"""

import numpy as np
from scipy.special import gammaln
from collections import defaultdict

from .samplerbase import SamplerBase
from ..models import CDPGMM


class CollapsedSampler(SamplerBase):
    """
    Class which encapsulates common functionality between all collapsed
    samplers for the Dirichlet process mixture model.
    """

    compatible_mixture_models = set()

    def inference(self, x_n, c_n, random_state):
        """
        Component inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : array-like
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        random_state : np.random.RandomState instance
            Used for drawing the random variates

        Returns
        -------
        c_n : ndarray
            Inferred component vector
        """
        mm = self.mixture_model

        random_state = mm._get_random_state(random_state)

        alpha = mm.alpha

        n, x_n = self._process_examples(x_n)

        c_n = self._process_components(n, c_n)

        # Maximum number of components is number of examples
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

        # Initialize model-dependent parameters
        params = [mm.Param(mm) for _ in range(c_max)]
        for k in active_components:
            # TODO: Substitute for inv_c?
            for i in inv_c[k]:
                params[k].update(x_n[i])

        for itn in range(self.max_iter):
            alpha = self._inference_step(alpha, n, x_n, c_n, inv_c, n_c,
                    params, active_components, inactive_components,
                    random_state)

        return c_n


class CollapsedGibbsSampler(CollapsedSampler):
    """
    Collapsed Gibbs sampler for the Dirichlet process mixture model.

    Methods
    -------
    ``inference(x_n, c_n, random_state)``
        Component inference.

    Parameters
    ----------
    mixture_model : compatible MMBase instance
        Compatible mixture model
    max_iter : int
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    """

    compatible_mixture_models = set([CDPGMM])

    def _alpha_iteration(self, alpha, n, active_components, random_state):
        """
        Sample a new value for alpha.
        """

        mm = self.mixture_model

        if mm.a and mm.b:

            k = len(active_components)

            x = random_state.beta(alpha + 1.0, n)

            shape = mm.a + k - 1.0
            scale = mm.b - np.log(x)

            pi_x = shape / (shape + n * scale)

            alpha = pi_x * random_state.gamma(mm.a + k, scale)
            alpha += (1 - pi_x) * random_state.gamma(shape, scale)

        return alpha

    def _algorithm_3_iteration(self, alpha, n, x_n, c_n, inv_c, n_c, params,
            active_components, inactive_components, random_state):
        """
        Performs a single iteration of Radford Neal's Algorithm 3, see Neal
        (2000).
        """

        mm = self.mixture_model

        for i in range(n):
            prev_k = c_n[i]

            # Bookkeeping. Note that Algorithm 3 doesn't need inv_c to work.
            # It's used only in the split & merge algorithm
            if inv_c is not None:
                inv_c[prev_k].remove(i)

            # Downdate counters
            n_c[prev_k] -= 1

            # Downdate model-dependent parameters
            params[prev_k].downdate(x_n[i])

            # Initialize total log probability accumulator. The conditional
            # priors of all existing components are proportional to the number
            # of examples assigned to them. Note that np.log(0) = -inf
            np.seterr(divide='ignore')
            log_dist = np.log(n_c)
            np.seterr(divide='warn')

            # If the previous component is empty after example i is removed,
            # recycle it and propose it as new component. If it is not empty,
            # we need to get a new component from the inactive_components set
            prop_k = prev_k if n_c[prev_k] == 0 else inactive_components.pop()
            # If prop_k is already marked as active, the following won't have
            # any effect
            active_components.add(prop_k)
            # Since prop_k is a new component, its conditional prior is
            # proportional to alpha
            log_dist[prop_k] = np.log(alpha)

            # Calculate the likelihoods
            for k in active_components:
                log_dist[k] += mm.log_likelihood(x_n[i], params[k])

            # Sample from log_dist. Normalization is not required
            # TODO: Find a better way to sample
            cdf = np.cumsum(np.exp(log_dist - log_dist.max()))
            r = random_state.uniform(size=1) * cdf[-1]
            [next_k] = cdf.searchsorted(r)

            c_n[i] = next_k

            # More bookkeeping
            if inv_c is not None:
                inv_c[next_k].add(i)

            # Update counters
            n_c[next_k] += 1

            # Update model-dependent parameters
            params[next_k].update(x_n[i])

            # Cleanup
            if next_k != prop_k:
                active_components.remove(prop_k)
                inactive_components.add(prop_k)

    def _inference_step(self, alpha, n, x_n, c_n, inv_c, n_c, params,
            active_components, inactive_components, random_state):

        self._algorithm_3_iteration(alpha, n, x_n, c_n, inv_c, n_c, params,
                active_components, inactive_components, random_state)

        alpha = self._alpha_iteration(alpha, n, active_components,
                random_state)

        return alpha


class CollapsedMSSampler(CollapsedGibbsSampler):
    """
    Class which encapsulates common functionality between all collapsed
    merge-split sampler for the Dirichlet process mixture model.
    """

    compatible_mixture_models = set()

    class LaunchState(object):

        def __init__(self, c, g, x_g, param):
            # Set the component
            self.c = c

            # The set inv_c will contain all examples that belong to the
            # component c
            self.inv_c = set([g])

            # Number of examples in the component c
            self.n_c = 1

            # Auxiliary, model-dependent parameters
            self.param = param.update(x_g)

        def update(self, g, x_g):
            # Add example g to component c
            self.inv_c.add(g)

            # Increment counter
            self.n_c += 1

            # Update model-dependent parameters
            self.param.update(x_g)

        def downdate(self, g, x_g):
            # Remove example g from component c
            self.inv_c.remove(g)

            # Reduce counter
            self.n_c -= 1

            # Downdate model-dependent parameters
            self.param.downdate(x_g)

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
    def _log_likelihood_quotient(mm, x_n, inv_c, n_c):
        """
        Logarithm of likelihood quotient. Note that the order in which the for
        loop is enumerated does not have an influence on the result.
        """

        llq = 0.0

        param = mm.Param(mm)

        for index, l in enumerate(inv_c):
            llq += mm.log_likelihood(x_n[l], param)
            if index < n_c-1:
                param.update(x_n[l])

        return llq

    @classmethod
    def _attempt_split(cls, alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
            launch_i, launch_j, active_components, inactive_components,
            random_state):
        # The probability q that the split proposal will be produced from the
        # launch state is in the denominator of the MH acceptance probability,
        # thus
        acc *= -1.0

        # Logarithm of prior quotient, see Eq. (3.4) in Jain & Neal (2004)
        acc += np.log(alpha)
        acc -= gammaln(n_c[launch_j.c])
        acc += gammaln(launch_i.n_c)
        acc += gammaln(launch_j.n_c)

        # Logarithm of likelihood quotient
        acc -= cls._log_likelihood_quotient(
                mm, x_n, inv_c[launch_j.c], n_c[launch_j.c])
        acc += cls._log_likelihood_quotient(
                mm, x_n, launch_i.inv_c, launch_i.n_c)
        acc += cls._log_likelihood_quotient(
                mm, x_n, launch_j.inv_c, launch_j.n_c)

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
            # Update params
            for l in launch_i.inv_c:
                params[launch_i.c].update(x_n[l])
                params[launch_j.c].downdate(x_n[l])
        else:
            # If the split proposal is rejected, then the old state remains as
            # the next state. Thus, remove launch_i.c from the active
            # components and put it back into the inactive components (if
            # necessary)
            active_components.remove(launch_i.c)
            inactive_components.add(launch_i.c)

    @classmethod
    def _attempt_merge(cls, alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
            launch_i, launch_j, active_components, inactive_components,
            random_state):
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
        acc -= np.log(alpha)
        acc -= gammaln(n_c[launch_i.c])
        acc -= gammaln(n_c[launch_j.c])
        acc += gammaln(launch_j.n_c)

        # Logarithm of likelihood quotient
        acc -= cls._log_likelihood_quotient(
                mm, x_n, inv_c[launch_i.c], n_c[launch_i.c])
        acc -= cls._log_likelihood_quotient(
                mm, x_n, inv_c[launch_j.c], n_c[launch_j.c])
        acc += cls._log_likelihood_quotient(
                mm, x_n, launch_j.inv_c, launch_j.n_c)

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
            # Update params
            params[launch_i.c] = mm.Param(mm)
            for l in launch_j.inv_c:
                params[launch_j.c].update(x_n[l])
        else:
            # There is nothing to do if the merge proposal is rejected
            pass


class CollapsedRGMSSampler(CollapsedMSSampler):
    """
    Collapsed restricted Gibbs merge-split sampler for Dirichlet process
    mixture model.

    Methods
    -------
    ``inference(x_n, c_n, random_state)``
        Component inference.

    Parameters
    ----------
    mixture_model : compatible MMBase instance
        Compatible mixture model
    max_iter : int
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    scheme : array-like
        Computation scheme. Default is (5,1,1): 5 intermediate scans to reach
        the split launch state, 1 split-merge move per iteration, and 1
        incremental Gibbs scan per iteration
    """

    compatible_mixture_models = set([CDPGMM])

    @staticmethod
    def _process_scheme(scheme):

        if scheme is None:
            max_intermediate_scans = 5
            max_split_merge_moves = 1
            max_Gibbs_scans = 1
        else:
            scheme = np.asarray(scheme, dtype=int)
            if scheme.ndim == 0:
                max_intermediate_scans = np.asscalar(scheme)
                max_split_merge_moves = 1
                max_Gibbs_scans = 1
            elif scheme.ndim == 1:
                max_intermediate_scans = scheme[0]
                try:
                    max_split_merge_moves = scheme[1]
                except:
                    max_split_merge_moves = 1
                try:
                    max_Gibbs_scans = scheme[2]
                except:
                    max_Gibbs_scans = 1
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

        if max_Gibbs_scans < 0:
            raise ValueError('The number of Gibbs scans per iteration'
                             ' cannot be smaller than zero; thus must have'
                             ' scheme[2] >= 0. Got scheme[2] ='
                             ' %s' % str(max_Gibbs_scans))

        return max_intermediate_scans, max_split_merge_moves, max_Gibbs_scans

    def __init__(self, mixture_model, max_iter=None, scheme=None):

        super(CollapsedRGMSSampler, self).__init__(
                mixture_model, max_iter)

        self.max_intermediate_scans, self.max_split_merge_moves, \
                self.max_Gibbs_scans = self._process_scheme(scheme)

    @classmethod
    def _init_launch_state(cls, mm, x_n, c_n, i, j, S, active_components,
            inactive_components, random_state):
        """
        Initialize the launch state that will be used to compute the
        restricted Gibbs sampling probabilities
        """

        # launch_i.c is the initial launch state component of example i
        if c_n[i] == c_n[j]:
            # This will be a split proposal, so let launch_i.c be a new
            # component
            launch_i = cls.LaunchState(
                    inactive_components.pop(), i, x_n[i], mm.Param(mm))
            active_components.add(launch_i.c)
        else:
            # This will be a merge proposal, so let launch_i.c be the current
            # component of i
            launch_i = cls.LaunchState(c_n[i], i, x_n[i], mm.Param(mm))

        # launch_j.c is the initial launch state component of example j
        launch_j = cls.LaunchState(c_n[j], j, x_n[j], mm.Param(mm))

        # Randomly select the launch state components, independently and with
        # equal probability, for the examples in S
        for l in S:
            if random_state.uniform() < 0.5:
                launch_i.update(l, x_n[l])
            else:
                launch_j.update(l, x_n[l])

        return launch_i, launch_j

    @staticmethod
    def _restricted_gibbs_scans(mm, x_n, c_n, i, j, S, launch_i, launch_j,
            max_intermediate_scans, random_state):
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
                    log_dist[index] = np.log(launch.n_c) \
                            + mm.log_likelihood(x_n[l], launch.param)

                # Normalization
                log_dist -= log_dist.max() \
                        + np.log((np.exp(log_dist - log_dist.max())).sum())

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

    def _conjugate_split_merge_iteration(self, alpha, n, x_n, c_n, inv_c, n_c,
            params, active_components, inactive_components, random_state):
        """
        Performs a single iteration of the Split-Merge MCMC procedure for the
        conjugate Dirichlet process mixture model, see Jain & Neal (2004).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j = self._init_launch_state(mm, x_n, c_n, i, j, S,
                active_components, inactive_components, random_state)

        acc = self._restricted_gibbs_scans(mm, x_n, c_n, i, j, S, launch_i,
                launch_j, self.max_intermediate_scans, random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
                    launch_i, launch_j, active_components,
                    inactive_components, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
                    launch_i, launch_j, active_components,
                    inactive_components, random_state)

    def _inference_step(self, alpha, n, x_n, c_n, inv_c, n_c, params,
            active_components, inactive_components, random_state):

        for _ in range(self.max_split_merge_moves):

            self._conjugate_split_merge_iteration(alpha, n, x_n, c_n, inv_c,
                    n_c, params, active_components, inactive_components,
                    random_state)

        for _ in range(self.max_Gibbs_scans):

            self._algorithm_3_iteration(alpha, n, x_n, c_n, inv_c, n_c,
                    params, active_components, inactive_components,
                    random_state)

            alpha = self._alpha_iteration(alpha, n, active_components,
                    random_state)

        return alpha


class CollapsedSAMSSampler(CollapsedMSSampler):
    """
    Collapsed sequentially-allocated merge-split (SAMS) sampler for Dirichlet
    process mixture model.

    Methods
    -------
    ``inference(x_n, c_n, random_state)``
        Component inference.

    Parameters
    ----------
    mixture_model : compatible MMBase instance
        Compatible mixture model
    max_iter : int
        The maximum number of iterations. The algorithm will be terminated
        once this many iterations have elapsed. This must be greater than 0.
        Default is 1000
    """

    compatible_mixture_models = set([CDPGMM])

    @classmethod
    def _sequential_allocation(cls, mm, x_n, c_n, i, j, S, active_components,
            inactive_components, random_state):
        """
        Proposes splits by sequentially allocating observations to one of two
        split components using allocation probabilities conditional on
        previously allocated data. Returns proposal densities for both splits
        and merges.
        """

        if c_n[i] == c_n[j]:
            launch_i = cls.LaunchState(
                    inactive_components.pop(), i, x_n[i], mm.Param(mm))
            active_components.add(launch_i.c)
        else:
            launch_i = cls.LaunchState(c_n[i], i, x_n[i], mm.Param(mm))

        launch_j = cls.LaunchState(c_n[j], j, x_n[j], mm.Param(mm))

        acc = 0.0

        log_dist = np.empty(2, dtype=float)

        for l in random_state.permutation(list(S)):
            for index, launch in enumerate([launch_i, launch_j]):
                log_dist[index] = np.log(launch.n_c) \
                        + mm.log_likelihood(x_n[l], launch.param)

            log_dist -= log_dist.max() \
                    + np.log((np.exp(log_dist - log_dist.max())).sum())

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

    def _sams_iteration(self, alpha, n, x_n, c_n, inv_c, n_c, params,
            active_components, inactive_components, random_state):
        """
        Performs a single iteration of the Sequentially-Allocated Merge-Split
        procedure for the conjugate Dirichlet process mixture model, see Dahl
        (2003).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j, acc = self._sequential_allocation(mm, x_n, c_n,
                i, j, S, active_components, inactive_components, random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
                    launch_i, launch_j, active_components,
                    inactive_components, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(alpha, mm, x_n, c_n, inv_c, n_c, params, acc,
                    launch_i, launch_j, active_components,
                    inactive_components, random_state)

    def _inference_step(self, alpha, n, x_n, c_n, inv_c, n_c, params,
            active_components, inactive_components, random_state):

        self._sams_iteration(alpha, n, x_n, c_n, inv_c, n_c, params,
                active_components, inactive_components, random_state)

        self._algorithm_3_iteration(alpha, n, x_n, c_n, inv_c, n_c, params,
                active_components, inactive_components, random_state)

        alpha = self._alpha_iteration(alpha, n, active_components,
                random_state)

        return alpha
