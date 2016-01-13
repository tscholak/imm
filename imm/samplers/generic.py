# -*- coding: utf-8 -*-

import abc
import numpy as np
from collections import defaultdict

from ..utils import _logsumexp
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


class GenericGibbsSampler(GenericSampler):
    """
    Class which encapsulates common functionality between all Gibbs samplers.
    """

    def __init__(self, process_model, m=10, max_iter=1000, warmup=None):

        super(GenericGibbsSampler, self).__init__(process_model,
                max_iter=max_iter, warmup=warmup)

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

    @property
    def m(self):

        return self._m

    @m.setter
    def m(self, m):

        self._m = self._check_m(m)

    def _get_m(self, m):

        if m is not None:
            return self._check_m(m)
        else:
            return self._m

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

            # Sample from log_dist. Normalization is required
            log_dist -= _logsumexp(len(n_c), log_dist)
            # TODO: Can we expect performance improvements if we exclude those
            #       elements of `log_dist` that are -inf?
            next_k = random_state.choice(a=len(n_c), p=np.exp(log_dist))

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

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, m,
            random_state):

        for k in active_components:
            mixture_params[k].iterate()

        self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params, m,
                random_state)

        process_param.iterate(n, len(active_components))

    def infer(self, x_n, c_n=None, m=None, max_iter=None, warmup=None,
            random_state=None):
        """
        Component and latent variable inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like, optional
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
        phi_c : ndarray
            Inferred latent variables
        """

        m = self._get_m(m)
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
        phi_c_samples = [{} for _ in range(max_iter-warmup)]

        for itn in range(max_iter):

            self._inference_step(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params, m,
                    random_state)

            if not itn-warmup < 0:
                c_n_samples[(itn-warmup,)] = c_n
                for k in active_components:
                    phi_c_samples[itn-warmup][k] = mixture_params[k].phi_c()

        return c_n_samples, phi_c_samples


class GenericMSSampler(GenericGibbsSampler):
    """
    Class which encapsulates common functionality between all merge-split (MS)
    samplers.
    """

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
            # TODO: A less ugly way to achieve parameter initialization
            mixture_param.iterate()
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

        i, j = random_state.choice(a=n, size=2, replace=False)

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

    def _attempt_split(self, n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
            launch_j, active_components, inactive_components, process_param,
            mixture_params, random_state):

        pm = self.process_model
        mm = self.mixture_model

        # Logarithm of prior quotient, see Eq. (3.4) in Jain & Neal (2004) and
        # Eq. (7) in Jain & Neal (2007)
        log_acc += pm._ms_log_prior_pre(n, len(active_components),
                len(active_components)-1, process_param)
        log_acc += pm._ms_log_prior_post(launch_i.n_c, process_param)
        log_acc += pm._ms_log_prior_post(launch_j.n_c, process_param)
        log_acc -= pm._ms_log_prior_post(n_c[launch_j.c], process_param)
        log_acc += mm._ms_log_prior(launch_i.mixture_param)
        log_acc += mm._ms_log_prior(launch_j.mixture_param)
        log_acc -= mm._ms_log_prior(mixture_params[launch_j.c])

        # Logarithm of likelihood quotient, see Eq. (3.8) in Jain & Neal
        # (2004) and Eq. (11) in Jain & Neal (2007)
        log_acc += mm._ms_log_likelihood(x_n, launch_i.inv_c,
                launch_i.mixture_param, random_state)
        log_acc += mm._ms_log_likelihood(x_n, launch_j.inv_c,
                launch_j.mixture_param, random_state)
        log_acc -= mm._ms_log_likelihood(x_n, inv_c[launch_j.c],
                mixture_params[launch_j.c], random_state)

        # Evaluate the split proposal by the MH acceptance probability
        if np.log(random_state.uniform()) < min(0.0, log_acc):

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
            mixture_params[launch_i.c] = launch_i.mixture_param
            mixture_params[launch_j.c] = launch_j.mixture_param

            # TODO: Logging
            # print "yay, accepted split with log-acc = {}".format(log_acc)

        else:

            # If the split proposal is rejected, then the old state remains as
            # the next state. Thus, remove launch_i.c from the active
            # components and put it back into the inactive components (if
            # necessary)
            active_components.remove(launch_i.c)
            inactive_components.add(launch_i.c)

            # TODO: Logging
            # print "nay, rejected split with log-acc = {}".format(log_acc)

    def _attempt_merge(self, n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
            launch_merge, active_components, inactive_components,
            process_param, mixture_params, random_state):

        pm = self.process_model
        mm = self.mixture_model

        # Logarithm of prior quotient, see Eq. (3.5) in Jain & Neal (2004) and
        # Eq. (8) in Jain & Neal (2007)
        log_acc += pm._ms_log_prior_pre(n, len(active_components),
                len(active_components)-1, process_param)
        log_acc += pm._ms_log_prior_post(launch_merge.n_c, process_param)
        log_acc -= pm._ms_log_prior_post(n_c[launch_i.c], process_param)
        log_acc -= pm._ms_log_prior_post(n_c[launch_merge.c], process_param)
        log_acc += mm._ms_log_prior(launch_merge.mixture_param)
        log_acc -= mm._ms_log_prior(mixture_params[launch_i.c])
        log_acc -= mm._ms_log_prior(mixture_params[launch_merge.c])

        # Logarithm of likelihood quotient, see Eq. (3.9) in Jain & Neal
        # (2004) and Eq. (12) in Jain & Neal (2007)
        log_acc += mm._ms_log_likelihood(x_n, launch_merge.inv_c,
                launch_merge.mixture_param, random_state)
        log_acc -= mm._ms_log_likelihood(x_n, inv_c[launch_i.c],
                mixture_params[launch_i.c], random_state)
        log_acc -= mm._ms_log_likelihood(x_n, inv_c[launch_merge.c],
                mixture_params[launch_merge.c], random_state)

        # Evaluate the split proposal by the MH acceptance probability
        if np.log(random_state.uniform()) < min(0.0, log_acc):

            # If the merge proposal is accepted, then it becomes the next
            # state
            active_components.remove(launch_i.c)
            inactive_components.add(launch_i.c)

            # Assign all examples to component launch_merge.c that in the
            # proposal were assigned to launch_merge.c
            c_n[list(launch_merge.inv_c)] = launch_merge.c

            # Remove assignments to launch_i.c from global component-example
            # mapping
            inv_c[launch_i.c].clear()
            # Add assignments to launch_merge.c to global component-example
            # mapping
            inv_c[launch_merge.c] = launch_merge.inv_c

            # Update counts
            n_c[launch_i.c] = 0
            n_c[launch_merge.c] = launch_merge.n_c

            # Update mixture parameters
            mixture_params[launch_i.c] = mm.InferParam(mm, random_state)
            mixture_params[launch_merge.c] = launch_merge.mixture_param

            # TODO: Logging
            #print "yay, accepted merge with log-acc = {}".format(log_acc)

        else:
            # There is nothing to do if the merge proposal is rejected
            pass

            # TODO: Logging
            # print "nay, rejected merge with log-acc = {}".format(log_acc)


class GenericRGMSSampler(GenericMSSampler):
    """
    Class which encapsulates common functionality between all restricted Gibbs
    merge-split (RGMS) samplers.
    """

    def __init__(self, process_model, m=10, scheme=None, max_iter=1000,
            warmup=None):

        super(GenericRGMSSampler, self).__init__(process_model, m=m,
                max_iter=max_iter, warmup=warmup)

        self._max_intermediate_scans_split, self._max_split_merge_moves, \
                self._max_gibbs_scans, self._max_intermediate_scans_merge = \
                        self._check_scheme(scheme)

    @staticmethod
    def _check_scheme(scheme):

        if scheme is None:
            max_intermediate_scans_split = 5
            max_split_merge_moves = 1
            max_gibbs_scans = 1
            max_intermediate_scans_merge = 5
        else:
            scheme = np.asarray(scheme, dtype=int)
            if scheme.ndim == 0:
                max_intermediate_scans_split = np.asscalar(scheme)
                max_split_merge_moves = 1
                max_gibbs_scans = 1
                max_intermediate_scans_merge = 5
            elif scheme.ndim == 1:
                max_intermediate_scans_split = scheme[0]
                try:
                    max_split_merge_moves = scheme[1]
                except:
                    max_split_merge_moves = 1
                try:
                    max_gibbs_scans = scheme[2]
                except:
                    max_gibbs_scans = 1
                try:
                    max_intermediate_scans_merge = scheme[3]
                except:
                    max_intermediate_scans_merge = 1
            elif scheme.ndim > 1:
                raise ValueError('Scheme must be an integer or tuple of'
                                 ' integers; thus must have dimension <= 1.'
                                 ' Got scheme.ndim = %s' % str(tuple(scheme)))

        if max_intermediate_scans_split < 1:
            raise ValueError('The sampler requires at least one intermediate'
                             ' restricted Gibbs sampling scan to reach the'
                             ' the split launch state; thus must have'
                             ' scheme[0] >= 1. Got scheme[0] ='
                             ' %s' % str(max_intermediate_scans_split))

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

        if max_intermediate_scans_merge < 1:
            raise ValueError('The sampler requires at least one intermediate'
                             ' restricted Gibbs sampling scan to reach the'
                             ' the merge launch state; thus must have'
                             ' scheme[3] >= 1. Got scheme[3] ='
                             ' %s' % str(max_intermediate_scans_merge))

        return max_intermediate_scans_split, max_split_merge_moves, \
                max_gibbs_scans, max_intermediate_scans_merge

    @property
    def scheme(self):

        return self._max_intermediate_scans_split, \
                self._max_split_merge_moves, self._max_gibbs_scans, \
                self._max_intermediate_scans_merge

    @scheme.setter
    def scheme(self, scheme):

        self._max_intermediate_scans_split, self._max_split_merge_moves, \
                self._max_gibbs_scans, self._max_intermediate_scans_merge = \
                        self._check_scheme(scheme)

    def _get_scheme(self, scheme):

        if scheme is not None:
            return self._check_scheme(scheme)
        else:
            return self._max_intermediate_scans_split, \
                    self._max_split_merge_moves, self._max_gibbs_scans, \
                    self._max_intermediate_scans_merge

    def _init_split_launch_state(self, x_n, c_n, i, j, S, active_components,
            inactive_components, random_state):
        """
        Initialize the split launch state that will be used to compute the
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

    def _init_merge_launch_state(self, x_n, c_n, i, j, S, random_state):
        """
        Initialize the merge launch state that will be used to compute the
        restricted Gibbs sampling probabilities
        """

        mm = self.mixture_model
        Launch = self.Launch

        # TODO: Should the model parameters of the merged component be set
        #       equal to the model parameters in the original component
        #       c_n[j]? According to Dahl (2005), they should. According to
        #       Jain & Neal (2007), they should not and instead be drawn from
        #       the prior distribution. Let's do the latter for now

        launch_merge = Launch(c_n[j], j, x_n[j],
                mm.InferParam(mm, random_state))

        for l in (S | set([i])):
            launch_merge.update(l, x_n[l])

        return launch_merge

    @staticmethod
    def _restricted_gibbs_scans(n, x_n, c_n, i, j, S, launch_i, launch_j,
            launch_merge, process_param, mixture_params,
            max_intermediate_scans_split, max_intermediate_scans_merge,
            random_state):
        """
        Modify the initial launch state by performing intermediate restricted
        Gibbs sampling scans. The last scan in this loop leads to the proposal
        state.
        """

        # Initialize acceptance probability aggregator
        log_acc = 0.0

        # Initialize a total log probability accumulator. Since there are
        # two possibilities (either component launch_i.c or launch_j.c),
        # this is a vector of length 2
        log_dist = np.empty(2, dtype=float)

        # Modify the split launch state by performing
        # `max_intermediate_scans_split` intermediate restricted Gibbs
        # sampling scans to update `launch_i` and `launch_j`. Then, conduct
        # one final restricted Gibbs sampling scan from the split launch
        # state.
        for scan in range(max_intermediate_scans_split+1):

            if scan == max_intermediate_scans_split:
                # The last iteration of restricted Gibbs sampling leads to
                # the split or merge proposal. We keep the corresponding
                # log-likelihood, since it contributes to the proposal
                # density in the M-H acceptance log-probability acc
                if c_n[i] == c_n[j]:
                    # This is a split and there won't be a merge proposal
                    log_acc -= launch_i.mixture_param.iterate(
                            compute_log_likelihood=True)
                    log_acc -= launch_j.mixture_param.iterate(
                            compute_log_likelihood=True)
                else:
                    # This is a merge and there won't be a split proposal.
                    # Reset component parameters to initial values
                    log_acc += launch_i.mixture_param.iterate_to(
                            mixture_params[launch_i.c],
                            compute_log_likelihood=True)
                    log_acc += launch_j.mixture_param.iterate_to(
                            mixture_params[launch_j.c],
                            compute_log_likelihood=True)
            else:
                launch_i.mixture_param.iterate()
                launch_j.mixture_param.iterate()

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
                log_dist -= _logsumexp(2, log_dist)

                if scan == max_intermediate_scans_split:
                    # The last iteration of restricted Gibbs sampling leads to
                    # the split or merge proposal. We keep the corresponding
                    # log-probability, since it contributes to the proposal
                    # density in the M-H acceptance log-probability acc
                    if c_n[i] == c_n[j]:
                        # This is a split and there won't be a merge proposal
                        index = random_state.choice(a=2, p=np.exp(log_dist))
                        log_acc -= log_dist[index]
                    else:
                        # This is a merge and there won't be a split proposal
                        index = 0 if c_n[l] == launch_i.c else 1
                        log_acc += log_dist[index]
                else:
                    index = random_state.choice(a=2, p=np.exp(log_dist))

                if index == 0:
                    launch_i.update(l, x_n[l])
                else:
                    launch_j.update(l, x_n[l])

        # Modify the merge launch state by performing
        # `max_intermediate_scans_merge` intermediate restricted Gibbs
        # sampling scans to update `launch_merge.mixture_param`. Then, conduct
        # one final restricted Gibbs sampling scan from the merge launch
        # state.
        for scan in range(max_intermediate_scans_merge+1):

            if scan == max_intermediate_scans_merge:
                # The last iteration of restricted Gibbs sampling leads to
                # the split or merge proposal. We keep the corresponding
                # log-likelihood, since it contributes to the proposal
                # density in the M-H acceptance log-probability acc
                if c_n[i] == c_n[j]:
                    # This is a split and there won't be a merge proposal.
                    # Reset component parameters to initial values
                    log_acc += launch_merge.mixture_param.iterate_to(
                            mixture_params[launch_merge.c],
                            compute_log_likelihood=True)
                else:
                    # This is a merge and there won't be a split proposal
                    log_acc -= launch_merge.mixture_param.iterate(
                            compute_log_likelihood=True)
            else:
                launch_merge.mixture_param.iterate()

        return log_acc

    def _merge_split_iterate(self, n, x_n, c_n, inv_c, n_c,
            active_components, inactive_components, process_param,
            mixture_params, max_intermediate_scans_split,
            max_intermediate_scans_merge, random_state):
        """
        Performs a single iteration of the Split-Merge MCMC procedure for the
        conjugate Dirichlet process mixture model, see Jain & Neal (2004).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j = self._init_split_launch_state(x_n, c_n, i, j, S,
                active_components, inactive_components, random_state)

        launch_merge = self._init_merge_launch_state(x_n, c_n, i, j, S,
                random_state)

        log_acc = self._restricted_gibbs_scans(n, x_n, c_n, i, j, S, launch_i,
                launch_j, launch_merge, process_param, mixture_params,
                max_intermediate_scans_split, max_intermediate_scans_merge,
                random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
                    launch_j, active_components, inactive_components,
                    process_param, mixture_params, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
                    launch_merge, active_components, inactive_components,
                    process_param, mixture_params, random_state)

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, m,
            max_intermediate_scans_split, max_split_merge_moves,
            max_gibbs_scans, max_intermediate_scans_merge, random_state):

        for _ in range(max_split_merge_moves):

            self._merge_split_iterate(n, x_n, c_n, inv_c, n_c,
                    active_components, inactive_components, process_param,
                    mixture_params, max_intermediate_scans_split,
                    max_intermediate_scans_merge, random_state)

        for _ in range(max_gibbs_scans):

            for k in active_components:
                mixture_params[k].iterate()

            self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params, m,
                    random_state)

            process_param.iterate(n, len(active_components))

    def infer(self, x_n, c_n=None, m=None, scheme=None, max_iter=None,
            warmup=None, random_state=None):
        """
        Component and latent variable inference.

        Parameters
        ----------
        x_n : array-like
            Examples
        c_n : None or array-like, optional
            Vector of component indicator variables. If None, then the
            examples will be assigned to the same component initially
        m : None or int, optional
            The number of auxiliary components
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
        phi_c : ndarray
            Inferred latent variables
        """

        m = self._get_m(m)
        max_intermediate_scans_split, max_split_merge_moves, \
                max_gibbs_scans, max_intermediate_scans_merge = \
                        self._get_scheme(scheme)
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
            mixture_params[k].iterate()

        c_n_samples = np.empty((max_iter-warmup)*n, dtype=int).reshape(
                (max_iter-warmup,n))
        phi_c_samples = [{} for _ in range(max_iter-warmup)]

        for itn in range(max_iter):

            self._inference_step(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params, m,
                    max_intermediate_scans_split, max_split_merge_moves,
                    max_gibbs_scans, max_intermediate_scans_merge,
                    random_state)

            if not itn-warmup < 0:
                c_n_samples[(itn-warmup,)] = c_n
                for k in active_components:
                    phi_c_samples[itn-warmup][k] = mixture_params[k].phi_c()

        return c_n_samples, phi_c_samples


class GenericSAMSSampler(GenericMSSampler):
    """
    Class which encapsulates common functionality between all
    sequentially-allocated merge-split (SAMS) samplers.
    """

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

        launch_merge = Launch(c_n[j], j, x_n[j],
                mm.InferParam(mm, random_state))

        log_acc = 0.0

        log_dist = np.empty(2, dtype=float)

        # TODO: Add code to sample component parameters (necessary for
        #       generalization to non-conjugate mixture models)

        for l in random_state.permutation(list(S)):

            for index, launch in enumerate([launch_i, launch_j]):
                # launch.n_c must never be zero!
                # TODO: Make sure of that?
                log_dist[index] = process_param.log_prior(n, launch.n_c, 1) \
                        + launch.mixture_param.log_likelihood(x_n[l])

            # Normalization
            log_dist -= _logsumexp(2, log_dist)

            if c_n[i] == c_n[j]:
                # This is a split and there won't be a merge proposal
                index = random_state.choice(a=2, p=np.exp(log_dist))
                log_acc -= log_dist[index]
            else:
                # This is a merge and there won't be a split proposal
                index = 0 if c_n[l] == launch_i.c else 1
                log_acc += log_dist[index]

            if index == 0:
                launch_i.update(l, x_n[l])
            else:
                launch_j.update(l, x_n[l])

        for l in (S | set([i])):
            launch_merge.update(l, x_n[l])

        return launch_i, launch_j, launch_merge, log_acc

    def _sams_iterate(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):
        """
        Performs a single iteration of the Sequentially-Allocated Merge-Split
        procedure for the conjugate Dirichlet process mixture model, see Dahl
        (2003).
        """

        mm = self.mixture_model

        i, j = self._select_random_pair(n, random_state)

        S = self._find_common_components(c_n, inv_c, i, j)

        launch_i, launch_j, launch_merge, log_acc = \
                self._sequential_allocation(n, x_n, c_n, i, j, S,
                        active_components, inactive_components, process_param,
                        random_state)

        # If i and j are in the same mixture component, then we attempt to
        # split
        if c_n[i] == c_n[j]:
            self._attempt_split(n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
                    launch_j, active_components, inactive_components,
                    process_param, mixture_params, random_state)
        # Otherwise, if i and j are in different mixture components, then we
        # attempt to merge
        else:
            self._attempt_merge(n, x_n, c_n, inv_c, n_c, log_acc, launch_i,
                    launch_merge, active_components, inactive_components,
                    process_param, mixture_params, random_state)

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, m,
            random_state):

        self._sams_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params,
                random_state)

        for k in active_components:
            mixture_params[k].iterate()

        self._gibbs_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params, m,
                random_state)

        process_param.iterate(n, len(active_components))


class GenericSliceSampler(GenericSampler):
    """
    Class which encapsulates common functionality between all slice samplers.
    """

    def __init__(self, process_model, max_iter=1000, warmup=None):

        super(GenericSliceSampler, self).__init__(process_model,
                max_iter=max_iter, warmup=warmup)

    @staticmethod
    def _slice_iterate(n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):

        # For each component `k`, sample component weights:
        dalpha = np.zeros(len(n_c), dtype=float)
        for k in active_components:
            dalpha[k] = n_c[k]
        new_k = inactive_components.pop()
        proposed_components = set([new_k])
        dalpha[new_k] = process_param.alpha
        beta = random_state.dirichlet(dalpha)
        mixture_params[new_k].iterate()
        beta_star = beta[new_k]

        # Sample slice variables and find the minimum
        u = random_state.uniform(size=n)
        for k in active_components:
            for i in inv_c[k]:
                u[i] *= beta[k]
        u_star = min(u)

        # Create new components through stick breaking until `beta_star` <
        # `u_star`
        while not beta_star < u_star:
            new_k = inactive_components.pop()
            proposed_components.add(new_k)
            nu = random_state.beta(1.0, process_param.alpha)
            beta[new_k] = beta_star * nu
            mixture_params[new_k].iterate()
            beta_star *= 1.0 - nu

        active_components |= proposed_components

        # For each observation `x_n[i]`, sample the component assignment
        # `c_n[i]`
        for i in range(n):

            # Bookkeeping: Downdate
            prev_k = c_n[i]
            inv_c[prev_k].remove(i)
            n_c[prev_k] -= 1
            mixture_params[prev_k].downdate(x_n[i])

            if n_c[prev_k] == 0:
                proposed_components.add(prev_k)

            # Initialize and populate the total log probability accumulator
            log_dist = np.empty(len(n_c), dtype=float)
            log_dist.fill(-np.inf)
            for k in active_components:
                if beta[k] > u[i]:
                    log_dist[k] = mixture_params[k].log_likelihood(x_n[i])

            # Sample from log_dist. Normalization is required
            log_dist -= _logsumexp(len(n_c), log_dist)
            # TODO: Can we expect performance improvements if we exclude those
            #       elements of `log_dist` that are -inf?
            next_k = random_state.choice(a=len(n_c), p=np.exp(log_dist))

            # Bookkeeping: Update
            c_n[i] = next_k
            inv_c[next_k].add(i)
            n_c[next_k] += 1
            mixture_params[next_k].update(x_n[i])

            proposed_components.discard(next_k)

        # Cleanup
        active_components -= proposed_components
        inactive_components |= proposed_components

    def _inference_step(self, n, x_n, c_n, inv_c, n_c, active_components,
            inactive_components, process_param, mixture_params, random_state):

        # For each active component `k`, sample component parameters
        for k in active_components:
            mixture_params[k].iterate()

        self._slice_iterate(n, x_n, c_n, inv_c, n_c, active_components,
                inactive_components, process_param, mixture_params,
                random_state)

        process_param.iterate(n, len(active_components))

    def infer(self, x_n, c_n=None, max_iter=None, warmup=None,
            random_state=None):
        """
        Component and latent variable inference.

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
        phi_c : ndarray
            Inferred latent variables
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
        phi_c_samples = [{} for _ in range(max_iter-warmup)]

        for itn in range(max_iter):

            self._inference_step(n, x_n, c_n, inv_c, n_c, active_components,
                    inactive_components, process_param, mixture_params,
                    random_state)

            if not itn-warmup < 0:
                c_n_samples[(itn-warmup,)] = c_n
                for k in active_components:
                    phi_c_samples[itn-warmup][k] = mixture_params[k].phi_c()

        return c_n_samples, phi_c_samples
