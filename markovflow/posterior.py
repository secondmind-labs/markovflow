#
# Copyright (c) 2021 The Markovflow Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module containing posterior processes for GP models."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.likelihoods import Likelihood as GPF_Likelihood

from markovflow.base import SampleShape
from markovflow.conditionals import conditional_predict, conditional_statistics, pairwise_marginals
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import SDEKernel, StackKernel
from markovflow.likelihoods import Likelihood as MF_Likelihood
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.utils import tf_scope_class_decorator, tf_scope_fn_decorator

Likelihood = Union[MF_Likelihood, GPF_Likelihood]


@tf_scope_class_decorator
class PosteriorProcess(gpflow.Module, ABC):
    """
    Abstract class for forming a posterior process.

    Posteriors that extend this class must implement the :meth:`sample_state_trajectories`,
    :meth:`sample_f`, :meth:`predict_state` and :meth:`predict_f` methods.
    """

    def sample_state(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> tf.Tensor:
        """
        Generate joint state samples at `new_time_points`.

        :param new_time_points: Time points to generate sample trajectories for, with shape
            ``batch_shape + [num_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape (or
            number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

            This is an optional argument only passed in for inference with an importance-weighted
            posterior.
        :return: A tensor containing:

            * Sampled trajectories at new points, with shape
              ``sample_shape + batch_shape + [num_time_points, state_dim]``
            * Sampled trajectories at conditioning points, with shape
              ``sample_shape + batch_shape + [num_conditioning_points, state_dim]``
        """
        samples, _ = self.sample_state_trajectories(
            new_time_points, sample_shape, input_data=input_data
        )
        return samples

    @abstractmethod
    def sample_state_trajectories(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate joint sampled state trajectories evaluated both at `new_time_points`
        and at some points that we condition on for obtaining the posterior.

        :param new_time_points: Time points to generate sample trajectories for, with shape
            ``batch_shape + [num_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape (or
            number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

            This is an optional argument only passed in for inference with an importance-weighted
            posterior.
        :return: A tensor containing:

            * Sampled trajectories at new points, with shape
              ``sample_shape + batch_shape + [num_time_points, state_dim]``
            * Sampled trajectories at conditioning points, with shape
              ``sample_shape + batch_shape + [num_conditioning_points, state_dim]``
        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_f(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> tf.Tensor:
        """
        Generate joint function evaluation samples (projected states) at `new_time_points`.

        :param new_time_points: Time points to generate sample trajectories for, with shape
            ``batch_shape + [num_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape (or
            number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

            This is an optional argument only passed in for inference with an importance-weighted
            posterior.
        :return: A tensor containing sampled trajectories, with shape
            ``sample_shape + batch_shape + [num_time_points, output_dim]``.
        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_state(self, new_time_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict state at `new_time_points`. Note these time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_f(
        self, new_time_points: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict marginal function values at `new_time_points`. Note these
        time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        """
        raise NotImplementedError()


@tf_scope_class_decorator
class ConditionalProcess(PosteriorProcess):
    """
    Represents a posterior process indexed on the real line.

    This means :math:`q(s(.))` is built by combining the marginals :math:`q(s(Z))` and the
    conditional process :math:`p(s(.)|s(Z))` into:

    .. math:: q(s(.)) = ∫p(s(.)|s(Z))q(s(Z)) ds(Z)

    The marginals at discrete time inputs are available in closed form
    (see the :meth:`predict_f` method).

    It also includes methods for sampling from the posterior process.
    """

    def __init__(
        self,
        posterior_dist: GaussMarkovDistribution,
        kernel: SDEKernel,
        conditioning_time_points: tf.Tensor,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        :param posterior_dist: The posterior represented by a Gauss-Markov distribution used for
            inference. For variational models this is the model defined by the variational
            distribution.
        :param kernel: The kernel of the prior process.
        :param conditioning_time_points: The time points to condition on for inference, with shape
            ``batch_shape + [num_time_points]``.
        :param mean_function: The mean function of the process that is added to fs.
        """
        super().__init__(self.__class__.__name__)
        self.gauss_markov_model = posterior_dist
        self.kernel = kernel
        self.conditioning_time_points = conditioning_time_points

        if mean_function is None:
            obs_dim = 1 if isinstance(kernel, StackKernel) else kernel.output_dim
            mean_function = ZeroMeanFunction(obs_dim=obs_dim)
        self.mean_function = mean_function

    def predict_state(self, new_time_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict state at `new_time_points`. Note these time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``.
        """
        pw_mu, pw_cov = pairwise_marginals(
            dist=self.gauss_markov_model,
            initial_mean=self.kernel.initial_mean(self.gauss_markov_model.batch_shape),
            initial_covariance=self.kernel.initial_covariance(new_time_points[..., :1]),
        )

        return conditional_predict(
            new_time_points,
            self.conditioning_time_points,
            self.kernel,
            training_pairwise_means=pw_mu,
            training_pairwise_covariances=pw_cov,
        )

    def predict_f(
        self, new_time_points: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict marginal function values at `new_time_points`. Note these
        time points should be sorted.

        .. note:: `new_time_points` that are far outside the `self.conditioning_time_points`
           specified when instantiating the class will revert to the prior.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        emission_model = self.kernel.generate_emission_model(new_time_points)
        f_res, f_covs = emission_model.project_state_marginals_to_f(
            *self.predict_state(new_time_points), full_output_cov=full_output_cov
        )

        mean = self.mean_function(new_time_points)
        mean = _correct_mean_shape(mean, self.kernel)

        return f_res + mean, f_covs

    def sample_state_trajectories(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate joint state samples at `new_time_points` and
        the `self.conditioning_time_points` specified when instantiating the class.

        See Appendix 2 of
        `"Doubly Sparse Variational Gaussian Processes" <https://arxiv.org/abs/2001.05363>`_
        for a derivation.

        The following notation is used:

            * :math:`t` - a vector of new time points
            * :math:`z` - a vector of the conditioning time points
            * :math:`sₚ/uₚ` - prior state sample at :math:`t/z`
            * :math:`sₒ/Uₒ` - posterior state sample at :math:`t/z`
            * :math:`p(.)` - the prior
            * :math:`q(.)` - the posterior

        Jointly sample from the prior at new and conditioning points:

        .. math:: [sₚ, uₚ] ~ p(s([t, z]))

        And sample from the posterior at the conditioning points:

        .. math:: uₒ ~ q(s(z))

        A sample from the posterior state is given by:

        .. math:: sₒ = sₚ - E[s(t) | s(z) = uₒ - uₚ]

        Noting :math:`z₋,z₊`, for each new point :math:`tₖ` the points in :math:`z` closest to
        :math:`tₖ` and :math:`vₖ = [s(z₋),s(z₊)]` are:

        .. math:: E[s(tₖ)|s(z)] = E[s(tₖ)|vₖ] = Pₖ vₖ

        That is, the conditional mean is local; it only depends on the nearing
        conditioning states.

        :param new_time_points: Time points to generate sample trajectories for, with shape
            ``batch_shape + [num_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape of sampled
            trajectories to draw. This can be either an integer or a tuple/list of integers.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

            Note this argument will be ignored if your posterior is an
            :class:`AnalyticPosteriorProcess`.
        :return: A tensor containing:

            * Sampled trajectories at new points, with shape
              ``sample_shape + batch_shape + [num_time_points, state_dim]``
            * Sampled trajectories at conditioning points, with shape
              ``sample_shape + batch_shape + [num_conditioning_points, state_dim]``
        """
        # convert shape to TensorShape
        sample_tensorshape = tf.TensorShape(sample_shape)

        # tf.gather needs to know how many batch dims there will be. that's the
        # same as self.batch_dims. When sorting *samples*, add extra for the samples dimension.
        sample_dims = sample_tensorshape.ndims
        batch_dims = self.gauss_markov_model.batch_shape.ndims

        # join the new and conditioning time points
        joint_time_points = tf.concat([self.conditioning_time_points, new_time_points], axis=-1)
        sort_ind = tf.argsort(joint_time_points)
        sorted_joint_time_points = tf.gather(joint_time_points, sort_ind, batch_dims=batch_dims)

        # sample jointly, [sₚ, uₚ] ~ p(s([t, z]))
        sorted_joint_samples = self.kernel.state_space_model(sorted_joint_time_points).sample(
            sample_shape
        )

        # separate the samples back into new and conditioning samples.
        n_conditioning_points = tf.shape(self.conditioning_time_points)[-1]
        unsort_ind = tf.argsort(sort_ind)
        unsort_ind = tf.broadcast_to(
            unsort_ind, tf.concat([sample_tensorshape, tf.shape(unsort_ind)], axis=0)
        )
        joint_samples = tf.gather(
            sorted_joint_samples, unsort_ind, batch_dims=batch_dims + sample_dims, axis=-2
        )
        prior_conditioning_samples = joint_samples[..., :n_conditioning_points, :]
        prior_new_samples = joint_samples[..., n_conditioning_points:, :]

        # select the difference between the prior and posterior samples, at times adjacent to the
        # new time points from the conditioning time points.
        posterior_conditioning_samples = self.gauss_markov_model.sample(sample_shape)
        delta = prior_conditioning_samples - posterior_conditioning_samples

        # The first and last value of delta is evaluated at -inf/inf, but infinitely far away from
        # the data the posterior will revert back to the prior. As such the delta expectation
        # will revert to zero.
        zero_pad = tf.zeros_like(delta[..., :1, :])
        delta_augmented = tf.concat([zero_pad, delta, zero_pad], axis=-2)
        indices = tf.searchsorted(self.conditioning_time_points, new_time_points)
        indices = tf.broadcast_to(
            indices, tf.concat([sample_tensorshape, tf.shape(indices)], axis=0)
        )
        u_minus = tf.gather(delta_augmented, indices, axis=-2, batch_dims=batch_dims + sample_dims)
        u_plus = tf.gather(
            delta_augmented, indices + 1, axis=-2, batch_dims=batch_dims + sample_dims
        )
        v = tf.concat([u_minus, u_plus], axis=-1)

        # compute the conditional mean projection for all new time points
        P, _ = conditional_statistics(new_time_points, self.conditioning_time_points, self.kernel)

        # sₚ - P (uₒ - uₚ), uₒ
        new_samples = prior_new_samples - tf.matmul(P, v[..., None])[..., 0]
        return new_samples, posterior_conditioning_samples

    def sample_f(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> tf.Tensor:
        """
        Generate joint function evaluation samples (projected states) at `new_time_points`.

        :param new_time_points: Time points to generate sample trajectories for, with shape
            ``batch_shape + [num_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape
            (or number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

            Note this argument will be ignored if your posterior is an
            :class:`AnalyticPosteriorProcess`.
        :return: A tensor containing the sampled trajectories, with shape
            ``sample_shape + batch_shape + [num_time_points, output_dim]``.
        """
        state_samples = self.sample_state(new_time_points, sample_shape)
        emission_model = self.kernel.generate_emission_model(new_time_points)
        centered_samples = emission_model.project_state_to_f(state_samples)

        mean = self.mean_function(new_time_points)
        mean = _correct_mean_shape(mean, self.kernel)

        return mean + centered_samples


@tf_scope_class_decorator
class AnalyticPosteriorProcess(ConditionalProcess):
    """
    Represents the (approximate) posterior process of a GP model.

    It inherits the marginal prediction and sampling methods from the parent
    :class:`ConditionalProcess` class.

    It also includes a method to predict the observations (see :meth:`predict_y`).
    """

    def __init__(
        self,
        posterior_dist: GaussMarkovDistribution,
        kernel: SDEKernel,
        conditioning_time_points: tf.Tensor,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        :param posterior_dist: The posterior represented by a Gauss-Markov distribution used
            for inference. For variational models this is the model defined by the variational
            distribution.
        :param kernel: The kernel of the prior process.
        :param conditioning_time_points: The time points to condition on for inference,
            with shape ``batch_shape + [num_time_points]``.
        :param likelihood: Likelihood defining how to project from f-space to an observation.
        :param mean_function: The mean function of the process that is added to fs.
        """
        super().__init__(posterior_dist, kernel, conditioning_time_points, mean_function)
        self.likelihood = likelihood

    def predict_y(
        self, new_time_points: tf.Tensor, full_output_cov=False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict observation marginals at `new_time_points`. Note these
        time points should be sorted.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, output_dim]`` and either
            ``batch_shape + [num_new_time_points, output_dim, output_dim]`` or
            ``batch_shape + [num_new_time_points, output_dim]``.
        """
        if full_output_cov:
            msg = "gpflow likelihoods do not support projecting the full covariance"
            assert not isinstance(self.likelihood, GPF_Likelihood), msg

        return self.likelihood.predict_mean_and_var(
            *self.predict_f(new_time_points=new_time_points, full_output_cov=full_output_cov)
        )


@tf_scope_class_decorator
class ImportanceWeightedPosteriorProcess(PosteriorProcess):
    """
    Represents the approximate posterior process of a GP model.

    The approximate posterior process is inferred via importance-weighted variational inference.
    """

    def __init__(
        self,
        num_importance_samples: int,
        proposal_dist: GaussMarkovDistribution,
        kernel: SDEKernel,
        conditioning_time_points: tf.Tensor,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
    ):
        """
        :param num_importance_samples: The number of importance-weighted samples.
        :param proposal_dist: The proposal represented by a Gauss-Markov distribution,
            from which we draw samples. This is the model defined by the variational distribution.
        :param kernel: The kernel of the prior process.
        :param conditioning_time_points: Time points to condition on for inference, with shape
            ``batch_shape + [num_time_points]``.
        :param likelihood: Likelihood defining how to project from f-space to an observation.
        :param mean_function: The mean function of the process that is added to fs.
        """
        super().__init__(self.__class__.__name__)
        self.proposal_process = ConditionalProcess(
            proposal_dist, kernel, conditioning_time_points, mean_function
        )
        self.num_importance_samples = num_importance_samples
        self.likelihood = likelihood

    def _log_qu_density(self, samples_u: tf.Tensor, stop_gradient: bool = False):
        """
        Log density of the posterior process evaluated at the conditioning points.

        :param samples_u: State samples at the conditioning time points, with shape
            ``sample_shape + [num_conditioning_points, state_dim]``.
        :param stop_gradient: Whether to stop the gradient flow through the samples. It is useful
            to do so when optimising the proposal distribution with control variates for reduced
            variance.
        :return: log q(u) [num_samples]
        """
        if stop_gradient:
            dist_q = self.proposal_process.gauss_markov_model.create_non_trainable_copy()
        else:
            dist_q = self.proposal_process.gauss_markov_model
        log_qu = dist_q.log_pdf(samples_u)
        return log_qu

    def log_importance_weights(
        self,
        samples_s: tf.Tensor,
        samples_u: tf.Tensor,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        stop_gradient: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log-importance weights for some state samples.

        The importance weights are given by:

        .. math:: w = p(Y | s) p(s, u) / q(s, u)

        Because it is assumed that :math:`q(s | u) = p(s | u)`, the weights reduce to:

        .. math:: w = p(Y | s) p(u) / q(u)

        We evaluate this ratio for some tensors of `samples_s` and `samples_u`, which
        are assumed to have been drawn from :math:`q(s, u)`. To do this, `samples_s` are
        projected to :math:`f` before being passed to the likelihood object.

        :param samples_s: A tensor of samples drawn from :math:`p(s|u)`, with shape
            ``sample_shape + batch_shape + [num_data, state_dim]``.
        :param samples_u: A tensor of samples drawn from :math:`q(u)`, with shape
            ``sample_shape + batch_shape + [num_inducing, state_dim]``.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :param stop_gradient: Whether to call stop gradient on :math:`q(u)`. This is
            useful for control variate schemes.
        :return log_weights: A tensor with shape ``[sample_shape]``.
        """
        conditioning_time_points = self.proposal_process.conditioning_time_points
        dist_p = self.proposal_process.kernel.state_space_model(conditioning_time_points)

        # sample_shape + batch_shape
        log_pu = dist_p.log_pdf(samples_u)
        log_qu = self._log_qu_density(samples_u, stop_gradient=stop_gradient)

        emission_model = self.proposal_process.kernel.generate_emission_model(input_data[0])

        # sample_shape + batch_shape + [num_data, obs_dim]
        samples_f = emission_model.project_state_to_f(samples_s)

        # apply mean function
        mean = self.proposal_process.mean_function(input_data[0])
        samples_f = samples_f + _correct_mean_shape(mean, self.proposal_process.kernel)

        # sample_shape, sum out over the number of data points.
        log_lik = tf.reduce_sum(self.likelihood.log_prob(samples_f, input_data[1]), axis=-1)

        # sum out the batch_shape dims from log_pu and log_qu and add log_lik
        batch_shape = dist_p.batch_shape
        batch_shape_axes = list(np.arange(batch_shape.ndims) - batch_shape.ndims)
        log_weights = log_lik + tf.reduce_sum(log_pu - log_qu, batch_shape_axes)

        return log_weights

    def _iwvi_samples_and_weights(
        self,
        new_time_points: tf.Tensor,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        sample_shape: SampleShape,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Sample from q(states) indexed by new_time_points and compute the log weights associated.

        :param new_time_points: ordered time input where to sample with shape
                        batch_shape + [num_new_time_points]
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

        :param sample_shape: A :data:`~markovflow.base.SampleShape` that specifies how many
            samples to draw, with shape ``(..., num_importance_samples)``.
        :return: state samples from the posterior and the log-weights with shapes:
                        sample_shape + batch_shape + [num_new_time_points, state_dim]
                        sample_shape
                        sample_shape + batch_shape + [num_conditioning_points, state_dim]
        """
        # batch_shape + [num_data + num_new_time_points]
        all_time_points = tf.concat([input_data[0], new_time_points], axis=-1)
        # sample_shape + batch_shape + [num_data+num_new_time_points, state_dim],
        # sample_shape + batch_shape + [num_inducing_points, state_dim]
        samples_s, samples_u = self.proposal_process.sample_state_trajectories(
            all_time_points, sample_shape=sample_shape
        )

        num_new_time_points = new_time_points.shape[-1]
        # sample_shape + batch_shape + [num_new_time_points, state_dim]
        samples_s_new = samples_s[..., -num_new_time_points:, :]
        # sample_shape + batch_shape + [num_data, state_dim]
        samples_s_data = samples_s[..., :-num_new_time_points, :]
        # sample_shape
        log_weights = self.log_importance_weights(samples_s_data, samples_u, input_data)
        # sample_shape + batch_shape + [num_new_time_points, state_dim], sample_shape
        return samples_s_new, log_weights, samples_u

    def sample_state_trajectories(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Tuple[tf.Tensor, tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample the importance-weighted posterior over states.

        :param new_time_points: Ordered time input from which to sample, with shape
            ``batch_shape + [num_new_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape
            (or number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :return: The ordered samples states, with shape
            ``sample_shape + batch_shape + [num_new_time_points, state_dim]``.
        """
        if input_data is None:
            raise ValueError("You need to provide `input_data` for doing inference with IW")
        tf_sample_shape = tf.TensorShape(sample_shape) + (self.num_importance_samples,)
        sample_shape = tuple(tf.TensorShape(sample_shape)) + (self.num_importance_samples,)
        # sample_shape + [num_importance_samples] + batch_shape + [num_new_time_points, state_dim],
        # sample_shape + [num_importance_samples],
        # sample_shape + [num_importance_samples] + batch_shape + [num_inducing_points, state_dim]
        samples, log_weights, conditioning_samples = self._iwvi_samples_and_weights(
            new_time_points, input_data, sample_shape
        )

        # get appropriate shape to flatten the MC samples (not the importance samples)
        non_sampling_dims = tf.shape(samples)[tf_sample_shape.ndims :]
        num_mc_samples = tf_sample_shape[:-1].num_elements()
        flat_shape = tf.concat(
            [[num_mc_samples, self.num_importance_samples], non_sampling_dims], axis=0
        )
        reshaped_samples = tf.reshape(samples, flat_shape)

        # resample of the importance sample dimension, sample_shape + [1]:
        indices = tf.random.categorical(log_weights, 1)
        ext_indices = tf.concat([np.arange(num_mc_samples)[:, None], indices], axis=1)
        flat_samples = tf.gather_nd(reshaped_samples, ext_indices)

        # restore shape
        # sample_shape + batch_shape + [num_new_time_points, state_dim]
        samples = tf.reshape(
            flat_samples, tf.concat([tf_sample_shape[:-1], non_sampling_dims], axis=0)
        )

        return samples, conditioning_samples

    def sample_f(
        self,
        new_time_points: tf.Tensor,
        sample_shape: SampleShape,
        *,
        input_data: Tuple[tf.Tensor, tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Sample the importance-weighted (IWVI) posterior over functions.

        Note that to compute the expected value of some function under the iwvi posterior,
        it is likely to be more efficient to use :meth:`expected_value`.

        :param new_time_points: Ordered time input from which to sample, with shape
            ``batch_shape + [num_new_time_points]``.
        :param sample_shape: A :data:`~markovflow.base.SampleShape` that is the shape
            (or number of) sampled trajectories to draw.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :return: The ordered samples on latent functions, with shape
            ``[num_samples] + batch_shape + [num_new_time_points, num_outputs]``.
        """
        if input_data is None:
            raise ValueError("You need to provide `input_data` for doing inference with IW")
        samples_s = self.sample_state(new_time_points, sample_shape, input_data=input_data)
        emission_model = self.proposal_process.kernel.generate_emission_model(new_time_points)
        mean = self.proposal_process.mean_function(new_time_points)
        mean = _correct_mean_shape(mean, self.proposal_process.kernel)
        return emission_model.project_state_to_f(samples_s) + mean

    def expected_value(
        self, new_time_points: tf.Tensor, input_data: Tuple[tf.Tensor, tf.Tensor], func=tf.identity,
    ) -> tf.Tensor:
        """
        Compute the expected value of the function `func` acting on a random variable
        :math:`f`.

        :math:`f` is represented by a GP in this case, using importance sampling at the times
        given in `new_time_points`. That is:

        .. math:: ∫qₚ(f) func(f) df = Σₖ wₖ func(fₖ)

        ...where:

            * :math:`qₚ` is the importance-weighted approximate posterior distribution of :math:`f`
            * :math:`wₖ` are the importance weights

        For example, to compute the posterior mean we set `func = tf.identify`.

        :param new_time_points: Ordered time input from which to sample, with shape
            ``batch_shape + [num_new_time_points]``.
        :param input_data: A tuple of time points and observations containing the data:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :param func: The function to compute the expected value of. `func` should act
            on the last dimension of a tensor. That last dimension will have length as specified
            by the output_dim of the underlying emission model. The return shape of `func` need
            not be the same, but we expect all other dimensions to broadcast.
        :returns: A tensor with shape ``batch_shape + [num_new_time_points, output_dim]``.
        """
        # [num_importance_samples] + batch_shape + [num_new_time_points, state_dim],
        # [num_importance_samples]
        samples, log_weights, _ = self._iwvi_samples_and_weights(
            new_time_points, input_data, (self.num_importance_samples,)
        )
        weights = tf.nn.softmax(log_weights)  # [num_importance_samples]
        emission_model = self.proposal_process.kernel.generate_emission_model(new_time_points)

        # get the mean function and bring it to the right shape
        mean = self.proposal_process.mean_function(new_time_points)
        mean = _correct_mean_shape(mean, self.proposal_process.kernel)
        # [num_importance_samples] + batch_shape + [num_new_time_points, output_dim]
        Fs = emission_model.project_state_to_f(samples) + mean

        func_val = func(Fs)
        func_val_shape = tf.shape(func_val[0, ...])
        func_val_reshaped = tf.reshape(func_val, [self.num_importance_samples, -1])

        expected_f = tf.reduce_sum(weights[:, None] * func_val_reshaped, axis=0)
        return tf.reshape(expected_f, func_val_shape)

    def predict_state(self, new_time_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Not applicable to :class:`ImportanceWeightedPosteriorProcess`.
        The marginal state predictions are not available in closed form.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
        """
        raise NotImplementedError(
            "The marginal state predictions are not available in closed form."
        )

    def predict_f(
        self, new_time_points: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Not applicable to :class:`ImportanceWeightedPosteriorProcess`.
        The marginal function predictions are not available in closed form.

        :param new_time_points: Time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        """
        raise NotImplementedError(
            "The marginal function predictions are not available in closed form."
        )


@tf_scope_fn_decorator
def _correct_mean_shape(mean: tf.Tensor, kernel: SDEKernel) -> tf.Tensor:
    """
    Helper function that checks if the state space model is defined over a `StackKernel` so that it
    can bring the output of the mean function to the right shape.
    In any other case, the mean is returned unaltered.

    :param mean: the mean value that has been computed via a `MeanFunction`, with shape
                        batch_shape + [num_data, output_dim]
                     or batch_shape + [num_data, 1] in the case of a `StackKernel`,
                            where batch_shape[-1] = output_dim
    :param kernel: the corresponding kernel of the `GaussMarkovDistribution`

    :return: the mean value with the correct shape which is
            batch_shape + [num_data, output_dim] or
            batch_shape[:-1] + [num_data, output_dim] in the case of a `StackKernel` as the last
            dimension of the `batch_shape` is the output_dim.
    """
    if isinstance(kernel, StackKernel):
        # assert that the last dim of mean in the case of StackKernel is 1.
        tf.debugging.assert_equal(tf.shape(mean)[-1], 1)

        # bring project in the right shape: batch_shape[:-1] + [num_data, output_dim]
        mean = tf.linalg.matrix_transpose(mean[..., 0])
    return mean
