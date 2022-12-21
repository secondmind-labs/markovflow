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
"""Module containing mean functions."""
import abc

import tensorflow as tf
import gpflow

from markovflow.block_tri_diag import LowerTriangularBlockTriDiagonal
from markovflow.kernels import SDEKernel
from markovflow.utils import tf_scope_class_decorator, to_delta_time


@tf_scope_class_decorator
class MeanFunction(gpflow.Module, abc.ABC):
    """
    Abstract class for mean functions.

    Represents the action :math:`u(t)` added to the latent states:

    .. math:: dx(t)/dt = F x(t) + u(t) + L w(t)

    ...resulting in the the mean function:

    .. math:: dμ(t)/dt = F μ(t) + u(t)

    We can then solve the pure SDE:

    .. math:: dg(t)/dt = F g(t)+ L w(t)

    ...where:

    .. math:: g(t) = x(t) - μ(t)

    This class provides a very general interface for the function :math:`μ(t)`.

    .. note:: Implementations of this class should typically avoid performing computation
       in their `__init__` method. Performing computation in the constructor conflicts with
       running in TensorFlow's eager mode.
    """

    def __call__(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Return the mean function evaluated at the given time points.

        :param time_points: A tensor with shape ``[... num_data]``.
        :return: The mean function evaluated at the time points, with shape
            ``[... num_data, obs_dim]``.
        """


@tf_scope_class_decorator
class ZeroMeanFunction(MeanFunction):
    """
    Represents a mean function that is zero everywhere.
    """

    def __init__(self, obs_dim: int):
        """
        :param obs_dim: The dimension of the zeros to output.
        """
        super().__init__(self.__class__.__name__)
        self._obs_dim = obs_dim

    def __call__(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Return the mean function evaluated at the given time points.

        :param time_points: A tensor with shape ``[... num_data]``.
        :return: The mean function evaluated at the time points, with shape
            ``[... num_data, obs_dim]``.
        """
        shape = tf.concat([tf.shape(time_points), [self._obs_dim]], axis=0)
        return tf.zeros(shape, dtype=time_points.dtype)


@tf_scope_class_decorator
class LinearMeanFunction(MeanFunction):
    """
    Represents a mean function that is linear. That is, where :math:`m(t) = a * t`.
    """

    def __init__(self, coefficient: float, obs_dim: int = 1):
        """
        :param coefficient: The linear coefficient.
        :param obs_dim: The output dimension of the mean function.
        """
        super().__init__(self.__class__.__name__)
        self._coefficient = coefficient
        self._obs_dim = obs_dim

    def __call__(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Return the mean function evaluated at the given time points.

        :param time_points: A tensor with shape ``[... num_data]``.
        :return: The mean function evaluated at the time points
            with shape ``[... num_data, obs_dim]``.
        """
        shape = tf.concat([tf.shape(time_points), [self._obs_dim]], axis=0)
        return tf.broadcast_to(self._coefficient * time_points[..., None], shape)


@tf_scope_class_decorator
class ImpulseMeanFunction(MeanFunction):
    r"""
    Represents a mean function that is an impulse action. This is given by:

    .. math:: u(t) = Σₖ uₖ δ(t - tₖ)

    ...in:

    .. math:: dx(t)/dt = F x(t) + u(t) + L w(t)

    ...and then:

    .. math::
        &μ_(t) = 0                               &     t ≤ t₀\\
        &μ₀(t) = exp(F (t - t₀)) u₀              &t₀ < t ≤ t₁\\
        &μₖ(t) = exp(F (t - tₖ))(μₖ₋₁(tₖ) + uₖ)  &tₖ < t ≤ tₖ₊₁

    Or:

    .. math::
        &μ₋₁(t) = 0                   &     t ≤ t₀\\
        &μₖ(t) = exp(F (t - tₖ))aₖ    &tₖ < t ≤ tₖ₊₁

    If we let:

    .. math::
        &a₋₁ = 0\\
        &a₀ =  u₀\\
        &aₖ = Aₖaₖ₋₁ + uₖ

    ...where:

    .. math:: Aₖ = exp(F (tₖ - tₖ₋₁))

    ...then we can write this as a :class:`LowerTriangularBlockTriDiagonal` equation::

        [ I             ] a₀   u₀
        [-A₁, I         ] a₁   u₁
        [    -A₂, I     ] a₂ = u₂
        [         ᨞  ᨞  ] ⋮     ⋮
        [         -Aₙ, I] aₙ   uₙ

    We can then determine the :math:`aₖ` using a matrix solve.

    .. note:: The effect of the action is not seen until fractionally after it is applied. That is,
       if an impulse is applied at time :math:`t`, :math:`μ(t)` will not see the effect
       but :math:`μ(t + ε)` will.
    """

    def __init__(
        self, action_times: tf.Tensor, state_perturbations: tf.Tensor, kernel: SDEKernel
    ) -> None:
        """
        :param action_times: The times at which actions occur, with shape ``[... num_actions]``.
        :param state_perturbations: The magnitude of the impulse, with shape
            ``[... num_actions, state_dim]``.
        :param kernel: The kernel that is used to generate this mean function.
        """
        super().__init__(self.__class__.__name__)
        self._state_perturbations = state_perturbations
        self._batch_shape = state_perturbations.shape[:-2]
        # TODO: make this use tf.shape, must replace if statements with tf.cond
        self._num_actions = state_perturbations.shape[-2]
        self._state_dim = state_perturbations.shape[-1]
        tf.debugging.assert_equal(tf.shape(action_times), tf.shape(state_perturbations)[:-1])

        # we don't handle placeholders
        assert self._num_actions is not None

        self._action_times_without_initial = action_times

        self._kernel = kernel
        self._impulses = state_perturbations

        # add dummy time point before the first for the initial zero function [... num_actions + 1]
        self._action_times = tf.concat([action_times[..., :1] - 1e-6, action_times], axis=-1)

    @property
    def _a_k(self):
        # [... num_actions, state_dim, state_dim]

        shape = tf.concat([self._batch_shape, [self._num_actions]], axis=0)
        identities = tf.eye(
            self._state_dim, dtype=self._state_perturbations.dtype, batch_shape=shape
        )

        if self._num_actions == 1:
            # No A_s if there is only one action
            A_s = None
        else:
            # [-exp(F (t₁ - t₀)), -exp(F (t₂ - t₁)) ... ] [... num_actions-1, state_dim, state_dim]

            A_s = -self._kernel.state_transitions(
                self._action_times_without_initial[..., :-1],
                to_delta_time(self._action_times_without_initial),
            )

        # see the class docstring [... num_actions + 1, state_dim]
        # concat with zeros to generate the initial zero function
        return tf.concat(
            [
                tf.zeros_like(self._impulses[..., :1, :]),
                LowerTriangularBlockTriDiagonal(identities, A_s).solve(self._impulses),
            ],
            axis=-2,
        )

    def __call__(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Return the mean function evaluated at the given time points.

        For each time point, we find the index of the function associated with it; that is,
        the closest previous impulse.

        This index is then used to find the parameters of the function:

        .. math:: μₖ(t) = exp(F (t - tₖ))aₖ

        ...where :math:`tₖ < t ≤ tₖ₊₁`.

        :param time_points: A tensor with shape ``[... num_data]``.
        :return: The mean function evaluated at the time points, with
            shape ``[... num_data, state_dim]``.
        """
        num_batch = len(time_points.shape) - 1
        # find the index that each of the time points corresponds to [... num_data]
        # need to slice data because of the way searchsorted returns indices
        indices = tf.searchsorted(self._action_times[..., 1:], time_points)
        # (t - tₖ) get time from the start of the corresponding interval [... num_data, state_dim]
        delta_times = time_points - tf.gather(
            self._action_times, indices, axis=-1, batch_dims=num_batch
        )

        # aₖ get the coefficient for this time point [..., num_data, state_dim, 1]
        a_k = tf.gather(self._a_k, indices, axis=-2, batch_dims=num_batch)[..., None]

        # μₖ(t) = exp(F (t - tₖ))aₖ  [... num_data, state_dim]  tₖ < t ≤ tₖ₊₁
        state_mean = tf.matmul(
            self._kernel.state_transitions(time_points[..., :-1], delta_times), a_k
        )[..., 0]
        return self._kernel.generate_emission_model(time_points).project_state_to_f(state_mean)


@tf_scope_class_decorator
class StepMeanFunction(MeanFunction):
    r"""
    Represents a mean function that is a step action. This is given by:

    .. math::
        &u(t) = 0       &t ≤ t₀\\
        &u(t) = uₖ      &tₖ < t ≤ tₖ₊₁

    ...in:

    .. math:: dx(t)/dt = F x(t) + u(t) + L w(t)

    Then:

    .. math::
        &μ_(t) = 0                                              &   t ≤ t₀\\
        &μ₀(t) = -F⁻¹u₀ + exp(F (t - t₀))F⁻¹u₀                  &t₀ < t ≤ t₁\\
        &μₖ(t) = -F⁻¹uₖ + exp(F (t - tₖ))(F⁻¹uₖ + μₖ₋₁(tₖ))     &tₖ < t ≤ tₖ₊₁

    Or:

    .. math::
        &μ₋₁(t) = 0                           &     t ≤ t₀\\
        &μₖ(t) = aₖ + exp(F (t - tₖ))bₖ       &tₖ < t ≤ tₖ₊₁

    If we let:

    .. math::
        &a₋₁ =  b₋₁ = 0\\
        &aₖ =  -F⁻¹uₖ\\
        &bₖ = -aₖ + μₖ₋₁(tₖ) = Aₖbₖ₋₁ + aₖ₋₁ - aₖ

    ...where:

    .. math:: Aₖ = exp(F (tₖ - tₖ₋₁))

    ...we can write this as a :class:`LowerTriangularBlockTriDiagonal` equation::

        [ I             ] b₀   [a₋₁ - a₀]
        [-A₁, I         ] b₁   [a₀ - a₁]
        [    -A₂, I     ] b₂ = [a₁ - a₂]
        [         ᨞  ᨞  ] ⋮         ⋮
        [         -Aₙ, I] bₙ   [aₙ₋₁ - aₙ]

    We can then determine the :math:`bₖ` using a matrix solve.
    """

    def __init__(self, action_times: tf.Tensor, state_perturbations: tf.Tensor, kernel: SDEKernel):
        """
        :param action_times: The times at which actions occur, with shape ``[... num_actions]``.
        :param state_perturbations: The magnitude of the impulse, with shape
            ``[... num_actions, obs_dim]``.
        :param kernel: The kernel that is used to generate this mean function.
        """
        super().__init__(self.__class__.__name__)
        self._batch_shape = state_perturbations.shape[:-2]
        # TODO: use tf.shape
        self._num_actions = state_perturbations.shape[-2]
        self._state_dim = state_perturbations.shape[-1]
        tf.debugging.assert_equal(tf.shape(action_times), tf.shape(state_perturbations)[:-1])

        # we don't handle placeholders
        assert self._num_actions is not None

        self._action_times_without_initial = action_times

        self._kernel = kernel
        self._state_perturbations = state_perturbations

        # add dummy time point before the first for the initial zero function [... num_actions + 1]
        self._action_times = tf.concat([action_times[..., :1], action_times], axis=-1)

    @property
    def _a_k(self):
        shape = tf.concat(
            [self._batch_shape, [self._num_actions, self._state_dim, self._state_dim]], axis=0
        )
        # [... num_actions, state_dim]
        F_broadcast = tf.broadcast_to(self._kernel.feedback_matrix, shape)
        # [F⁻¹u₀, F⁻¹u₁, F⁻¹u₂ ... ] [... num_actions, state_dim]
        F_inv_perturb = tf.linalg.solve(F_broadcast, self._state_perturbations[..., None])[..., 0]

        # [0, -F⁻¹u₀, -F⁻¹u₁, -F⁻¹u₂ ...] [... num_actions + 1, state_dim]
        return tf.concat([tf.zeros_like(F_inv_perturb[..., :1, :]), -F_inv_perturb], axis=-2)

    @property
    def _b_k(self):
        # [-a₀, a₀ - a₁, a₁ - a₂, ... ] [... num_actions, state_dim]
        a_diff = self._a_k[..., :-1, :] - self._a_k[..., 1:, :]

        shape = tf.concat([self._batch_shape, [self._num_actions]], axis=0)
        # [... num_actions, state_dim, state_dim]
        identities = tf.eye(
            self._state_dim, dtype=self._state_perturbations.dtype, batch_shape=shape
        )
        if self._num_actions == 1:
            # No A_s if there is only one action
            A_s = None
        else:
            # [-exp(F (t₁ - t₀)), -exp(F (t₂ - t₁)) ... ] [... num_actions-1, state_dim, state_dim]
            A_s = -self._kernel.state_transitions(
                self._action_times_without_initial[..., :-1],
                to_delta_time(self._action_times_without_initial),
            )

        # see the class docstring [... num_actions + 1, state_dim]
        # concat with zeros to generate the initial zero function
        return tf.concat(
            [
                tf.zeros_like(a_diff[..., :1, :]),
                LowerTriangularBlockTriDiagonal(identities, A_s).solve(a_diff),
            ],
            axis=-2,
        )

    def __call__(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Return the mean function evaluated at the given time points.

        For each time point, we find the index of the function associated with it; that is,
        the closest previous step (call it :math:`k`).

        This index is then used to find the parameters of the function:

        .. math:: μₖ(t) = aₖ + exp(F (t - tₖ))bₖ

        ...where :math:`tₖ < t ≤ tₖ₊₁`.

        :param time_points: A tensor with shape ``[... num_data]``.
        :return: The mean function evaluated at the time points, with
            shape ``[... num_data, obs_dim]``.
        """
        num_batch = len(time_points.shape) - 1
        # find the index that each of the time points corresponds to [... num_data]
        # need to slice data because of the way searchsorted returns indices
        indices = tf.searchsorted(self._action_times[..., 1:], time_points)
        # (t - tₖ) the time from the start of the corresponding interval [... num_data, state_dim]
        delta_times = time_points - tf.gather(
            self._action_times, indices, axis=-1, batch_dims=num_batch
        )

        # get the constant part for this time point [..., num_data, state_dim]
        a_k = tf.gather(self._a_k, indices, axis=-2, batch_dims=num_batch)
        # get the coefficient of the exponential term [..., num_data, state_dim, 1]
        b_k = tf.gather(self._b_k, indices, axis=-2, batch_dims=num_batch)[..., None]

        # μₖ(t) = aₖ + exp(F (t - tₖ))bₖ     tₖ < t ≤ tₖ₊₁
        state_mean = (
            a_k + tf.matmul(self._kernel.state_transitions(time_points, delta_times), b_k)[..., 0]
        )
        return self._kernel.generate_emission_model(time_points).project_state_to_f(state_mean)
