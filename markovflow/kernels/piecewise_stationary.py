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
"""Module containing a piecewise stationary kernel."""
from typing import List, Tuple

import tensorflow as tf
from gpflow.base import TensorType

from markovflow.base import APPROX_INF
from markovflow.emission_model import EmissionModel
from markovflow.kernels.sde_kernel import NonStationaryKernel, StationaryKernel
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class PiecewiseKernel(NonStationaryKernel):
    r"""
    Construct an SDE kernel whose state dynamic is governed by different SDEs. These are on the
    :math:`K+1` intervals specified by the :math:`K` change points :math:`cₖ`.

    On interval :math:`[cₖ, cₖ₊₁]`, the dynamics are governed by a SDE kernel :math:`kₖ`
    where :math:`c₀ = -∞`:

    .. math::
        &dx(t)/dt = Fₖ x(t) + Lₖ w(t),\\
        &f(t) = Hₖ x(t)

    Note the following:

        * This is currently restricted to cases where the kernels are the same.
        * State space models constructed by marginalizing out the process to
          time points :math:`t` are only valid if no transitions cross a change point.

    """

    def __init__(
        self,
        kernels: List[StationaryKernel],
        change_points: TensorType,
        output_dim: int = 1,
        jitter: float = 0.0,
    ):
        """
        :param kernels: An iterable over the kernels forming this kernel.
        :param change_points: Sorted change points.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        self.kernels = kernels
        assert self.kernels, "There must be at least one child kernel."
        kernels_output_dims = set(k.output_dim for k in kernels)
        assert len(kernels_output_dims) == 1, "All kernels must have the same output dimension"
        kernels_state_dims = set(k.state_dim for k in kernels)
        assert len(kernels_state_dims) == 1, "All kernels must have the same state dimension"
        if not all(isinstance(k, StationaryKernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")
        if not all(isinstance(k, kernels[0].__class__) for k in kernels):
            raise TypeError("can only combine kernels from the same class")

        self.change_points = change_points
        self.num_change_points = change_points.shape[0]
        self._output_dim = kernels[0].output_dim
        self._state_dim = kernels[0].state_dim
        self.jitter = jitter
        super().__init__(output_dim=output_dim, jitter=jitter)

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return self._state_dim

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Generate the :class:`~markovflow.emission_model.EmissionModel` associated with this kernel
        that maps from the latent :class:`~markovflow.state_space_model.StateSpaceModel`
        to the observations.

        The emission matrix is the Kronecker product of all the children emission matrices.

        :param time_points: The time points over which the emission model is defined, with shape
            ``batch_shape + [num_data]``.
        """
        # hack assuming for now a shared emission model
        indicies = self.split_time_indices(time_points)
        split_time_points = self.split_input(time_points, indicies)
        # apply different kernel method to each segments
        split_emissions = []
        for i_k, time_points_k in enumerate(split_time_points):
            split_emissions.append(
                self.kernels[i_k].generate_emission_model(time_points_k).emission_matrix
            )
        return EmissionModel(tf.concat(split_emissions, axis=-3))

    def initial_covariance(self, initial_time_point: tf.Tensor) -> tf.Tensor:
        """
        Return the initial covariance of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This is the covariance of the stationary distribution :math:`P∞` for the kernel active at
        the time passed in.

        :param initial_time_point: The time point associated with the first state, with shape
            ``batch_shape + [1,]``.
        :return: A tensor with shape ``batch_shape + [state_dim, state_dim]``.
        """
        steady_state_covariance = self.steady_state_covariances(initial_time_point)
        return steady_state_covariance[0]

    def split_time_indices(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        Gives each time point an index that refers to which interval it resides in.

        The sub-kernel that governs the SDE is different on different sub intervals,
        as specified by the change points.

        If there are :math:`K` change points, then :math:`0` is the index before the first
        change point and :math:`K + 1` is the index after the last change point.

        :param time_points: A tensor with shape ``batch_shape + [num_time_points]``.
        :return: A tensor of indices in range 0 - `num_change_points`, with shape
            ``batch_shape + [num_time_points]``.
        """
        # which change point is closest to the time points
        inf = APPROX_INF * tf.ones_like(self.change_points[..., -1:])
        change_points_augmented = tf.concat([-inf, self.change_points, inf], axis=-1)
        return tf.searchsorted(change_points_augmented, time_points, "right") - 1

    def split_input(self, input_tensor: tf.Tensor, indices: tf.Tensor) -> List[tf.Tensor]:
        """
        Partitions `input_tensor` into regions determined by the change points.

        If there are :math:`K` change points, then :math:`0` is the index before the first
        change point and :math:`K + 1` is the index after the last change point.

        :param input_tensor: An arbitrary input tensor, with shape ``batch_shape + [N]``.
        :param indices: The index for each input of the input tensor,
            with shape ``batch_shape + [N]``.
        :return: A list of tensors each with shape ``batch_shape + [Nₖ]``,
            where :math:`(Σₖ Nₖ = N)`.
        """
        return tf.dynamic_partition(
            input_tensor, indices, num_partitions=self.num_change_points + 1
        )

    def steady_state_covariances(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        For each time point, return the steady state covariance of the kernel active for that
        time point.

        :param time_points: A tensor with shape ``batch_shape + [num_time_points]``.
        :return: The steady state covariance at each time point, with shape
            ``batch_shape + [num_time_points, state_dim, state_dim]``.
        """
        # state covariances for each time interval
        steady_state_covariances = [k.steady_state_covariance for k in self.kernels]
        # counting time points falling in each interval
        indices = self.split_time_indices(time_points)
        y, idx, _ = tf.unique_with_counts(indices)
        # tiling the steady state covariance accordingly
        selected_steady_state_covariances = tf.gather(steady_state_covariances, y)
        return tf.gather(selected_steady_state_covariances, idx)

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel` :math:`Aₖ = exp(FΔtₖ)`.

        .. note:: Transitions are only valid if they do not cross a change point.

        :param transition_times: Time points at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: Time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        indices = self.split_time_indices(transition_times)
        split_transition_times = self.split_input(transition_times, indices)
        split_time_deltas = self.split_input(time_deltas, indices)
        split_state_transitions_args = zip(split_transition_times, split_time_deltas)
        # apply different kernel method to each segment
        return tf.concat(
            [
                self.kernels[i].state_transitions(*state_transitions_args)
                for i, state_transitions_args in enumerate(split_state_transitions_args)
            ],
            axis=-3,
        )

    def transition_statistics(
        self, transition_times: tf.Tensor, time_deltas: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return :meth:`state_transitions` and :meth:`process_covariances` together to
        save having to compute them twice.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tuple of two tensors, with respective shapes
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        state_transitions = self.state_transitions(transition_times, time_deltas)
        # steady state covariances for all transitions
        steady_state_covariances = self.steady_state_covariances(transition_times)
        # process noise covariance for all transitions
        A_Pinf = tf.matmul(state_transitions, steady_state_covariances)
        A_Pinf_A_T = tf.matmul(A_Pinf, state_transitions, transpose_b=True)
        process_covariances = steady_state_covariances - A_Pinf_A_T + self.jitter_matrix
        return state_transitions, process_covariances

    def feedback_matrices(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        For each time point, return the non-stationary feedback matrix :math:`F(t)`
        of the kernel active for that time point.

        :param time_points: A tensor with shape ``batch_shape + [num_time_points]``.
        :return: The feedback matrix at each time point, with shape
            ``batch_shape + [num_time_points, state_dim, state_dim]``.
        """
        # feedback matrices for each time interval
        feedback_matrices = [k.feedback_matrix for k in self.kernels]
        # counting time points falling in each interval
        indices = self.split_time_indices(time_points)
        y, idx, _ = tf.unique_with_counts(indices)
        # tiling the steady state covariance accordingly
        selected_feedback_matrices = tf.gather(feedback_matrices, y)
        return tf.gather(selected_feedback_matrices, idx)

    def state_offsets(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state offsets :math:`bₖ` of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This will usually be zero, but can be overridden if necessary.
        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim]``
        """
        indices = self.split_time_indices(transition_times)
        split_transition_times = self.split_input(transition_times, indices)
        split_time_deltas = self.split_input(time_deltas, indices)
        split_state_transitions_args = zip(split_transition_times, split_time_deltas)
        # apply different kernel method to each segment
        return tf.concat(
            [
                self.kernels[i].state_offsets(*state_transitions_args)
                for i, state_transitions_args in enumerate(split_state_transitions_args)
            ],
            axis=-2,
        )

    def state_means(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        For each time point, return the state mean of the kernel active for that
        time point.

        :param time_points: A tensor with shape ``batch_shape + [num_time_points]``.
        :return: The state mean at each time point ``batch_shape + [num_time_points, state_dim]``.
        """
        # state means for each time interval
        state_means = [k.state_mean for k in self.kernels]
        # counting time points falling in each interval
        indices = self.split_time_indices(time_points)
        y, idx, _ = tf.unique_with_counts(indices)
        # tiling the state means accordingly
        selected_state_means = tf.gather(state_means, y)
        return tf.gather(selected_state_means, idx)
