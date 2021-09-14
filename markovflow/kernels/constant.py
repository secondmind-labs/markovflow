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
"""Module containing a kernel with a constant variance."""
from typing import Tuple

import tensorflow as tf
from gpflow import Parameter, default_float
from gpflow.utilities import positive

from markovflow.kernels.sde_kernel import StationaryKernel
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class Constant(StationaryKernel):
    """
    Introduces a constant variance. This kernel has the formula:

    .. math:: C(x, x') = σ²

    ...where :math:`σ²` is a kernel parameter representing the constant variance, which is
    supplied as a parameter to the constructor.

    The transition matrix :math:`F` in the SDE form for this kernel is :math:`F = [[1]]`.

    Covariance for the steady state is :math:`P∞ = [[σ²]]`.

    The state transition matrix is :math:`Aₖ = [[1]]`.

    The process covariance is :math:`Qₖ = [[0]]`.
    """

    def __init__(self, variance: float, output_dim: int = 1, jitter: float = 0.0) -> None:
        """
        :param variance: Initial variance for the kernel. Must be a positive float.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number used to make sure that
            matrix inversion is numerically stable.
        """
        super().__init__(output_dim, jitter)
        if variance <= 0:
            raise ValueError("variance must be positive.")

        self._variance = Parameter(variance, transform=positive(), name="variance")

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return 1

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        The state transition matrix at time step :math:`k` is :math:`Aₖ = [[1]]`.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Note this is ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        return tf.ones(shape, dtype=default_float())

    def process_covariances(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the process covariance matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        The process covariance for time step k is :math:`Qₖ = [[0]]`.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
           `` batch_shape + [num_transitions]``. Note this is ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        process_covariances = tf.zeros_like(time_deltas[..., None, None], dtype=default_float())
        return process_covariances + self.jitter_matrix

    def transition_statistics(
        self, transition_times: tf.Tensor, time_deltas: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the `state_transitions` and `process_covariances`.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Note this is ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tuple of two tensors with respective shapes
            ``batch_shape + [num_transitions, state_dim, state_dim]``
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        return (
            self.state_transitions(transition_times, time_deltas),
            self.process_covariances(transition_times, time_deltas),
        )

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.zeros((self.state_dim, self.state_dim), dtype=default_float())

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance :math:`P∞` of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        This is given by :math:`P∞ = [[σ²]]`.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        # Need the 1. multiplier because self._variance cannot be used directly as a tf.Tensor.
        return tf.identity([[1.0 * self._variance]])

    @property
    def variance(self) -> Parameter:
        """
        Return the variance parameter.
        """
        return self._variance
