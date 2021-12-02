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
"""Module containing the LEG-GPs family of kernels."""


import tensorflow as tf
from gpflow import Parameter, default_float

from markovflow.kernels.sde_kernel import StationaryKernel
from markovflow.utils import tf_scope_class_decorator

expm = tf.linalg.expm


@tf_scope_class_decorator
class LatentExponentiallyGenerated(StationaryKernel):
    """
    Represents the LEG-GPs kernel.

    This kernel defines an SDE with state dimension :math:`d`, whose dynamics are governed by:

    .. math:: dx = -½ G x dt + N dw (w Brownian motion)

    ...with :math:`G = N Nᵀ + R - Rᵀ`, and :math:`N, R` both arbitrary square matrices of
    size :math:`d × d`.

    Note that:

        * :math:`C = R - Rᵀ` is skew symmetric :math:`(Cᵀ = -C)`
        * If :math:`d` is even, :math:`C` has imaginary conjugate eigenvalue pairs
          :math:`(iλ₁ ,-iλ₁, ...)`
        * :math:`expm(C)` is an orthogonal matrix (specifying an isometry)

    The key reference is::

      @article{loper2020general,
          title={General linear-time inference for Gaussian Processes on one dimension},
          author={Loper, Jackson and Blei, David and Cunningham, John P and Paninski, Liam},
          journal={arXiv preprint arXiv:2003.05554},
          year={2020}
    }
    """

    def __init__(self, N: tf.Tensor, R: tf.Tensor, jitter: float = 0.0) -> None:
        """
        :param N: The Noise mixing matrix.
        :param R: The Rotation inducing matrix.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        tf.debugging.assert_shapes(
            [(N, ["state_dim", "state_dim"]), (R, ["state_dim", "state_dim"])]
        )
        self._state_dim = N.shape[-1]
        super().__init__(output_dim=self._state_dim, jitter=jitter)

        with self.name_scope:
            self.R = Parameter(R, name="R")
            self.N = Parameter(N, name="N")

    @property
    def state_dim(self) -> int:
        """Return the state dimension."""
        return self._state_dim

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Obtain the state transition matrices. That is:

        .. math:: Aₖ = expm[-½G Δtₖ]

        :param transition_times: Time points at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: Time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor of shape batch_shape + [num_transitions, state_dim, state_dim]
        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")

        # [..., num_transitions, 1, 1]
        extended_time_deltas = time_deltas[..., None, None]
        state_transitions = expm(extended_time_deltas * self.feedback_matrix)

        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(state_transitions), shape)
        return state_transitions

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix.

        Here, this is :math:`F (=-G/2)` with shape :math:`d × d`.
        """
        return (
            -(
                tf.matmul(self.N, self.N, transpose_b=True)
                + self.R
                - tf.linalg.matrix_transpose(self.R)
            )
            / 2.0
        )

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Obtain the steady state covariance :math:`P∞ = I`.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """

        return tf.eye(self.state_dim, dtype=default_float())

    def process_covariances(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Obtain the process covariance at time :math:`k`. This is calculated as:

        .. math:: Qₖ = P∞ - Aₖ P∞ Aₖᵀ = I - Aₖ Aₖᵀ

        :param transition_times: Time points at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: Time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        # [state_dim, state_dim]
        state_transitions = self.state_transitions(transition_times, time_deltas)
        I = tf.eye(self.state_dim, dtype=default_float())
        return I - tf.matmul(state_transitions, state_transitions, transpose_b=True)
