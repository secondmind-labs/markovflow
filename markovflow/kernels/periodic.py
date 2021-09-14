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
"""Module containing a periodic kernel."""
import numpy as np
import tensorflow as tf
from gpflow import Parameter, default_float
from gpflow.utilities import positive

from markovflow.kernels.sde_kernel import StationaryKernel
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class HarmonicOscillator(StationaryKernel):
    r"""
    Represents a periodic kernel. The definition is in the paper `"Explicit Link Between Periodic
    Covariance Functions and State Space Models" <http://proceedings.mlr.press/v33/solin14.pdf>`_.

    This kernel has the formula:

    .. math:: C(x, x') = σ² cos(2π/p * (x-x'))

    ...where:

        * :math:`σ²` is a kernel parameter, representing the constant variance
          this kernel introduces
        * :math:`p` is the period of the oscillator in radius

    The transition matrix :math:`F` in the SDE form for this kernel is:

    .. math::
        F = [&[0,  -λ],\\
             &[λ,  0]].

    ...where :math:`λ = 2π / period`.

    Covariance for the steady state is:

    .. math::
        P∞ = [&[σ², 0],\\
              &[0,  σ²]].

    The state transition matrix is:

    .. math::
        Aₖ = [&[cos(Δtₖλ),  -sin(Δtₖλ)],\\
              &[sin(Δtₖλ),  cos(Δtₖλ)]]

    The process covariance is:

    .. math::
        Qₖ = [&[0, 0],\\
              &[0, 0]].
    """

    def __init__(
        self, variance: float, period: float, output_dim: int = 1, jitter: float = 0.0
    ) -> None:
        """
        :param variance: Initial variance for the kernel. Must be a positive float.
        :param period: The period of the Harmonic oscillator, in radius. Must be a positive float.
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number used to make sure that
            matrix inversion is numerically stable.
        """
        super().__init__(output_dim, jitter=jitter)

        if variance <= 0.0:
            raise ValueError("variance must be positive.")

        if period <= 0.0:
            raise ValueError("period must be positive.")

        self._variance = Parameter(variance, transform=positive(), name="variance")
        self._period = Parameter(period, transform=positive(), name="period")

    @property
    def _lambda(self) -> tf.Tensor:
        """ λ the scalar used elsewhere in the docstrings """
        return 2.0 * np.pi / self._period

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return 2

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        r"""
        Return the state transition matrices of the kernel.

        The state transition matrix at time step :math:`k` is:

        .. math::
            Aₖ = [&[cos(Δtₖλ),  -sin(Δtₖλ)],\\
                  &[sin(Δtₖλ),  cos(Δtₖλ)]].

        ...where :math:`λ = 2π / period`.

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.

        """
        tf.debugging.assert_rank_at_least(time_deltas, 1, message="time_deltas cannot be a scalar.")
        deltas = time_deltas[..., None, None] * self._lambda
        cos = tf.cos(deltas)
        sin = tf.sin(deltas)
        result = tf.concat(
            [tf.concat([cos, -sin], axis=-1), tf.concat([sin, cos], axis=-1)], axis=-2
        )
        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    def process_covariances(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        r"""
        Return the state transition matrices of the kernel.

        The process covariance for time step k is:

        .. math::
            Qₖ = [&[0, 0],\\
                  &[0, 0]].

        Because this is a stationary kernel, `transition_times` is ignored.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``. Ignored.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        shape = tf.concat([time_deltas.shape, [self.state_dim, self.state_dim]], axis=0)
        return tf.zeros(shape, dtype=default_float()) + self.jitter_matrix

    @property
    def feedback_matrix(self) -> tf.Tensor:
        r"""
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        For this kernel, note that:

        .. math::
            F = [&[0,  -λ],\\
                 &[λ,  0]].

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.convert_to_tensor(value=[[0, -self._lambda], [self._lambda, 0]])

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        r"""
        Return the initial covariance of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        The steady state covariance :math:`P∞` is given by:

        .. math::
            P∞ = [&[σ², 0],\\
                  &[0,  σ²]].

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return self._variance * tf.eye(self.state_dim, dtype=default_float())

    @property
    def variance(self) -> Parameter:
        """
        Return the variance parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._variance

    @property
    def period(self) -> Parameter:
        """
        Return the period parameter. This is a GPflow
        `Parameter <https://gpflow.readthedocs.io/en/master/gpflow/index.html#gpflow-parameter>`_.
        """
        return self._period
