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
"""Module containing Stochastic Differential Equation (SDE) kernels."""
from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.base import Parameter, TensorType

from markovflow.emission_model import ComposedPairEmissionModel, EmissionModel, StackEmissionModel
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels.kernel import Kernel
from markovflow.state_space_model import StateSpaceModel, state_space_model_from_covariances
from markovflow.utils import (
    augment_matrix,
    augment_square_matrix,
    block_diag,
    kronecker_product,
    tf_scope_class_decorator,
    to_delta_time,
)


@tf_scope_class_decorator
class SDEKernel(Kernel, abc.ABC):
    r"""
    Abstract class representing kernels defined by the Stochastic Differential Equation:

    .. math::
        &dx(t)/dt = F(t) x(t) + L(t) w(t),\\
        &f(t) = H(t) x(t)

    For most kernels :math:`F, L, H` are not time varying; these have the more restricted form:

    .. math::
        &dx(t)/dt = F x(t) + L w(t),\\
        &f(t) = H x(t)

    ...with :math:`w(t)` white noise process with spectral density :math:`Q_c`, where:

    .. math::
        &x ∈ ℝ^d\\
        &F, L ∈ ℝ^{d × d}\\
        &H ∈ ℝ^{d × o}\\
        &Q_c ∈ ℝ^d\\
        &d \verb|is the state dimension|\\
        &o \verb|is the observation dimension|

    See the documentation for the :class:`StationaryKernel` class.

    Usually:

    .. math:: x(t) = [a(t), da(t)/dt, d²a(t)/dt ...]

    ...for some :math:`a(t)`, so the state dimension represents the degree of the stochastic
    differential equation in terms of :math:`a(t)`. Writing it in the above form is a standard
    trick for converting a higher order linear differential equation into a first order linear one.

    Since :math:`F, L, H` are constant matrices, the solution can be written analytically.
    For a given set of time points :math:`tₖ`, we can solve this SDE and define a state
    space model of the form:

    .. math:: xₖ₊₁ = Aₖ xₖ + bₖ + qₖ

    ...where:

    .. math::
        &qₖ \sim 𝓝(0, Qₖ)\\
        &x₀ \sim 𝓝(μ₀, P₀)\\
        &xₖ ∈ ℝ^d\\
        &Aₖ ∈ ℝ^{d × d}\\
        &bₖ ∈ ℝ^d\\
        &Qₖ ∈ ℝ^{d × d}\\
        &μ₀ ∈ ℝ^{d × 1}\\
        &P₀ ∈ ℝ^{d × d}

    If :math:`Δtₖ = tₖ₊₁ - tₖ`, then the transition matrix :math:`Aₜ` between states
    :math:`x(tₖ)` and :math:`x(tₖ₊₁)` is given by:

    .. math:: Aₖ = exp(FΔtₖ)

    The process noise covariance matrix :math:`Qₖ` between states :math:`x(tₖ)` and
    :math:`x(tₖ₊₁)` is given by:

    .. math:: Qₖ = ∫ exp(F (Δtₖ - τ)) L Q_c Lᵀ exp(F (Δtₖ - τ))ᵀ dτ

    We can write this in terms of the steady state covariance :math:`P∞` as:

    .. math:: Qₖ = P∞ - Aₖ P∞ Aₖᵀ

    We also define an emission model for a given output dimension:

    .. math:: fₖ = H xₖ

    ...where:

    .. math::
        &x ∈ ℝ^d\\
        &f ∈ ℝ^m\\
        &H ∈ ℝ^{m × d}\\
        &m \verb| is the output_dim|
    """

    def __init__(self, output_dim: int = 1, jitter: float = 0) -> None:
        """
        :param output_dim: The output dimension of the kernel.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(self.__class__.__name__)
        assert jitter >= 0.0, "jitter must be a non-negative float number."
        self._jitter = jitter
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """
        Return the output dimension of the kernel.
        """
        return self._output_dim

    def build_finite_distribution(self, time_points: tf.Tensor) -> GaussMarkovDistribution:
        """
        Return the :class:`~markovflow.gauss_markov.GaussMarkovDistribution` that this kernel
        represents on the provided time points.

        .. note:: Currently the only representation we can use is
            :class:`~markovflow.state_space_model.StateSpaceModel`.

        :param time_points: The times between which to define the distribution, with
            shape ``batch_shape + [num_data]``.
        """
        return self.state_space_model(time_points)

    def state_space_model(self, time_points: tf.Tensor) -> StateSpaceModel:
        """
        Return the :class:`~markovflow.state_space_model.StateSpaceModel` that this
        kernel represents on the provided time points.

        :param time_points: The times between which to define the state space model, with shape
            ``batch_shape + [num_data]``. This must be strictly increasing.
        """
        batch_shape = time_points.shape[:-1]

        As, Qs = self.transition_statistics_from_time_points(time_points)

        return state_space_model_from_covariances(
            initial_mean=self.initial_mean(batch_shape),
            initial_covariance=self.initial_covariance(time_points[..., 0:1]),
            state_transitions=As,
            state_offsets=self.state_offsets(time_points[..., :-1], to_delta_time(time_points)),
            process_covariances=Qs,
        )

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        r"""
        Generate the :class:`~markovflow.emission_model.EmissionModel` associated with this kernel
        that maps from the latent :class:`~markovflow.state_space_model.StateSpaceModel`
        to the observations.

        For any :class:`SDEKernel`, the state representation is usually:

        .. math:: x(t) = [a(t), da(t)/dt, d²a(t)/dt ...] \verb| for some | a(t)

        In this case, we are interested only in the first element of :math:`x`. That is, the
        output :math:`f(t)` is given by :math:`f(t) = a(t)`, so :math:`H` is given by
        :math:`[1, 0, 0, ...]`.

        If different behaviour is required, this method should be overridden.

        :param time_points: The time points over which the emission model is defined, with shape
            ``batch_shape + [num_data]``.
        """
        # create 2D matrix
        emission_matrix = tf.concat(
            [
                tf.ones((self.output_dim, 1), dtype=default_float()),
                tf.zeros((self.output_dim, self.state_dim - 1), dtype=default_float()),
            ],
            axis=-1,
        )
        # tile for each time point
        # expand emission_matrix from [output_dim, state_dim], to [1, 1 ... output_dim, state_dim]
        # where there is a singleton dimension for the dimensions of time points
        batch_shape = time_points.shape[:-1]  # num_data may be undefined so skip last dim
        shape = tf.concat(
            [tf.ones(len(batch_shape) + 1, dtype=tf.int32), tf.shape(emission_matrix)], axis=0
        )
        emission_matrix = tf.reshape(emission_matrix, shape)

        # tile the emission matrix into shape batch_shape + [num_data, output_dim, state_dim]
        repetitions = tf.concat([tf.shape(time_points), [1, 1]], axis=0)
        return EmissionModel(tf.tile(emission_matrix, repetitions))

    def initial_mean(self, batch_shape: tf.TensorShape) -> tf.Tensor:
        """
        Return the initial mean of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This will usually be zero, but can be overridden if necessary.

        :param batch_shape: Leading dimensions for the initial mean.
        :return: A tensor of zeros with shape ``batch_shape + [state_dim]``.
        """
        shape = tf.concat([tf.TensorShape(batch_shape), [self.state_dim]], axis=0)
        return tf.zeros(shape, dtype=default_float())

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        raise NotImplementedError

    @abstractmethod
    def initial_covariance(self, initial_time_point: tf.Tensor) -> tf.Tensor:
        """
        Return the initial covariance of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        For stationary kernels this is typically the covariance of the stationary distribution for
        :math:`x, P∞`.

        In the general case the initial covariance depends on time, so we  need the
        `initial_time_point` to generate it.

        :param initial_time_point: The time_point associated with the first state, with shape
            ``batch_shape + [1,]``.
        :return: A tensor with shape ``batch_shape + [state_dim, state_dim]``.
        """
        raise NotImplementedError

    def transition_statistics_from_time_points(self, time_points: tf.Tensor):
        """
        Generate the transition matrices when the time deltas are
        between adjacent `time_points`.

        :param time_points: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions + 1]``.
        :return: A tuple of two tensors, with respective shapes
            ``batch_shape + [num_transitions, state_dim, state_dim]``
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        return self.transition_statistics(time_points[..., :-1], to_delta_time(time_points))

    @abstractmethod
    def transition_statistics(
        self, transition_times: tf.Tensor, time_deltas: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the :meth:`state_transitions` and :meth:`process_covariances` together to
        save having to compute them twice.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tuple of two tensors, with respective shapes
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        raise NotImplementedError

    @abstractmethod
    def state_offsets(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state offsets :math:`bₖ` of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim]``.
        """
        raise NotImplementedError

    @abstractmethod
    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel` :math:`Aₖ = exp(FΔtₖ)`.

        :param transition_times: Time points at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: Time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        raise NotImplementedError

    def process_covariances(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the process covariance matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        The process covariance at time :math:`k` is calculated as:

        .. math:: Qₖ = P∞ - Aₖ P∞ Aₖᵀ

        These transition matrices can be overridden for more specific use cases if necessary.

        :param transition_times: Time points at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: Time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        _, Qs = self.transition_statistics(transition_times, time_deltas)
        return Qs

    @property
    def jitter_matrix(self) -> tf.Tensor:
        """
        Jitter to add to the output of :meth:`process_covariances` and
        :meth:`initial_covariance` shape.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        return tf.eye(self.state_dim, dtype=default_float()) * self._jitter

    def __add__(self, other: "SDEKernel") -> "Sum":
        """ Operator for combining kernel objects by summing them. """
        assert self.output_dim == other.output_dim
        return Sum([self, other])

    def __mul__(self, other: "SDEKernel") -> "Product":
        """ Operator for combining kernel objects by multiplying them. """
        assert self.output_dim == other.output_dim
        return Product(kernels=[self, other])


class StationaryKernel(SDEKernel, abc.ABC):
    r"""
    Abstract class representing stationary kernels defined by the Stochastic Differential Equation:

    .. math::
        &dx(t)/dt = F x(t) + L w(t),\\
        &f(t) = H(t) x(t)

    For most kernels :math:`H` will not be time varying; that is, :math:`f(t) = H x(t)`.
    """

    def __init__(
        self,
        output_dim: int = 1,
        jitter: float = 0,
        state_mean: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        :param output_dim: The output dimension of the kernel.
        :param state_mean: A tensor with shape [state_dim,].
        """
        super().__init__(output_dim, jitter)
        if state_mean is None:
            state_mean = tf.zeros([self.state_dim], dtype=default_float())
        self._state_mean = Parameter(state_mean, trainable=False)

    def initial_mean(self, batch_shape: tf.TensorShape) -> tf.Tensor:
        """
        Return the initial mean of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This will usually be zero, but can be overridden if necessary.

        :param batch_shape: Leading dimensions for the initial mean.
        :return: A tensor of zeros with shape ``batch_shape + [state_dim]``.
        """
        shape = tf.concat([tf.TensorShape(batch_shape), self._state_mean.shape], axis=0)
        return tf.broadcast_to(self._state_mean, shape)

    def set_state_mean(self, state_mean: tf.Tensor, trainable: bool = False):
        """
        Sets the state mean for the kernel.

        :param state_mean: A tensor with shape [state_dim,].
        :param trainable: Boolean value to set the state mean trainable.
        """
        self._state_mean = Parameter(state_mean, trainable=trainable)

    def initial_covariance(self, initial_time_point: tf.Tensor) -> tf.Tensor:
        """
        Return the initial covariance of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        For stationary kernels this is the covariance of the stationary distribution for
        :math:`x,P∞` and is independent of the time passed in.

        :param initial_time_point: The time point associated with the first state, with shape
            ``batch_shape + [1,]``.
        :return: A tensor with shape ``batch_shape + [state_dim, state_dim]``.
        """
        tf.debugging.assert_equal(tf.shape(initial_time_point)[-1], 1)
        shape = tf.concat([tf.shape(initial_time_point), [self.state_dim, self.state_dim]], axis=0)
        initial_covariance = self.steady_state_covariance * tf.ones(shape, dtype=default_float())
        # remove the time dimension after multiplying with `steady_state_covariance`
        # this allows it to broadcast in the case of `StackKernel`
        return initial_covariance[..., 0, :, :] + self.jitter_matrix

    def transition_statistics(
        self, transition_times: tf.Tensor, time_deltas: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return :meth:`state_transitions` and :meth:`process_covariances` together to save
        having to compute them twice.

        By default this uses the state transitions to calculate the process covariance:

        .. math:: Qₖ = P∞ - Aₖ P∞ Aₖᵀ

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tuple of two tensors, with respective shapes
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
            ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        state_transitions = self.state_transitions(transition_times, time_deltas)
        # A P∞ (A)ᵀ [... num_transitions, state_dim, state_dim]
        # matmuls between [..., num_tranistions, state_dim, state_dim] and [state_dim, state_dim]
        A_Pinf = tf.matmul(state_transitions, self.steady_state_covariance)
        A_Pinf_A_T = tf.matmul(A_Pinf, state_transitions, transpose_b=True)
        process_covariances = self.steady_state_covariance - A_Pinf_A_T
        return state_transitions, process_covariances + self.jitter_matrix

    @property
    @abstractmethod
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix :math:`F`. This is where:

        .. math:: dx(t)/dt = F x(t) + L w(t)

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        raise NotImplementedError

    def state_offsets(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        :math:`dx = F (x - m)dt  \to  x(t) = A x(0) + (I-A)m`

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim]``
        """
        state_transitions = self.state_transitions(transition_times, time_deltas)
        return tf.einsum(
            "...ij,j->...i",
            -(state_transitions - tf.eye(self.state_dim, dtype=default_float())),
            self._state_mean,
        )

    @property
    @abstractmethod
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance :math:`P∞`, given implicitly by:

        .. math:: F P∞ + P∞ Fᵀ + LQ_cLᵀ = 0

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        raise NotImplementedError

    @property
    def state_mean(self) -> tf.Tensor:
        """
        Return the state mean.

        :return: A tensor with shape ``[state_dim,]``.
        """
        return tf.identity(self._state_mean)


class NonStationaryKernel(SDEKernel, abc.ABC):
    r"""
    Abstract class representing non-stationary kernels defined by the Stochastic Differential
    Equation:

    .. math::
        &dx(t)/dt = F(t) x(t) + L(t) w(t),\\
        &f(t) = H(t) x(t)

    For most kernels :math:`H` will not be time varying; that is, :math:`f(t) = H x(t)`.
    """

    @abstractmethod
    def feedback_matrices(self, time_points: tf.Tensor) -> tf.Tensor:
        """
        The non-stationary feedback matrix :math:`F(t)` at times :math:`t`, where:

        .. math:: dx(t)/dt = F(t) x(t) + L w(t)

        :param time_points: The times at which the feedback matrix is evaluated, with shape
            ``batch_shape + [num_time_points]``.
        :return: A tensor with shape ``batch_shape + [num_time_points, state_dim, state_dim]``.
        """
        raise NotImplementedError

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
        raise NotImplementedError


@tf_scope_class_decorator
class ConcatKernel(StationaryKernel, abc.ABC):
    r"""
    Abstract class implementing the state space model of multiple kernels that have been
    combined together. Combined with differing emission models this can give rise to the
    :class:`Sum` kernel or to a multi-output kernel.

    The state space of any :class:`ConcatKernel` consists of all the state spaces of
    child kernels concatenated (in the tensorflow.concat sense) together:

    .. math::
       [x¹(t),\\
       x²(t)]

    So the SDE of the kernel becomes:

    .. math::
        &dx(t)/dt = &[[F¹ 0],     &[x¹(t)    &[[L¹ 0],   &[w¹(t),\\
        &           &[0  F²]]     &x²(t)]  + &[0  L²]]   &w²(t)]\\
        &f(t) = [H¹ H²] x(t)
    """

    def __init__(self, kernels: List[SDEKernel], jitter: float = 0.0):
        """
        :param kernels: A list of child kernels that will have their state spaces
                        concatenated together.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
                       maintain numerical stability during inversion.
        """
        self._kernels = kernels
        assert self._kernels, "There must be at least one child kernel."
        kernels_output_dims = set(k.output_dim for k in kernels)
        assert len(kernels_output_dims) == 1, "All kernels must have the same output dimension"
        if not all(isinstance(k, SDEKernel) for k in kernels):
            raise TypeError("Can only combine SDEKernel instances.")
        self._state_dim = sum(k.state_dim for k in kernels)
        super().__init__(kernels_output_dims.pop(), jitter=jitter)

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return self._state_dim

    @property
    def kernels(self) -> List[SDEKernel]:
        """
        Return a list of child kernels.
        """
        return self._kernels

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel` :math:`Aₖ = exp(FΔtₖ)`.

        The state transition matrix is the block diagonal matrix of the child state
        transition matrices.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        result = block_diag(
            [k.state_transitions(transition_times, time_deltas) for k in self.kernels]
        )
        tf.debugging.assert_shapes([(result, [*time_deltas.shape, self.state_dim, self.state_dim])])
        return result

    def initial_mean(self, batch_shape: tf.TensorShape) -> tf.Tensor:
        """
        Return the initial mean of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        The combined mean is the child means concatenated together:

        .. math:: [μ1 μ2, ...]

        ...to form a longer mean vector.

        :param batch_shape: A tuple of leading dimensions for the initial mean.
        :return: A tensor of zeros with shape ``batch_shape + [state_dim]``.
        """
        result = tf.concat([k.initial_mean(batch_shape) for k in self.kernels], axis=-1)
        shape = tf.concat([tf.TensorShape(batch_shape), [self.state_dim]], axis=0)
        tf.debugging.assert_equal(result.shape, shape)
        return result

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix. This is the block diagonal matrix of
        child feedback matrices.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        result = block_diag([k.feedback_matrix for k in self.kernels])
        shape = tf.TensorShape([self.state_dim, self.state_dim])
        tf.debugging.assert_equal(result.shape, shape)
        return result

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance. This is the block diagonal matrix of
        child steady state covariance matrices.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        result = block_diag([k.steady_state_covariance for k in self.kernels])

        shape = tf.TensorShape([self.state_dim, self.state_dim])
        tf.debugging.assert_equal(result.shape, shape)
        return result


@tf_scope_class_decorator
class Sum(ConcatKernel):
    """
    Sums a list of child kernels.

    There are two ways to implement this kernel: Stacked and Concatenated.

    This class implements the Concatenated version, where the state space of the :class:`Sum`
    kernel includes covariance terms between the child kernels.
    """

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        r"""
        Generate the emission matrix :math:`H`. This is the concatenation:

        .. math:: H = [H₁, H₂, ..., Hₙ]

        ...where :math:`\{Hᵢ\}ₙ` are the emission matrices of the child kernels. Thus the state
        dimension for this kernel is the sum of the state dimension of the child kernels.

        :param time_points: The time points over which the emission model is defined, with shape
                        ``batch_shape + [num_data]``.
        :return: The emission model associated with this kernel, with emission matrix with shape
                        ``batch_shape + [num_data, output_dim, state_dim]``.
        """
        emission_matrix = tf.concat(
            [k.generate_emission_model(time_points).emission_matrix for k in self.kernels], axis=-1,
        )
        return EmissionModel(emission_matrix)


@tf_scope_class_decorator
class Product(StationaryKernel):
    r"""
    Multiplies a list of child kernels.

    The feedback matrix is the Kronecker product of the feedback matrices from the child kernels.
    We will use a product kernel with two child kernels as an example. Let :math:`A` and
    :math:`B` be the feedback matrix from these two child kernels. The feedback matrix :math:`F`
    of the product kernel is:

    .. math::
        &F &= &A ⊗ B\\
        &  &= &[[A₁₁ B, ..., A₁ₙ B],\\
        &  &  &...,\\
        &  &  &[Aₙ₁ B, ..., Aₙₙ B]]

    ...where :math:`⊗` is the Kronecker product operator.

    The state transition matrix is the Kronecker product of the state transition matrices from
    the child kernels. Let :math:`Aₖ` and :math:`Bₖ` be the state transition matrix from these two
    child kernels at time step :math:`k`. The state transition matrix
    :math:`Sₖ` of the product kernel is:

    .. math::
        &Sₖ &= &Aₖ ⊗ Bₖ\\
        &   &= &[[Aₖ₁₁ Bₖ, ..., Aₖ₁ₙ Bₖ],\\
        &   &  &...,\\
        &   &  &[Aₖₙ₁ Bₖ, ..., Aₖₙₙ Bₖ]]

    The steady state covariance matrix is the Kronecker product of the steady covariance matrix
    from the child kernels. Let :math:`A∞` and :math:`B∞` be the steady covariance matrix from
    these two child kernels. The state transition matrix :math:`P∞` of the product kernel is:

    .. math::
        &P∞ &= &A∞ ⊗ B∞\\
        &   &= &[[A∞₁₁ B∞, ..., A∞ B∞],\\
        &   &  &...,\\
        &   &  &[A∞ₙ₁ B∞, ..., A∞ₙₙ B∞]]

    The process covariance matrix :math:`Qₖ` at time step :math:`k` is calculated using the
    same formula as defined in the parent class SDEKernel:

    .. math:: Qₖ = P∞ - Sₖ P∞ Sₖᵀ

    ...where the steady state matrix :math:`P∞` and the state transition :math:`Sₖ`
    are defined above.
    """

    def __init__(self, kernels: List[SDEKernel], jitter: float = 0.0):
        """
        :param kernels: An iterable over the kernels to be multiplied together.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        self._kernels = kernels
        assert self._kernels, "There must be at least one child kernel."
        kernels_output_dims = set(k.output_dim for k in kernels)
        assert len(kernels_output_dims) == 1, "All kernels must have the same output dimension"
        if not all(isinstance(k, SDEKernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")
        self._state_dim = np.prod([k.state_dim for k in kernels])
        super().__init__(kernels_output_dims.pop(), jitter=jitter)

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return self._state_dim

    @property
    def kernels(self) -> List[SDEKernel]:
        """Return a list of child kernels."""
        return self._kernels

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition. This is the Kronecker product of the child state transitions.

        :param transition_times: A tensor of times at which to produce matrices, shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        result = kronecker_product(
            [k.state_transitions(transition_times, time_deltas) for k in self.kernels]
        )
        shape = tf.concat([tf.shape(time_deltas), [self.state_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix. This is the Kronecker product of the child feedback matrices.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        result = kronecker_product([k.feedback_matrix for k in self.kernels])
        shape = tf.TensorShape([self.state_dim, self.state_dim])
        tf.debugging.assert_equal(result.shape, shape)
        return result

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance. This is the Kronecker product
        of the child steady state covariances.

        :return: A tensor with shape ``[state_dim, state_dim]``.
        """
        result = kronecker_product([k.steady_state_covariance for k in self.kernels])
        shape = tf.TensorShape([self.state_dim, self.state_dim])
        tf.debugging.assert_equal(result.shape, shape)
        return result

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Generate the emission matrix. This is the
        Kronecker product of all the child emission matrices.

        :param time_points: The time points over which the emission model is defined, with shape
            ``batch_shape + [num_data]``.
        """
        result = kronecker_product(
            [k.generate_emission_model(time_points).emission_matrix for k in self.kernels]
        )

        shape = tf.concat([tf.shape(time_points), [self.output_dim, self.state_dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(result), shape)
        return EmissionModel(result)


@tf_scope_class_decorator
class IndependentMultiOutput(ConcatKernel):
    """
    Takes a concatenated state space model consisting of multiple child
    kernels and projects the state space associated with each kernel into a separate observation
    vector.

    The result is similar to training several kernels on the same data separately,
    except that because of the covariance terms in the state space there can be correlation
    between the separate observation vectors.
    """

    def __init__(self, kernels: List[SDEKernel], jitter: float = 0.0):
        """
        :param kernels: An iterable over child kernels which will have their state spaces
                        concatenated together.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
                       maintain numerical stability during inversion.
        """
        super().__init__(kernels, jitter)
        self._output_dim = self._output_dim * len(kernels)

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        r"""
        Generate the emission matrix :math:`H`. This is the direct sum of the child emission
        matrices, for example:

        .. math:: H = H₁ ⊕ H₂ ⊕ ... ⊕ Hₙ

        ...where :math:`\{Hᵢ\}ₙ` are the emission matrices of the child kernels.

        :param time_points: The time points over which the emission model is defined, with shape
                        ``batch_shape + [num_data]``.
        :return: The emission model associated with this kernel.
        """
        emission_matricies = (
            k.generate_emission_model(time_points).emission_matrix for k in self.kernels
        )
        # tf.linalg.LinearOperatorBlockDiag almost does this, but only works for square inputs
        pre_pad = 0
        post_pad = self.state_dim
        padded_emission_matricies = []
        for emission_matrix in emission_matricies:
            d = emission_matrix.shape[-1]
            post_pad -= d
            batch_shape = emission_matrix.shape[:-3]  # num_data may be unspecified
            shape = (len(batch_shape) + 2, 2)
            paddings = tf.concat([tf.zeros(shape, dtype=tf.int64), [[pre_pad, post_pad]]], axis=0)
            pre_pad += d
            padded_emission_matricies.append(tf.pad(emission_matrix, paddings, "CONSTANT"))

        emission_matrix = tf.concat(padded_emission_matricies, axis=-2)
        return EmissionModel(emission_matrix)


@tf_scope_class_decorator
class FactorAnalysisKernel(ConcatKernel):
    r"""
    Produces an emission model which performs a linear mixing of Gaussian
    processes according to a known time varying weight function and a learnable loading matrix:

    .. math:: fᵢ(t) = Σⱼₖ Aᵢⱼ(t)Bⱼₖgₖ(t)

    ...where:

        * :math:`\{fᵢ\}ₙ` are the observable processes
        * :math:`\{gₖ\}ₘ` are the latent GPs
        * :math:`A^{n × m}` is a known, possibly time dependant, weight matrix
        * :math:`B^{m × m}` is either the identity or a trainable loading matrix
    """

    def __init__(
        self,
        weight_function: Callable[[TensorType], TensorType],
        kernels: List[SDEKernel],
        output_dim: int,
        trainable: bool = True,
        jitter: float = 0.0,
    ):
        """
        :param weight_function: A function that, given :data:`~markovflow.base.TensorType`
            time points with shape ``batch_shape + [num_data, ]``, returns a weight matrix
            with the relative mixing of the tensors, with shape
            ``batch_shape + [num_data, output_dim, n_latents]``.
        :param kernels: An iterable over child kernels that will have their state spaces
                concatenated together, with shape ``[n_latents, ]``.
        :param output_dim: The output dimension of the kernel. This should have the same shape as
                the `output_dim` of the weight matrix returned by the weight function.
        :param trainable: Whether the loading matrix :math:`B` should be trainable.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
                       maintain numerical stability during inversion.
        """
        super().__init__(kernels, jitter)
        self._latent_components = IndependentMultiOutput(kernels, jitter)
        self._output_dim = output_dim
        self.latent_dim = self._latent_components.output_dim
        self._Afn = weight_function
        self._B = Parameter(
            np.identity(self.latent_dim), name="loading_matrix", trainable=trainable
        )

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Generate the emission matrix :math:`WH`. This is where:

        .. math:: H = H₁ ⊕ H₂ ⊕ ... ⊕ Hₙ

        ...as per the multi-output kernel, and :math:`W = AB`.

        :param time_points: The time points over which the emission model is defined, with shape
                        ``batch_shape + [num_data, ]``.
        :return: The emission model associated with this kernel.
        """
        W = self._Afn(time_points) @ self._B
        outer_emission_model = EmissionModel(W)
        latent_emission_model = self._latent_components.generate_emission_model(time_points)
        return ComposedPairEmissionModel(outer_emission_model, latent_emission_model)


@tf_scope_class_decorator
class StackKernel(StationaryKernel):
    r"""
    Implements the state space model of multiple kernels that have been combined together.
    Unlike a :class:`ConcatKernel`, it manages the multiple
    kernels by introducing a leading dimension (stacking), rather than forming a block
    diagonal form of each parameter explicitly.

    The prior of both a :class:`StackKernel` and a :class:`ConcatKernel` is the same (independent).
    However, posterior state space models built upon a :class:`StackKernel` will maintain this
    independency, in contrast to the posteriors building upon a :class:`ConcatKernel`,
    which model correlations between the processes.

    Combined with different emission models this can give rise to a multi-output stack
    kernel, and perhaps in the future an additive kernel.

    The state space of this kernel consists of all the state space of the child kernels
    stacked (in the tensorflow.stack sense) together, with padded zeros when the state space
    of one of the kernels is larger than any of the others::

        [ x₁⁽¹⁾(t) ] ᨞
        [   0   ]   [ x₁⁽ᵐ⁾(t) ]
                  ᨞ [ x₂⁽ᵐ⁾(t) ]

    ...where :math:`m` are the number of kernels / outputs.

    So the SDE of the kernel becomes::

        dx(t)/dt = [F⁽¹⁾] ᨞   [x⁽¹⁾(t)] ᨞    + [L⁽¹⁾] ᨞    [w⁽¹⁾(t)] ᨞
                      ᨞ [F⁽ᵐ⁾]   ᨞ [x⁽ᵐ⁾(t)]      ᨞ [L⁽ᵐ⁾]    ᨞ [w⁽ᵐ⁾(t)]

        f(t) = [H⁽¹⁾] ᨞   [x⁽¹⁾(t)] ᨞
                  ᨞ [H⁽ᵐ⁾]   ᨞ [x⁽ᵐ⁾(t)]
    """

    def __init__(self, kernels: List[SDEKernel], jitter: float = 0.0):
        """
        :param kernels: A list of child kernels that will have their state spaces
            concatenated together. Since we model each output independently, the length of the
            kernel list defines the number of the outputs. Note that each kernel should have
            individual `output_dim` 1.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
                       maintain numerical stability during inversion.
        """
        self._kernels = kernels
        assert self._kernels, "There must be at least one child kernel."
        kernels_output_dims = set(k.output_dim for k in kernels)
        assert len(kernels_output_dims) == 1, "All kernels must have the same output dimension"
        assert kernels[0].output_dim == 1, "All kernels must have individual output dimensions of 1"
        self._state_dim = max(k.state_dim for k in kernels)
        super().__init__(kernels_output_dims.pop(), jitter=jitter)
        if not all(isinstance(k, SDEKernel) for k in kernels):
            raise TypeError("Can only combine SDEKernel instances.")
        self.num_kernels = len(kernels)

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.
        """
        return self._state_dim

    @property
    def kernels(self) -> List[SDEKernel]:
        """
        Return a list of child kernels.
        """
        return self._kernels

    def _check_batch_shape_is_compatible(self, batch_shape: tf.TensorShape) -> None:
        """
        Helper method to check the compatibility of batch_shape.
        For the `StackKernel` the batch_shape must have the following shape:

                    (..., num_kernels)

        In any other case this method raises a tf.errors.InvalidArgumentError.

        :param batch_shape: a tuple with the shape to check
        """
        # raise if batch_shape is provided and batch_shape's last dim is not num_kernel
        shape_chk = tf.logical_and(
            tf.greater(batch_shape.rank, 0), batch_shape[-1:] == [self.num_kernels]
        )
        tf.debugging.assert_equal(
            shape_chk,
            True,
            message="""Batch shape's last dimension must be equal to the
                                             number of kernels""",
        )

    def state_transitions(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state transition matrices of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel` :math:`Aₖ = exp(FΔtₖ)`.

        The state transition matrix is the stacked matrix of the child state
        transition matrices, padded with zeros (if necessary) to match the largest state dim
        across kernels.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]`` where ``batch_shape = (..., num_kernels)``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim, state_dim]``
            where ``batch_shape = (..., num_kernels)``.
        """
        batch_shape = time_deltas.shape[:-1]
        num_transitions = tf.shape(time_deltas)[-1]

        # raise if batch_shape is provided and batch_shape's last dim is not num_kernel or 1
        self._check_batch_shape_is_compatible(batch_shape)

        result = tf.stack(
            [
                augment_square_matrix(
                    k.state_transitions(transition_times[..., i, :], time_deltas[..., i, :]),
                    self.state_dim - k.state_dim,
                    fill_zeros=True,
                )
                for i, k in enumerate(self.kernels)
            ],
            axis=-4,
        )

        shape = tf.concat(
            [
                batch_shape[:-1],
                [self.num_kernels, num_transitions, self.state_dim, self.state_dim],
            ],
            axis=0,
        )
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    def initial_mean(self, batch_shape: tf.TensorShape) -> tf.Tensor:
        """
        Return the initial mean of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This will usually be zero, but can be overridden if necessary.

        We override :meth:`SDEKernel.initial_mean` from the
        parent class to check there is a compatible `batch_shape`.

        :param batch_shape: A tuple of leading dimensions for the initial mean, where batch_shape
            can be ``(..., num_kernels)``.
        :return: A tensor of zeros with shape ``batch_shape + [state_dim]``, where
            ``batch_shape = (..., num_kernels)``.
        """
        # raise if batch_shape is provided and batch_shape's last dim is not num_kernel or 1
        self._check_batch_shape_is_compatible(batch_shape)

        return super().initial_mean(batch_shape)

    def state_offsets(self, transition_times: tf.Tensor, time_deltas: tf.Tensor) -> tf.Tensor:
        """
        Return the state offsets :math:`bₖ` of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This will usually be zero, but can be overridden if necessary.

        We override :meth:`SDEKernel.state_offsets` from the
        parent class to check there is a compatible `batch_shape`.

        :param transition_times: A tensor of times at which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :param time_deltas: A tensor of time gaps for which to produce matrices, with shape
            ``batch_shape + [num_transitions]``.
        :return: A tensor with shape ``batch_shape + [num_transitions, state_dim]``
        """
        batch_shape = time_deltas.shape[:-1]
        num_transitions = tf.shape(time_deltas)[-1]

        # raise if batch_shape is provided and batch_shape's last dim is not num_kernel or 1
        self._check_batch_shape_is_compatible(batch_shape)

        result = tf.stack(
            [
                augment_matrix(
                    k.state_offsets(transition_times[..., i, :], time_deltas[..., i, :]),
                    self.state_dim - k.state_dim,
                )
                for i, k in enumerate(self.kernels)
            ],
            axis=-3,
        )

        shape = tf.concat(
            [batch_shape[:-1], [self.num_kernels, num_transitions, self.state_dim],], axis=0,
        )
        tf.debugging.assert_equal(tf.shape(result), shape)
        return result

    @property
    def feedback_matrix(self) -> tf.Tensor:
        """
        Return the feedback matrix. This is the stacked matrix of child feedback matrices, padded
        with zeros to have matching state dims.

        :return: A tensor with shape ``[num_kernels, state_dim, state_dim]``.
        """
        result = tf.stack(
            [
                augment_square_matrix(k.feedback_matrix, self.state_dim - k.state_dim)
                for k in self.kernels
            ],
            axis=-3,
        )
        tf.debugging.assert_equal(
            tf.shape(result), [self.num_kernels, self.state_dim, self.state_dim]
        )
        return result

    @property
    def steady_state_covariance(self) -> tf.Tensor:
        """
        Return the steady state covariance. This is the stacked matrix of child steady state
        covariance matrices, padded with the identity (if necessary) to have matching state dims.

        Note that we further append a singleton dimensions after the `num_kernels` so it
        can broadcast across the number of data.

        :return: A tensor with shape ``[num_kernels, 1, state_dim, state_dim]``.
        """
        result = tf.stack(
            [
                augment_square_matrix(k.steady_state_covariance, self.state_dim - k.state_dim)
                for k in self.kernels
            ],
            axis=-3,
        )[..., None, :, :]

        # make sure the shape has a singleton dimensions to broadcast across the number of data
        tf.debugging.assert_equal(
            tf.shape(result), [self.num_kernels, 1, self.state_dim, self.state_dim]
        )
        return result

    def initial_covariance(self, initial_time_point: tf.Tensor) -> tf.Tensor:
        """
        Return the initial covariance of the generated
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        This is typically the covariance of the stationary distribution for :math:`x, P∞`.

        We override :meth:`SDEKernel.initial_covariance` from the
        parent class to check there is a compatible `batch_shape`.

        :param initial_time_point: The time point associated with the first state, shape
            ``batch_shape + [1,]``.
        :return: A tensor with shape ``batch_shape + [state_dim, state_dim]``,
            where ``batch_shape = (..., num_kernels)``.
        """
        batch_shape = initial_time_point.shape[:-1]
        self._check_batch_shape_is_compatible(batch_shape)
        return super().initial_covariance(initial_time_point)


@tf_scope_class_decorator
class IndependentMultiOutputStack(StackKernel):
    """
    Takes a stacked state space model consisting of multiple child kernels and projects the
    state space associated with each kernel into a separate observation vector.

    The result is similar to training several kernels on the same data separately.
    There will be no correlations between the processes, in the prior or the posterior.
    """

    def __init__(self, kernels: List[SDEKernel], jitter: float = 0.0):
        """
        :param kernels: An iterable over child kernels which will have their state spaces
            concatenated together. Since we model each output independently the length of the
            kernel list defines the number of the outputs.
        :param jitter: A small non-negative number to add into a matrix's diagonal to
            maintain numerical stability during inversion.
        """
        super().__init__(kernels, jitter)
        self._output_dim = len(kernels)

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Generate the emission matrix :math:`H`. This is a stacking of the child emission matrices,
        which are first augmented (if necessary) so that they have the same state_dim.

        :param time_points: The time points over which the emission model is defined, with shape
            ``batch_shape + [num_data]`` where ``batch_shape = (..., num_kernels)``.
        :return: The emission model associated with this kernel.
        """
        batch_shape = time_points.shape[:-1]

        # raise if batch_shape is provided and batch_shape's last dim is not num_kernel or 1
        self._check_batch_shape_is_compatible(batch_shape)

        # Each kernel's individual output_dim is 1.
        # Remember that num_kernels = self.output_dim (the overall output_dim)
        # and batch_shape = (..., num_kernels). The final shape of the emission matrix is
        # batch_shape + [num_data, 1, state_dim]
        emission_matrix = tf.stack(
            [
                augment_matrix(
                    k.generate_emission_model(time_points[..., i, :]).emission_matrix,
                    self.state_dim - k.state_dim,
                )
                for i, k in enumerate(self.kernels)
            ],
            axis=-4,
        )

        return StackEmissionModel(emission_matrix)

    def __add__(self, other: "IndependentMultiOutputStack") -> "IndependentMultiOutputStack":
        """
        Operator for combining kernel objects by summing them.

        Overrides the base class :meth:`SDEKernel.__add__` method.
        """
        assert self.output_dim == other.output_dim
        assert self.num_kernels == other.num_kernels
        summed_kernels = [k1 + k2 for k1, k2 in zip(self.kernels, other.kernels)]
        return IndependentMultiOutputStack(summed_kernels, jitter=self._jitter)

    def __mul__(self, other: "IndependentMultiOutputStack") -> "IndependentMultiOutputStack":
        """
        Operator for combining kernel objects by multiplying them.

        Overrides the base class :meth:`SDEKernel.__mul__` method.
        """
        assert self.output_dim == other.output_dim
        assert self.num_kernels == other.num_kernels
        multiplied_kernels = [k1 * k2 for k1, k2 in zip(self.kernels, other.kernels)]
        return IndependentMultiOutputStack(multiplied_kernels, jitter=self._jitter)
