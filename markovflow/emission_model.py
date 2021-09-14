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
"""Module containing emission models for projection."""
from typing import Tuple

import tensorflow as tf

from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class EmissionModel:
    r"""
    Takes output from :class:`~markovflow.state_space_model.StateSpaceModel` methods and
    linearly projects it into a space of dimension :math:`m` (`output_dim`):

    .. math::
        &fₖ = Hₖ xₖ\\
        &x ∈ ℝ^d\\
        &f ∈ ℝ^m\\
        &H ∈ ℝ^{m × d}

    This class provides methods for projecting states or covariances, sampling and calculating the
    marginals.
    """

    def __init__(self, emission_matrix: tf.Tensor) -> None:
        """
        :param emission_matrix: The emission matrix that projects at each time point from the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_dim + [num_data, output_dim, state_dim]``.
        """
        if len(emission_matrix.shape) < 3:
            raise ValueError(
                f"Emission Matrix must be at least 3D but has shape {emission_matrix.shape}"
            )

        self._H = emission_matrix

    @property
    def batch_shape(self) -> tf.TensorShape:
        """
        Return the shape of any leading dimension in the emission matrix that comes before
        the last three.
        """
        return self._H.shape[:-3]

    @property
    def num_data(self) -> tf.Tensor:
        """
        Return the number of time points that the emission matrix is applied to.
        """
        return tf.shape(self._H)[-3]

    @property
    def output_dim(self) -> int:
        """
        Return the dimension of the output after the emission matrix is applied.
        """
        return self._H.shape[-2]

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the :class:`~markovflow.state_space_model.StateSpaceModel`
        we emit from.
        """
        return self._H.shape[-1]

    @property
    def emission_matrix(self) -> tf.Tensor:
        """
        Return the emission matrix.

        :return: A tensor for the emission matrix, with shape
            ``batch_dim + [num_data, output_dim, state_dim]``.
        """
        return self._H

    def project_state_marginals_to_f(
        self, means: tf.Tensor, covariances: tf.Tensor, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Project the marginal mean and covariance of states to get means and (co)variance
        of :math:`f`.

        :param means: A tensor of means with shape
                    ``batch_shape + [num_data, state_dim]``.
        :param covariances: A tensor of covariances with shape
                    ``batch_shape + [num_data, state_dim, state_dim]``.
        :param full_output_cov: Full output covariance (`True`) or marginal variances (`False`).
        :return: The means and covariances with respective shapes
            ``batch_shape + [num_data, output_dim]``,
            and either ``batch_shape + [num_data, output_dim, output_dim]``
            or ``batch_shape + [num_data, output_dim]``.
        """
        return (
            self.project_state_to_f(means),
            self.project_state_covariance_to_f(covariances, full_output_cov),
        )

    def project_state_to_f(self, state: tf.Tensor) -> tf.Tensor:
        """
        Project a state to :math:`f` by multiplying by :math:`H`.

        :param state: A tensor with shape ``batch_shape + [num_data, state_dim]``.
        :return: A tensor with shape ``batch_shape + [num_data, output_dim]``.
        """
        tf.debugging.assert_shapes(
            [
                (state, (..., "num_data", "state_dim")),
                (self._H, (..., "num_data", "num_outputs", "state_dim")),
            ]
        )
        return tf.matmul(self._H, state[..., None])[..., 0]

    def project_state_covariance_to_f(
        self, covariance: tf.Tensor, full_output_cov: bool = False
    ) -> tf.Tensor:
        """
        Project a state covariance :math:`S` to an :math:`f` covariance by calculating
        :math:`HSHᵀ` (or its diagonal).

        :param covariance: A tensor with shape
                    ``batch_shape + [num_data, state_dim, state_dim]``.
        :param full_output_cov: Full output covariance (`True`) or marginal variances (`False`).
        :return: A tensor either with shape ``batch_shape + [num_data, output_dim, output_dim]``
                or ``batch_shape + [num_data, output_dim]``.
        """
        tf.debugging.assert_equal(
            tf.shape(covariance)[-3:], (self.num_data, self.state_dim, self.state_dim)
        )
        if full_output_cov:
            # [... output_dim, output_dim]
            return tf.einsum("...ij,...jk,...lk->...il", self._H, covariance, self._H)
        else:
            # [... output_dim]
            return tf.einsum(
                "...ij,...ij->...i", self._H, tf.einsum("...ij,...jk->...ik", self._H, covariance),
            )


@tf_scope_class_decorator
class ComposedPairEmissionModel(EmissionModel):
    r"""
    Linear projection for use with kernels that have an intermediate projection. That is,
    there exists a projection from the state space to an intermediate space, and from that space to
    the observation space:

    .. math::
        &fₖ = Hₒₖ gₖ = Hₒₖ Hₗ xₖ = Hₖ xₖ\\
        &gₖ = Hₗ xₖ\\
        &x ∈ ℝ^d\\
        &g ∈ ℝ^l \verb|, the inner space|\\
        &f ∈ ℝ^m \verb|, the outer space|\\
        &H ∈ ℝ^{m × d}\\
        &Hₒ ∈ ℝ^{m × l}\\
        &Hₗ ∈ ℝ^{l × d}

    This class provides methods for projecting states or covariances, sampling and calculating the
    marginals, from state space to both the observation and intermediate space.
    """

    def __init__(
        self, outer_emission_model: EmissionModel, inner_emission_model: EmissionModel
    ) -> None:
        """
        :param outer_emission_model: The emission model for projecting from the intermediate
            space to the observation space.
        :param inner_emission_model: The emission model for projecting from state space to the
                intermediate space.
        """
        super().__init__(
            tf.einsum(
                "...ij,...jk->...ik",
                outer_emission_model.emission_matrix,
                inner_emission_model.emission_matrix,
            )
        )
        self._inner_emission_model = inner_emission_model

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the
        :class:`~markovflow.state_space_model.StateSpaceModel` we emit from.
        """
        return self._inner_emission_model.state_dim

    @property
    def inner_dim(self) -> int:
        """
        Return the output dimension of the inner emission model.
        """
        return self._inner_emission_model.output_dim

    @property
    def inner_emission_matrix(self) -> tf.Tensor:
        """
        Return the emission matrix used for projecting from the state space
        to the intermediate space.

        :return: A tensor for the emission matrix, with shape
            ``batch_dim + [num_data, inner_dim, state_dim]``.
        """
        return self._inner_emission_model.emission_matrix

    def project_state_marginals_to_g(
        self, means: tf.Tensor, covariances: tf.Tensor, full_output_cov: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Project the marginal mean and covariance of states to get means for :math:`g`.

        :param means: A tensor of means with shape
                    ``batch_shape + [num_data, state_dim]``.
        :param covariances: A tensor of covariances with shape
                    ``batch_shape + [num_data, state_dim, state_dim]``.
        :param full_output_cov: Full output covariance (`True`) or marginal variances (`False`).
        :return: The means and covariances with respective shapes
                ``batch_shape + [num_data, inner_dim]``,
                and either ``batch_shape + [num_data, inner_dim, inner_dim]``
                or ``batch_shape + [num_data, inner_dim]``.
        """
        return self._inner_emission_model.project_state_marginals_to_f(
            means, covariances, full_output_cov
        )

    def project_state_to_g(self, state: tf.Tensor) -> tf.Tensor:
        """
        Project a state to :math:`g` by multiplying by the inner emission matrix.

        :param state: A tensor with shape
                    ``batch_shape + [num_data, state_dim]``.
        :return: A tensor with shape
                    ``batch_shape + [num_data, inner_dim]``.
        """
        return self._inner_emission_model.project_state_to_f(state)

    def project_state_covariance_to_g(
        self, covariance: tf.Tensor, full_output_cov: bool = True
    ) -> tf.Tensor:
        """
        Project a state covariance :math:`S` to a :math:`g` covariance by
        calculating :math:`HSHᵀ` with the inner :math:`H`.

        :param covariance: A tensor with shape
                    ``batch_shape + [num_data, state_dim, state_dim]``.
        :param full_output_cov: Full output covariance (`True`) or marginal variances (`False`).
        :return: A tensor either with shape
                ``batch_shape + [num_data, inner_dim, inner_dim]``
                or ``batch_shape + [num_data, inner_dim]``.
        """
        return self._inner_emission_model.project_state_covariance_to_f(covariance, full_output_cov)


@tf_scope_class_decorator
class StackEmissionModel(EmissionModel):
    r"""
    Linear projection for use with a :class:`~markovflow.kernels.sde_kernel.StackKernel`, where we
    implicitly assume that we have parallel independent SDEs that model each one of the output
    dimensions.

    In such a scenario we assume that the :math:`m` (`output_dim`) independent SDEs can be broadcast
    together so the `output_dim` is part of the `batch_shape` (last dim in the `batch_shape`).

    So the emission matrix that defines the `StackEmissionModel` has the following shape:

    ``batch_shape + [num_data, 1, state_dim]``

    ...where ``batch_shape = (..., num_kernels)`` and ``num_kernels = output_dim``.
    The singleton pre-last dimension is for the individual `output_dim` of each kernel.
    Remember that each kernel explicitly models one of the output dimensions.

    We effectively run :math:`m` (`output_dim`) independent SDEs as follows:

    .. math::
        &fₖ⁽ᵐ⁾ = Hₖ⁽ᵐ⁾ xₖ⁽ᵐ⁾\\
        &x⁽ᵐ⁾ ∈ ℝᵈ\\
        &f⁽ᵐ⁾∈ ℝ\\
        &H⁽ᵐ⁾ ∈ ℝ^{1 × d}

    This class provides methods for projecting states or covariances and calculating the marginals
    from the state space to the observation space. It is acting as the base :class:`EmissionModel`
    class with an extra transposition in the end to make sure that the `output_dim` gets moved from
    the `batch_shape` to the last dim of the projected matrices.
    """

    def __init__(self, emission_matrix: tf.Tensor) -> None:
        """
        :param emission_matrix: The emission matrix that projects from the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, 1, state_dim]`` where ``batch_shape = (..., num_kernels)``.
        """
        if len(emission_matrix.shape) < 4:
            raise ValueError(
                f"Emission Matrix must be at least 4D but has shape {emission_matrix.shape}"
            )

        # assert that all kernels have individual output_dim 1
        individual_output_dim = tf.shape(emission_matrix)[-2]
        tf.debugging.assert_equal(individual_output_dim, 1)

        super().__init__(emission_matrix)

    @property
    def output_dim(self) -> int:
        """
        Return the dimension of the output after the emission matrix is applied.
        """
        return self.batch_shape[-1]

    def project_state_to_f(self, state: tf.Tensor) -> tf.Tensor:
        """
        Project each of the `num_kernel` states :math:`s` to :math:`f` by multiplying by
        the corresponding :math:`H`.

        :param state: A tensor with shape ``batch_shape + [num_data, state_dim]``
            where ``batch_shape = (..., num_kernels)`` and ``num_kernels = output_dim``.
        :return: A tensor with shape ``batch_shape[:-1] + [num_data, output_dim]``.
        """
        tf.debugging.assert_shapes(
            [
                (state, (..., "num_data", "state_dim")),
                (self._H, (..., "num_data", "num_outputs", "state_dim")),
            ]
        )

        # [..., num_kernels, num_data, 1, state_dim] @ [..., num_kernels, num_data, state_dim, 1]
        projection = (self._H @ state[..., None])[..., 0, 0]
        # [..., num_data, output_dim]
        return tf.linalg.matrix_transpose(projection)

    def project_state_covariance_to_f(
        self, covariance: tf.Tensor, full_output_cov: bool = False
    ) -> tf.Tensor:
        """
        Project a state covariance :math:`S` to an :math:`f` covariance by calculating
        :math:`HSHᵀ` (or its diagonal). If it is called with `full_output_cov=True` it will return
        the same as `full_output_cov=False` but in a compatible (diagonal) shape.

        :param covariance: A tensor with shape ``batch_shape + [num_data, state_dim, state_dim]``
            where ``batch_shape = (..., num_kernels)`` and ``num_kernels = output_dim``
        :param full_output_cov: Full output covariance (`True`) or marginal variances (`False`).
        :return: A tensor either with shape
            ``batch_shape[:-1] + [num_data, output_dim, output_dim]``
            or ``batch_shape[:-1] + [num_data, output_dim]``.
        """
        shape = tf.concat(
            [self.batch_shape, [self.num_data, self.state_dim, self.state_dim]], axis=0
        )
        tf.debugging.assert_equal(tf.shape(covariance), shape)

        # N = num_data and D = state_dim
        # [..., num_kernels, N, 1, D] * [..., num_kernels, N, D, D] * [..., num_kernels, N, D, 1]
        # = [..., num_kernels, N]
        HcovHT = tf.reduce_sum(
            self._H * covariance * tf.linalg.matrix_transpose(self._H), axis=[-2, -1]
        )

        # [..., num_data, output_dim]
        diag_HcovHT = tf.linalg.matrix_transpose(HcovHT)
        if full_output_cov:
            return tf.linalg.diag(diag_HcovHT)
        else:
            return diag_HcovHT
