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
"""Module representing a Gauss-Markov chain."""
from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
import gpflow

from markovflow.base import SampleShape
from markovflow.block_tri_diag import SymmetricBlockTriDiagonal
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class GaussMarkovDistribution(gpflow.Module, ABC):
    """
    Abstract class for representing a Gauss-Markov chain. Classes that extend this one (such as
    :class:`~markovflow.state_space_model.StateSpaceModel`) represent a different parameterisation
    of the joint Gaussian distribution.
    """

    @property
    @abstractmethod
    def event_shape(self) -> tf.Tensor:
        """
        Return the shape of the event in the Gauss-Markov chain that is
        ``[num_transitions + 1, state_dim]``.
        """

    @property
    @abstractmethod
    def batch_shape(self) -> tf.TensorShape:
        """
        Return the shape of any leading dimensions in the Gauss-Markov chain that come before
        :attr:`event_shape`.
        """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Return the state dimension of the Gauss-Markov chain.
        """

    @property
    @abstractmethod
    def num_transitions(self) -> tf.Tensor:
        """
        Return the number of transitions in the Gauss-Markov chain.
        """

    @abstractmethod
    def _build_precision(self) -> SymmetricBlockTriDiagonal:
        """
        Compute the compact banded representation of the precision matrix.
        """

    @property
    def precision(self) -> SymmetricBlockTriDiagonal:
        """
        Return the precision matrix of the joint Gaussian.
        """
        return self._build_precision()

    @property
    @abstractmethod
    def marginal_means(self) -> tf.Tensor:
        """
        Return the marginal means of the joint Gaussian.

        :return: A tensor with shape ``batch_shape + [num_transitions + 1, state_dim]``.
        """

    @property
    @abstractmethod
    def marginal_covariances(self) -> tf.Tensor:
        """
        Return the marginal covariances of the joint Gaussian.

        :return: A tensor with shape ``batch_shape + [num_transitions + 1, state_dim, state_dim]``.
        """

    @abstractmethod
    def covariance_blocks(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the diagonal and lower off-diagonal blocks of the covariance.

        :return: A tuple of tensors, with respective shapes
                ``batch_shape + [num_transitions + 1, state_dim]``,
                ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """

    @property
    def marginals(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the means :math:`μₖ` and the covariances :math:`Σₖₖ` of the marginal distributions
        over consecutive states :math:`xₖ`.

        :return: The means and covariances, with respective shapes
                 ``batch_shape + [num_transitions + 1, state_dim]``,
                 ``batch_shape + [num_transitions + 1, state_dim, state_dim]``.
        """
        return self.marginal_means, self.marginal_covariances

    @abstractmethod
    def sample(self, sample_shape: SampleShape) -> tf.Tensor:
        """
        Sample trajectories from the distribution.

        :param sample_shape: The shape (and hence number of) trajectories to sample from
            the distribution.

        :return: The state samples, with shape
            ``sample_shape + self.batch_shape + self.event_shape``.
        """

    @abstractmethod
    def log_det_precision(self) -> tf.Tensor:
        """
        Calculate the log determinant of the precision matrix.

        :return: A tensor with shape ``batch_shape``.
        """

    @abstractmethod
    def log_pdf(self, states) -> tf.Tensor:
        r"""
        Return the value of the log of the PDF evaluated at states.

        That is:

        .. math:: log p(x) = log p(x₀) + Σₖ log p(xₖ₊₁|xₖ)  \verb|(for 0 ⩽ k < n)|

        :param states: The state trajectory, with shape
            ``sample_shape + self.batch_shape + self.event_shape``.

        :return: The log pdf, with shape ``sample_shape + self.batch_shape``.
        """

    @abstractmethod
    def create_trainable_copy(self) -> "GaussMarkovDistribution":
        """
        Create a trainable version.

        This is primarily for use with variational approaches where we want to optimise
        the parameters of the Gauss-Markov distribution.

        :return: A Gauss-Markov distribution that is a copy of this one with trainable parameters.
        """

    @abstractmethod
    def create_non_trainable_copy(self) -> "GaussMarkovDistribution":
        """
        Create a non-trainable version.

        Convert a trainable version of this class back to being non-trainable.

        :return: A Gauss-Markov distribution that is a copy of this one.
        """

    @abstractmethod
    def kl_divergence(self, dist: "GaussMarkovDistribution") -> tf.Tensor:
        r"""
        Return the KL divergence of the current Gauss-Markov distribution from the specified
        input `dist`:

        .. math:: KL(dist₁ ∥ dist₂)

        To do so we first compute the marginal distributions from the Gauss-Markov form:

        .. math::
            dist₁ = 𝓝(μ₁, P⁻¹₁)\\
            dist₂ = 𝓝(μ₂, P⁻¹₂)

        ...where :math:`μᵢ` are the marginal means and :math:`Pᵢ` are the banded precisions.

        The KL divergence is then given by:

        .. math::
            KL(dist₁ ∥ dist₂) = ½(tr(P₂P₁⁻¹) + (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁) - N - log(|P₂|) + log(|P₁|))

        ...where :math:`N = (\verb|num_transitions| + 1) * \verb|state_dim|` (that is,
        the dimensionality of the Gaussian).

        :param dist: Another similarly-parameterised Gauss-Markov distribution.
        :return: The KL divergences, with shape ``self.batch_shape``.
        """


def check_compatible(dist_1: GaussMarkovDistribution, dist_2: GaussMarkovDistribution) -> None:
    """
    Check that two :class:`~markovflow.gauss_markov.GaussMarkovDistribution` objects are
    compatible.

    If not, raise an exception.
    """

    assert isinstance(dist_2, type(dist_1)), TypeError(
        """`dist_2` has different representation than `dist_1`"""
    )
    tf.debugging.assert_equal(dist_1.state_dim, dist_2.state_dim)
    tf.debugging.assert_equal(dist_1.batch_shape, dist_2.batch_shape)
    tf.debugging.assert_equal(dist_1.num_transitions, dist_2.num_transitions)
