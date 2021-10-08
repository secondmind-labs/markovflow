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
"""Module containing a multivariate Gaussian likelihood."""

from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.base import Parameter, TensorType
from gpflow.utilities import triangular

from markovflow.likelihoods.likelihoods import Likelihood, check_input_shapes
from markovflow.utils import tf_scope_class_decorator


@tf_scope_class_decorator
class MultivariateGaussian(Likelihood):
    """
    Represents a multivariate Gaussian likelihood. For example:

    .. math:: p(yᵢ | fᵢ) = 𝓝(yᵢ; fᵢ, Σ= LLᵀ)

    See also the documentation for the base
    :class:`~markovflow.likelihoods.likelihoods.Likelihood` class.
    """

    def __init__(self, chol_covariance: TensorType):
        """
        :param chol_covariance: A :data:`~markovflow.base.TensorType` containing the Cholesky
            factor of the covariance of the Gaussian noise, with shape ``[obs_dim, obs_dim]``.
        """
        super().__init__(self.__class__.__name__)
        self.chol_covariance = Parameter(
            chol_covariance, transform=triangular(), name="chol_covariance"
        )
        self._obs_dim = self.chol_covariance.shape[-1]

        tf.debugging.assert_shapes(
            [(self.chol_covariance, [..., self.obs_dim, self.obs_dim]),]
        )

    @property
    def obs_dim(self) -> int:
        """
        Return the dimensionality of each observation.
        """
        return self._obs_dim

    def log_probability_density(self, fs: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
        """
        Compute the log probability density :math:`log p(Y|F)`.

        For a multivariate Gaussian, this is :math:`log 𝓝(yᵢ; fᵢ, Σ)`.

        :param fs: A tensor representing a conditioning variable, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param observations: A tensor representing a conditioned variable,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :return: A tensor representing :math:`log p(yᵢ | fᵢ)`,
            with shape ``batch_shape + [num_data]``.
        """
        return tfp.distributions.MultivariateNormalTriL(fs, self.chol_covariance).log_prob(
            observations
        )

    def variational_expectations(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor, observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate a variational expectation for each observation:

        .. math:: ∫ q(fᵢ) log p(yᵢ|fᵢ) dfᵢ

        ...where:

            * :math:`q(fᵢ) ~ N(μᵢ, Σᵢ)`
            * :math:`p(y |f)` is a general likelihood function

        For a multivariate Gaussian this is:

        .. math:: ∫ 𝓝(fᵢ; μᵢ, Sᵢ) log𝓝(yᵢ; fᵢ, Σ= LLᵀ) dfᵢ = -½ Tr(Σ⁻¹Sᵢ) + log𝓝(yᵢ; μᵢ, Σ)

        :param f_means: The marginal :math:`f` means for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :param observations: The :math:`y` values at which to evaluate the log probability,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        Y = observations
        check_input_shapes(f_means, Y, self.obs_dim, f_covariances=f_covariances)

        inv_covariance = tf.linalg.cholesky_solve(
            self.chol_covariance, tf.eye(self.obs_dim, dtype=default_float())
        )
        return -0.5 * tf.reduce_sum(
            inv_covariance * f_covariances, axis=[-1, -2]
        ) + tfp.distributions.MultivariateNormalTriL(f_means, self.chol_covariance).log_prob(Y)

    def predict_density(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor, observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Predict a density. This calculates:

        .. math:: ∫ q(F) p(Y|F) dF

        ...of a Gaussian approximation :math:`q(F) ~ N(μ, Σ)` to the posterior
        density :math:`p(F|Y)`.

        For a multivariate Gaussian this is:

        .. math:: log ∫ 𝓝(fᵢ; μᵢ, Sᵢ) 𝓝(yᵢ; fᵢ, Σ= LLᵀ) dfᵢ = log 𝓝(yᵢ; μᵢ, Σ + Sᵢ)

        :param f_means: The marginal :math:`f` means for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :param observations: The :math:`y` values at which to evaluate the log probability,
            with shape ``batch_shape + [num_data, obs_dim]``.
        """
        Y = observations
        check_input_shapes(f_means, Y, self.obs_dim, f_covariances=f_covariances)

        covariance = tf.matmul(self.chol_covariance, self.chol_covariance, transpose_b=True)
        f_chol_covariances = tf.linalg.cholesky(f_covariances + covariance)
        return tfp.distributions.MultivariateNormalTriL(f_means, f_chol_covariances).log_prob(Y)

    def predict_mean_and_var(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict the observation means and covariances given the f-space means and covariances.

        That is, calculate:

        .. math:: p(y* | x*, x, y) = ∫ p(y* | f*) p(f* | x*, x, y) df*

        ...where:

            * `f_means` and `f_covariances` is our representation of :math:`p(f* | x*, x, y)`
            * :math:`p(y* | f*)` is defined by the likelihood

        For a multivariate Gaussian this is:

        .. math:: p(y* | x*, x, y) = ∫ 𝓝(f*; μ*, S*) 𝓝(y*; f*, Σ= LLᵀ) dfᵢ = 𝓝(y*; μ*, Σ + S*)

        :param f_means: The marginal :math:`f` means for some arbitrary predicted time points,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for some arbitrary predicted
            time points, with shape ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :return: A tuple of tensors containing observation means and covariances, with
            respective shapes
            ``batch_shape + [num_time_points, obs_dim]``,
            ``batch_shape + [num_time_points, obs_dim, obs_dim]``.
        """
        covariance = tf.matmul(self.chol_covariance, self.chol_covariance, transpose_b=True)
        return f_means, covariance + f_covariances
