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
"""Module containing base classes for likelihoods."""

import abc
from abc import abstractmethod
from typing import Tuple

import gpflow
import tensorflow as tf

from markovflow.utils import tf_scope_fn_decorator


class Likelihood(gpflow.Module, abc.ABC):
    """
    Abstract class for likelihoods.

    A likelihood defines the observation model relating the observed variables :math:`Y`
    to the latent variables :math:`F` of a generative model. The observation model is specified
    through its conditional density :math:`p(Y|F)`.

    In order to perform variational inference with non-Gaussian likelihoods, a 'variational
    expectation' should be computed under a Gaussian distribution :math:`q(F) ~ N(μ, Σ)`.
    This can be defined as:

    .. math:: ∫ q(F) log p(Y|F) dF

    Note that the predictive density:

    .. math:: ∫ q(F) p(Y|F) dF

    ...is a useful metric to evaluate the quality of a Gaussian approximation
    :math:`q(F) ~ N(μ, Σ)` to the posterior density :math:`p(F|Y)`.

    .. note:: Implementations of this class should typically avoid performing computation in their
        `__init__` method. Performing computation in the constructor conflicts with
        running in TensorFlow's eager mode (and computation of gradients etc).
    """

    @abstractmethod
    def log_probability_density(self, fs: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
        """
        Compute the log probability density :math:`log p(Y|F)`.

        :param fs: A conditioning variable, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param observations: A conditioned variable,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :return: A tensor representing :math:`log p(yᵢ | fᵢ)`,
            with shape ``batch_shape + [num_data]``.
        """
        raise NotImplementedError

    @abstractmethod
    def variational_expectations(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor, observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate a variational expectation for each observation:

        .. math:: ∫ log(p(yᵢ|fᵢ)) q(fᵢ) df

        ...where :math:`q(f) ~ N(μ, P)`.

        Note that :math:`p(y |f)` is defined by the type of likelihood function, as
        specified by the observation model.

        This term is used when calculating the evidence lower bound (ELBO):

        .. math:: ℒ(q) = Σᵢ ∫ log(p(yᵢ|fᵢ)) q(fᵢ) df - KL[q(F) ‖ p(F)]

        :param f_means: The marginal :math:`f` means for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with
            shape ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :param observations: The :math:`y` values at which to evaluate the log probability,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_density(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor, observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Predict the density.

        That is, calculate :math:`∫ q(F) p(Y|F) dF` of a Gaussian
        approximation :math:`q(F) ~ N(μ, Σ)` to the posterior density :math:`p(F|Y)`.

        :param f_means: The marginal :math:`f` means for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`, with shape
            ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for each state of the
            :class:`~markovflow.state_space_model.StateSpaceModel`,
            with shape ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :param observations: The :math:`y` values at which to evaluate the log probability,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_mean_and_var(
        self, f_means: tf.Tensor, f_covariances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the means and covariances of the posterior predictive distribution over outputs
        :math:`y*` at :math:`x*`.

        The (in most case intractable) density is given by:

        .. math:: p(y* | x*, x, y) = ∫ p(y* | f*) p(f* | x*, x, y) df*

        ...where:

            * :math:`p(f* | x*, x, y)` is Gaussian with moments `f_means` and `f_covariances`
            * :math:`p(y* | f*)` is defined by the likelihood

        :param f_means: The marginal :math:`f` means for some arbitrary predicted time points,
            with shape ``batch_shape + [num_data, obs_dim]``.
        :param f_covariances: The marginal :math:`f` covariances for some arbitrary predicted time
            points, with shape ``batch_shape + [num_data, obs_dim, obs_dim]``.
        :return: A tuple of tensors containing observation means and covariances, with
            respective shapes
            ``batch_shape + [num_time_points, obs_dim]``,
            ``batch_shape + [num_time_points, obs_dim, obs_dim]``.
        """
        raise NotImplementedError


class PEPScalarLikelihood(gpflow.likelihoods.ScalarLikelihood):
    """
    Wrapper around GPflow likelihoods, adding
    functionality to compute Power Expectation Propagation updates
    """

    def __init__(
        self, base: gpflow.likelihoods.ScalarLikelihood, num_gauss_hermite_points=20, **kwargs
    ):
        """
        :param base: base likelihood object
        :param num_gauss_hermite_points: number of Gauss-Hermite points
        :param kwargs: additional arguments dictionary
        """
        super().__init__(**kwargs)
        self.base = base
        self.quadrature = gpflow.quadrature.NDiagGHQuadrature(1, num_gauss_hermite_points)

    def _scalar_log_prob(self, F, Y):
        r"""
        Compute log p(Y|F).
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        return self.base.log_prob(F, Y)

    def _scalar_alpha_prob(self, F, Y, alpha=1.0):
        r"""
        Compute p(Y|F)
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        :param alpha: scalar
        """
        return tf.exp(self._scalar_log_prob(F, Y) * alpha)

    def log_expected_density(self, Fmu, Fvar, Y, alpha=1.0):
        r"""
        Compute log ∫ p(y=Y|f)ᵃ q(f) df, where  q(f) = N(Fmu, Fvar)
        :param Fmu: mean function evaluation Tenself._quadrature_reduction(
                self.quadrature.logspace(self._scalar_log_prob, Fmu, Fvar, Y=Y)
        )sor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :param alpha: scalar
        """
        return self.base.predict_log_density(Fmu, Fvar, Y=Y)

    def grad_log_expected_density(self, Fmu, Fvar, Y, alpha=1.0):
        """
        Noting I(q) = log ∫ p(y=Y|f)ᵃ q(f) df, where  q(f) = N(Fmu, Fvar),
        this computes ∇I(q) and ∇∇I(q), where the gradient is wrt Fmu.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :param alpha: scalar
        """
        with tf.GradientTape() as g:
            g.watch(Fmu)
            with tf.GradientTape() as gg:
                gg.watch(Fmu)
                led = self.log_expected_density(Fmu, Fvar, Y, alpha=alpha)
            dled_dfmu = gg.gradient(led, Fmu)
        d2led_dfmu2 = g.gradient(dled_dfmu, Fmu)
        return led, (dled_dfmu, d2led_dfmu2)

    def _conditional_mean(self, F):
        """ The conditional mean of Y|F """
        raise NotImplementedError

    def _conditional_variance(self, F):
        """ The conditional variance of Y|F """
        raise NotImplementedError


class PEPGaussian(PEPScalarLikelihood):
    """
    Wrapper around the univariate Gaussian Likelihood.
    """

    def __init__(self, base: gpflow.likelihoods.Gaussian, **kwargs):
        """
        :param base: A Gaussian Likelihood object
        :param kwargs: dictionary of additional parameters
        """
        assert isinstance(base, gpflow.likelihoods.Gaussian)
        super().__init__(base, **kwargs)

    def log_expected_density(self, Fmu, Fvar, Y, alpha=1.0):
        r"""
        Compute log ∫ p(y=Y|f)ᵃ q(f) df, where  q(f) = N(f;Fmu, Fvar)

        log ∫ p(y=Y|f)ᵃ q(f) df
        =  log ∫ N(y; f, σ²) ᵃ N(f; Fmu, Fvar) df
        =  log N(y; Fmu, σ² + Fvar)

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        :param alpha: scalar
        """
        return alpha * tf.squeeze(
            gpflow.logdensities.gaussian(Y, Fmu, self.base.variance + Fvar), axis=-1
        )

    def grad_log_expected_density(self, Fmu, Fvar, Y, alpha=1.0):
        """
        Noting I(q) = log ∫ p(y=Y|f)ᵃ q(f) df, where  q(f) = N(Fmu, Fvar),
        this computes ∇I(q) and ∇∇I(q), where the gradient is wrt Fmu.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :param alpha: scalar
        """
        val = self.log_expected_density(Fmu, Fvar, Y, alpha)
        var = self.base.variance + Fvar
        grads = (alpha * (Y - Fmu) / var, -alpha * 1.0 / var)  # * (1. - ((Y - Fmu) ** 2) / var)
        return val, grads

    def _conditional_mean(self, F):
        """ The conditional mean of Y|F """
        raise NotImplementedError

    def _conditional_variance(self, F):
        """ The conditional variance of Y|F """
        raise NotImplementedError


@tf_scope_fn_decorator
def check_input_shapes(
    f_means: tf.Tensor,
    observations: tf.Tensor,
    expected_obs_dim: int,
    f_covariances: tf.Tensor = None,
) -> None:
    """
    Check that the shapes of inputs to likelihood methods are valid.

    :param f_means: A tensor with shape ``batch_shape + [num_data, obs_dim]``.
    :param observations: A tensor with shape ``batch_shape + [num_data, obs_dim]``.
    :param expected_obs_dim: The expected number of dimensions.
    :param f_covariances: A tensor with shape ``batch_shape + [num_data, obs_dim, obs_dim]``.
    """
    shape_list = [
        (observations, (..., "num_data", expected_obs_dim)),
        (f_means, (..., "num_data", expected_obs_dim)),
    ]
    tf.debugging.assert_shapes(shape_list)
    if f_covariances is not None:
        shape_list_cov = [(f_covariances, (..., "num_data", expected_obs_dim, expected_obs_dim))]
        tf.debugging.assert_shapes(shape_list_cov)
