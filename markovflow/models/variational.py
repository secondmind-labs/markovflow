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
"""Module containing a model for variational inference, for GP classification."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.likelihoods import Likelihood

from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import SDEKernel
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowModel
from markovflow.posterior import AnalyticPosteriorProcess, PosteriorProcess


class VariationalGaussianProcess(MarkovFlowModel):
    """
    Approximates a :class:`~markovflow.gauss_markov.GaussMarkovDistribution`
    with a general likelihood using a Gaussian posterior.

    The following notation is used:

        * :math:`x` - the time points of the training data
        * :math:`y` - observations corresponding to time points :math:`x`
        * :math:`s(.)` - the latent state of the Markov chain
        * :math:`f(.)` - the noise free predictions of the model
        * :math:`p(y | f)` - the likelihood
        * :math:`p(.)` - the true distribution
        * :math:`q(.)` - the variational distribution

    Subscript is used to denote dependence for notational convenience,
    for example :math:`fₖ === f(k)`.

    With a prior generative model comprising a Gauss-Markov distribution, an emission model and an
    arbitrary likelihood on the emitted variables, these define:

        * :math:`p(xₖ₊₁| xₖ)`
        * :math:`fₖ = H xₖ`
        * :math:`p(yₖ | fₖ)`

    We would like to approximate the posterior of this generative model with a parametric
    model :math:`q`, comprising of the same distribution as the prior.

    To approximate the posterior, we maximise the evidence lower bound (ELBO) :math:`ℒ` with
    respect to the parameters of the variational distribution, since:

    .. math:: log p(y) = ℒ(q) + KL[q ‖ p(f | y)]

    ...where:

    .. math:: ℒ(q) = ∫ log(p(f, y) / q(f)) q(f) df

    Since the last term is non-negative, the ELBO provides a lower bound to the log-likelihood of
    the model. This bound is exact when :math:`KL[q ‖ p(f | y)] = 0`; that is, our approximation is
    sufficiently flexible to capture the true posterior.

    This turns the inference into an optimisation problem: find the optional :math:`q`.

    To calculate the ELBO, we rewrite it as:

    .. math:: ℒ(q) = Σᵢ ∫ log(p(yᵢ | f)) q(f) df - KL[q(f) ‖ p(f)]

    The first term is the 'variational expectation' of the model likelihood;
    the second is the KL from the prior to the approximation.
    """

    def __init__(
        self,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        kernel: SDEKernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        initial_distribution: Optional[GaussMarkovDistribution] = None,
    ) -> None:
        """
        :param input_data: A tuple of ``(time_points, observations)`` containing the observed data:
            time points of observations, with shape ``batch_shape + [num_data]``,
            observations with shape ``batch_shape + [num_data, observation_dim]``.
        :param kernel: A kernel that defines a prior over functions.
        :param likelihood: A likelihood.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param initial_distribution: An initial configuration for the variational distribution,
            with shape ``batch_shape + [num_inducing]``.
        """
        super().__init__(self.__class__.__name__)
        time_points, observations = input_data

        # To collect kernel and mean function gpflow.Module trainable_variables
        self._kernel = kernel
        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        self._likelihood = likelihood

        self._time_points = time_points
        self._observations = observations

        if initial_distribution is None:
            initial_distribution = kernel.build_finite_distribution(time_points)

        # q will approximate the posterior after optimisation.
        # This needs to be an instance attribute to provide trainable variables
        # when calling gpflow.Module trainable_variables. This is fine though, since
        # StateSpaceModel doesn't do any computation in its initialiser.
        self._dist_q = initial_distribution.create_trainable_copy()

        self._posterior = AnalyticPosteriorProcess(
            posterior_dist=self._dist_q,
            kernel=self._kernel,
            conditioning_time_points=self._time_points,
            likelihood=self._likelihood,
            mean_function=self._mean_function,
        )

    def elbo(self) -> tf.Tensor:
        """
        Calculate the evidence lower bound (ELBO) :math:`log p(y)`. We rewrite the ELBO as:

        .. math:: ℒ(q(x)) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) df - KL[q(sₓ) ‖ p(sₓ)]

        The first term is the 'variational expectation' (VE); the second is the KL divergence from
        the prior to the approximation.

        :return: A scalar tensor (summed over the batch_shape dimension) representing the ELBO.
        """
        # s ~ q(s) = N(μ, P)
        # Project to function space, fₓ = H*s ~ q(fₓ)
        fx_mus, fx_covs = self.posterior.predict_f(self._time_points)
        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(
                fx_mus, fx_covs, self._observations
            )
        )
        # KL[q(sₓ) || p(sₓ)]
        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))
        # Return ELBO(fₓ) = VE(fₓ) - KL[q(sₓ) || p(sₓ)]
        return ve_fx - kl_fx

    @property
    def time_points(self) -> tf.Tensor:
        """
        Return the time points of our observations.

        :return: A tensor with shape ``batch_shape + [num_data]``.
        """
        return self._time_points

    @property
    def observations(self) -> tf.Tensor:
        """
        Return the observations.

        :return: A tensor with shape ``batch_shape + [num_data, observation_dim]``.
        """
        return self._observations

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel of the GP.
        """
        return self._kernel

    @property
    def likelihood(self) -> Likelihood:
        """
        Return the likelihood of the GP.
        """
        return self._likelihood

    @property
    def mean_function(self) -> MeanFunction:
        """
        Return the mean function of the GP.
        """
        return self._mean_function

    @property
    def dist_p(self) -> GaussMarkovDistribution:
        """
        Return the prior Gauss-Markov distribution.
        """
        return self._kernel.build_finite_distribution(self._time_points)

    @property
    def dist_q(self) -> GaussMarkovDistribution:
        """
        Return the variational distribution as a Gauss-Markov distribution.
        """
        return self._dist_q

    @property
    def posterior(self) -> PosteriorProcess:
        """
        Obtain a posterior process for inference.

        For this class this is the :class:`~markovflow.posterior.AnalyticPosteriorProcess`
        built from the variational distribution. This will be a locally optimal variational
        approximation of the posterior after optimisation.
        """
        return self._posterior

    def loss(self) -> tf.Tensor:
        """
        Return the loss, which is the negative ELBO.
        """
        return -self.elbo()
