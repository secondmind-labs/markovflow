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
"""Module containing a model for sparse variational inference, for use with large data sets."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.base import Parameter
from gpflow.likelihoods import Likelihood

from markovflow.base import ordered
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import SDEKernel
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowSparseModel
from markovflow.posterior import AnalyticPosteriorProcess, PosteriorProcess


class SparseVariationalGaussianProcess(MarkovFlowSparseModel):
    """
    Approximate a :class:`~markovflow.gauss_markov.GaussMarkovDistribution` with a general
    likelihood using a Gaussian posterior. Additionally uses a number of pseudo, or inducing,
    points to represent the distribution over a typically larger number of data points.

    The following notation is used:

        * :math:`x` - the time points of the training data
        * :math:`z` - the time points of the inducing/pseudo points
        * :math:`y` - observations corresponding to time points :math:`x`
        * :math:`s(.)` - the latent state of the Markov chain
        * :math:`f(.)` - the noise free predictions of the model
        * :math:`p(y | f)` - the likelihood
        * :math:`p(.)` - the true distribution
        * :math:`q(.)` - the variational distribution

    Subscript is used to denote dependence for notational convenience, for
    example :math:`fₖ === f(k)`.

    With a prior generative model comprising a Gauss-Markov distribution, an emission model and an
    arbitrary likelihood on the emitted variables, these define:

        * :math:`p(xₖ₊₁| xₖ)`
        * :math:`fₖ = H xₖ`
        * :math:`p(yₖ | fₖ)`

    As per a :class:`~markovflow.models.variational.VariationalGaussianProcess`
    (VGP) model, we have:

    .. math::
        &log p(y) >= ℒ(q)

        &ℒ(q) = Σᵢ ∫ log(p(yᵢ | f)) q(f) df - KL[q(f) ‖ p(f)]

    ...where :math:`f` is defined over the entire function space.

    Here this reduces to the joint of the evidence lower bound (ELBO) defined over both the
    data :math:`x` and the inducing points :math:`z`, which we rewrite as:

    .. math:: ℒ(q(x, z)) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) df - KL[q(f(z)) ‖ p(f(z))]

    This turns the inference problem into an optimisation problem: find the optimal :math:`q`.

    The first term is the variational expectations and have the same form as a VGP model.
    However, we must now use use the inducing states to predict the marginals of the
    variational distribution at the original data points.

    The second is the KL from the prior to the approximation, but evaluated at the inducing points.

    The key reference is::

      @inproceedings{,
          title={Doubly Sparse Variational Gaussian Processes},
          author={Adam, Eleftheriadis, Artemev, Durrande, Hensman},
          booktitle={},
          pages={},
          year={},
          organization={}
      }

    .. note:: Since this class extends :class:`~markovflow.models.models.MarkovFlowSparseModel`,
       it does not depend on input data. Input data is passed during the optimisation
       step as a tuple of time points and observations.
    """

    def __init__(
        self,
        kernel: SDEKernel,
        likelihood: Likelihood,
        inducing_points: tf.Tensor,
        mean_function: Optional[MeanFunction] = None,
        num_data: Optional[int] = None,
        initial_distribution: Optional[GaussMarkovDistribution] = None,
    ) -> None:
        """
        :param kernel: A kernel that defines a prior over functions.
        :param likelihood: A likelihood.
        :param inducing_points: The points in time on which inference should be performed,
            with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param num_data: The total number of observations.
            (relevant when feeding in external minibatches).
        :param initial_distribution: An initial configuration for the variational distribution,
            with shape ``batch_shape + [num_inducing]``.
        """
        super().__init__(self.__class__.__name__)

        self._num_data = num_data
        self._kernel = kernel  # To collect gpflow.Module trainable_variables
        self._likelihood = likelihood

        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        # by default set the inducing points to not trainable
        self.inducing_inputs = Parameter(inducing_points, transform=ordered(), trainable=False)

        if initial_distribution is None:
            initial_distribution = kernel.build_finite_distribution(inducing_points)

        # q will approximate the posterior after optimisation.
        # This needs to be an instance attribute to provide trainable variables
        # when calling gpflow.Module trainable_variables. This is fine though, since
        # `GaussMarkovDistribution` doesn't do any computation in its initialiser.
        self._dist_q = initial_distribution.create_trainable_copy()

        self._posterior = AnalyticPosteriorProcess(
            posterior_dist=self._dist_q,
            kernel=self._kernel,
            conditioning_time_points=self.inducing_inputs,
            likelihood=self._likelihood,
            mean_function=self._mean_function,
        )
        self.num_data = num_data

    def elbo(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Calculates the evidence lower bound (ELBO) :math:`log p(y)`. We rewrite this as:

        .. math:: ℒ(q(x, z)) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) df - KL[q(s(z)) ‖ p(s(z))]

        The first term is the 'variational expectation' (VE), and has the same form as per a
        :class:`~markovflow.models.variational.VariationalGaussianProcess` (VGP) model. However,
        we must now use the inducing states to predict the marginals of the
        variational distribution at the original data points.

        The second is the KL divergence from the prior to the approximation, but evaluated at the
        inducing points.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``

        :return: A scalar tensor (summed over the batch_shape dimension) representing the ELBO.
        """
        X, Y = input_data
        # predict the variational posterior at the original time points.
        # calculate sₓ ~ q(sₓ) given that we know q(s(z)), x, z,
        # and then project to function space fₓ = H*sₓ ~ q(fₓ).
        fx_mus, fx_covs = self.posterior.predict_f(X)

        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfx
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(fx_mus, fx_covs, Y)
        )
        # KL[q(s(z))|| p(s(z))]
        kl_fz = tf.reduce_sum(self._dist_q.kl_divergence(self.dist_p))

        if self._num_data is not None:
            num_data = tf.cast(self._num_data, kl_fz.dtype)
            minibatch_size = tf.cast(tf.shape(X)[-1], kl_fz.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_fz.dtype)

        # Return ELBO({fₓ, fz}) = VE(fₓ) - KL[q(s(z)) || p(s(z))]
        return ve_fx * scale - kl_fz

    @property
    def time_points(self) -> tf.Tensor:
        """
        Return the time points of the sparse process which essentially are the locations of the
        inducing points.

        :return: A tensor with shape ``batch_shape + [num_inducing]``. Same as inducing inputs.
        """
        return self.inducing_inputs

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
        return self._kernel.build_finite_distribution(self.inducing_inputs)

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
        built from the variational distribution. This will be a locally optimal
        variational approximation of the posterior after optimisation.
        """
        return self._posterior

    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Return the loss, which is the negative evidence lower bound (ELBO).

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``.
        """
        return -self.elbo(input_data)

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = input_data
        f_mean, f_var = self.predict_f(X)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)
