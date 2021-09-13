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
"""Module containing a model for importance-weighted variational inference."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.likelihoods import Likelihood

from markovflow.kernels import SDEKernel
from markovflow.mean_function import MeanFunction
from markovflow.models.sparse_variational import SparseVariationalGaussianProcess
from markovflow.posterior import ImportanceWeightedPosteriorProcess
from markovflow.state_space_model import StateSpaceModel


class ImportanceWeightedVI(SparseVariationalGaussianProcess):
    """
    Performs importance-weighted variational inference (IWVI).

    The `key reference <https://papers.nips.cc/paper/7699-importance-weighting-and-
    variational-inference.pdf>`_ is::

        @inproceedings{domke2018importance,
          title={Importance weighting and variational inference},
          author={Domke, Justin and Sheldon, Daniel R},
          booktitle={Advances in neural information processing systems},
          pages={4470--4479},
          year={2018}
        }

    The idea is based on the observation that an estimator of the evidence lower bound (ELBO)
    can be obtained from an importance weight :math:`w`:

    .. math:: L₁ = log w(x₁),    x₁ ~ q(x)

    ...where :math:`x` is the latent variable of the model (a GP, or set of GPs in our case)
    and the function :math:`w` is:

    .. math:: w(x) = p(y | x) p(x) / q(x)

    It follows that:

    .. math:: ELBO = 𝔼ₓ₁[ L₁ ]

    ...and:

    .. math:: log p(y) = log 𝔼ₓ₁[ w(x₁) ]

    It turns out that there are a series of lower bounds given by taking multiple importance
    samples:

    .. math:: Lₙ = log (1/n) Σᵢⁿ w(xᵢ),     xᵢ ~ q(x)

    And we have the relation:

    .. math:: log p(y) >= 𝔼[Lₙ] >= 𝔼[Lₙ₋₁] >= ... >= 𝔼[L₁] = ELBO

    This means that we can improve tightness of the ELBO to the log marginal likelihood by
    increasing :math:`n`, which we refer to in this class as `num_importance_samples`.
    The trade-offs are:

        * The objective function is now always stochastic, even for cases where the ELBO
          of the parent class is non-stochastic
        * We have to do more computations (evaluate the weights :math:`n` times)
    """

    def __init__(
        self,
        kernel: SDEKernel,
        inducing_points: tf.Tensor,
        likelihood: Likelihood,
        num_importance_samples: int,
        initial_distribution: Optional[StateSpaceModel] = None,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        :param kernel: A kernel that defines a prior over functions.
        :param inducing_points: The points in time on which inference should be performed,
            with shape ``batch_shape + [num_inducing]``.
        :param likelihood: A likelihood.
        :param num_importance_samples: The number of samples for the importance-weighted estimator.
        :param initial_distribution: An initial configuration for the variational distribution,
            with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        """
        super().__init__(kernel, likelihood, inducing_points, mean_function, initial_distribution)
        self._posterior = ImportanceWeightedPosteriorProcess(
            num_importance_samples,
            self._dist_q,
            self._kernel,
            inducing_points,
            self._likelihood,
            self._mean_function,
        )

    def elbo(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        r"""
        Compute the importance-weighted ELBO using K samples. The procedure is::

            for k=1...K:
                uₖ ~ q(u)
                sₖ ~ p(s | u)
                wₖ = p(y | sₖ)p(uₖ) / q(uₖ)

            ELBO = log (1/K) Σₖwₖ

        Everything is computed in log-space for stability. Note that gradients
        of this ELBO may have high variance with regard to the variational parameters;
        see the DREGS gradient estimator method.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :return: A scalar tensor.
        """
        num_importance_samples = self.posterior.num_importance_samples
        time_points, _ = input_data
        log_num_samples = tf.math.log(tf.cast(num_importance_samples, dtype=tf.float64))

        # draw samples from the proposeal distribution q(s, u) = q(u) p(s | u)
        sample_shape = (num_importance_samples,)
        samples_s, samples_u = self.posterior.proposal_process.sample_state_trajectories(
            time_points, sample_shape=sample_shape
        )
        log_weights = self.posterior.log_importance_weights(samples_s, samples_u, input_data)
        return tf.math.reduce_logsumexp(log_weights) - log_num_samples

    def dregs_objective(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Compute a scalar tensor that, when differentiated using `tf.gradients`,
        produces the DREGS variance controlled gradient.

        See `"Doubly Reparameterized Gradient Estimators For Monte Carlo Objectives"
        <https://openreview.net/pdf?id=HkG3e205K7>`_ for a derivation.

        We recommend using these gradients for training variational
        parameters and gradients of the importance-weighted ELBO for training hyperparameters.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:

            * A tensor of inputs with shape ``batch_shape + [num_data]``
            * A tensor of observations with shape ``batch_shape + [num_data, observation_dim]``
        :return: A scalar tensor.
        """
        num_importance_samples = self.posterior.num_importance_samples
        time_points, _ = input_data

        # draw samples from the proposal distribution q(s, u) = q(u) p(s | u)
        sample_shape = (num_importance_samples,)
        samples_s, samples_u = self.posterior.proposal_process.sample_state_trajectories(
            time_points, sample_shape=sample_shape
        )
        log_weights = self.posterior.log_importance_weights(
            samples_s, samples_u, input_data, stop_gradient=True
        )
        normalized_weights = tf.stop_gradient(tf.nn.softmax(log_weights))
        return tf.reduce_sum(tf.square(normalized_weights) * log_weights)
