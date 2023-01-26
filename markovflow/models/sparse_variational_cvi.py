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
"""Module containing a model for Sparse CVI"""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow import default_float
from gpflow.base import Parameter
from gpflow.likelihoods import Likelihood

from markovflow.base import ordered
from markovflow.conditionals import conditional_statistics
from markovflow.kernels import SDEKernel
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowSparseModel
from markovflow.models.variational_cvi import (
    back_project_nats,
    gradient_transformation_mean_var_to_expectation,
)
from markovflow.posterior import ConditionalProcess
from markovflow.ssm_gaussian_transformations import naturals_to_ssm_params
from markovflow.state_space_model import StateSpaceModel


class SparseCVIGaussianProcess(MarkovFlowSparseModel):
    """
    This is an alternative parameterization to the `SparseVariationalGaussianProcess`

    Approximates a the posterior of a model with GP prior and a general likelihood
    using a Gaussian posterior parameterized with Gaussian sites on
    inducing states u at inducing points z.

    The following notation is used:

        * x - the time points of the training data.
        * z - the time points of the inducing/pseudo points.
        * y - observations corresponding to time points x.
        * s(.) - the continuous time latent state process
        * u = s(z) - the discrete inducing latent state space model
        * f(.) - the noise free predictions of the model
        * p(y | f) - the likelihood
        * t(u) - a site (indices will refer to the associated data point)
        * p(.) the prior distribution
        * q(.) the variational distribution

    We use the state space formulation of Markovian Gaussian Processes that specifies:
    the conditional density of neighbouring latent states: p(sₖ₊₁| sₖ)
    how to read out the latent process from these states: fₖ = H sₖ

    The likelihood links data to the latent process and p(yₖ | fₖ).
    We would like to approximate the posterior over the latent state space model of this model.

    To approximate the posterior, we maximise the evidence lower bound (ELBO) (ℒ) with
    respect to the parameters of the variational distribution, since::

        log p(y) = ℒ(q) + KL[q(s) ‖ p(s | y)]

    ...where::

        ℒ(q) = ∫ log(p(s, y) / q(s)) q(s) ds

    We parameterize the variational posterior through M sites tₘ(vₘ)

        q(s) = p(s) ∏ₘ  tₘ(vₘ)

    where tₘ(vₘ) are multivariate Gaussian sites on vₘ = [uₘ, uₘ₊₁],
    i.e. consecutive inducing states.

    The sites are parameterized in the natural form

        t(v) = exp(𝜽ᵀφ(v) - A(𝜽)), where 𝜽=[θ₁, θ₂] and 𝛗(u)=[Wv, WᵀvᵀvW]

    with 𝛗(v) are the sufficient statistics and 𝜽 the natural parameters
    and W is the projection of the conditional mean E_p(f|v)[f] = W v

    Each data point indexed k contributes a fraction of the site it belongs to.
    If vₘ = [uₘ, uₘ₊₁], and zₘ < xₖ <= zₘ₊₁, then xₖ `belongs` to vₘ.

    The natural gradient update of the sites are similar to that of the
    CVIGaussianProcess except that they apply to a different parameterization of
    the sites
    """

    def __init__(
        self,
        kernel: SDEKernel,
        inducing_points: tf.Tensor,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        learning_rate=0.1,
    ) -> None:
        """
        :param kernel: A kernel that defines a prior over functions.
        :param inducing_points: The points in time on which inference should be performed,
            with shape ``batch_shape + [num_inducing]``.
        :param likelihood: A likelihood.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param learning_rate: the learning rate.
        """

        super().__init__(self.__class__.__name__)

        self._kernel = kernel
        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        self._likelihood = likelihood

        self.learning_rate = learning_rate
        self.inducing_inputs = Parameter(inducing_points, transform=ordered(), trainable=False)

        # q will approximate the posterior after optimisation.
        # This needs to be an instance attribute to provide trainable variables
        # when calling gpflow.Module trainable_variables. This is fine though, since
        # StateSpaceModel doesn't do any computation in its initialiser.

        # initialize sites
        num_inducing = inducing_points.shape[0]
        state_dim = self.kernel.state_dim
        zeros1 = tf.zeros((num_inducing + 1, 2 * state_dim), dtype=default_float())
        zeros2 = tf.zeros((num_inducing + 1, 2 * state_dim, 2 * state_dim), dtype=default_float())
        self.nat1 = Parameter(zeros1)
        self.nat2 = Parameter(zeros2)

    @property
    def dist_q(self):
        """
        Computes the variational posterior distribution on the vector of inducing states
        """
        # get prior precision
        prec = self.dist_p.precision

        # [..., num_transitions + 1, state_dim, state_dim]
        prec_diag = prec.block_diagonal
        # [..., num_transitions, state_dim, state_dim]
        prec_subdiag = prec.block_sub_diagonal

        sd = self.kernel.state_dim

        summed_lik_nat1_diag = self.nat1[..., 1:, :sd] + self.nat1[..., :-1, sd:]

        summed_lik_nat2_diag = self.nat2[..., 1:, :sd, :sd] + self.nat2[..., :-1, sd:, sd:]
        summed_lik_nat2_subdiag = self.nat2[..., 1:-1, sd:, :sd]

        # conjugate update of the natural parameter: post_nat = prior_nat + lik_nat
        theta_diag = -0.5 * prec_diag + summed_lik_nat2_diag
        theta_subdiag = -prec_subdiag + summed_lik_nat2_subdiag * 2.0

        post_ssm_params = naturals_to_ssm_params(
            theta_linear=summed_lik_nat1_diag, theta_diag=theta_diag, theta_subdiag=theta_subdiag
        )

        post_ssm = StateSpaceModel(
            state_transitions=post_ssm_params[0],
            state_offsets=post_ssm_params[1],
            chol_initial_covariance=post_ssm_params[2],
            chol_process_covariances=post_ssm_params[3],
            initial_mean=post_ssm_params[4],
        )
        return post_ssm

    def update_sites(self, input_data: Tuple[tf.Tensor, tf.Tensor]):
        """
        Perform one joint update of the Gaussian sites
                𝜽ₘ ← ρ𝜽ₘ + (1-ρ)𝐠ₘ

        Here 𝐠ₘ are the sum of the gradient of the variational expectation for each data point
        indexed k, projected back to the site vₘ, through the conditional p(fₖ|vₘ)
        :param input_data: A tuple of time points and observations
        """
        time_points, observations = input_data

        fx_mus, fx_covs = self.posterior.predict_f(time_points)

        # get gradient of variational expectations wrt mu, sigma
        _, grads = self.local_objective_and_gradients(fx_mus, fx_covs, observations)

        H = self.kernel.generate_emission_model(time_points=time_points).emission_matrix
        P, _ = conditional_statistics(time_points, self.inducing_inputs, self.kernel)
        HP = tf.matmul(H, P)

        theta_linear, lik_nat2 = back_project_nats(grads[0], grads[1], HP)

        # sum sites together
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        num_partition = self.inducing_inputs.shape[0] + 1

        summed_theta_linear = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(theta_linear, indices, num_partitions=num_partition)
            ]
        )
        summed_lik_nat2 = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(lik_nat2, indices, num_partitions=num_partition)
            ]
        )

        # update
        lr = self.learning_rate
        new_nat1 = (1 - lr) * self.nat1 + lr * summed_theta_linear
        new_nat2 = (1 - lr) * self.nat2 + lr * summed_lik_nat2

        self.nat2.assign(new_nat2)
        self.nat1.assign(new_nat1)

    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Obtain a `Tensor` representing the loss, which can be used to train the model.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model.
        """
        return -self.classic_elbo(input_data)

    @property
    def posterior(self):
        """ Posterior object to predict outside of the training time points """
        return ConditionalProcess(
            posterior_dist=self.dist_q,
            kernel=self.kernel,
            conditioning_time_points=self.inducing_inputs,
        )

    def local_objective_and_gradients(self, Fmu, Fvar, Y):
        """
        Returs the local_objective and its gradients wrt to the expectation parameters
        :param Fmu: means μ [..., latent_dim]
        :param Fvar: variances σ² [..., latent_dim]
        :param Y: observations Y [..., observation_dim]
        :return: local objective and gradient wrt [μ, σ² + μ²]
        """

        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(input_tensor=self.local_objective(Fmu, Fvar, Y))
        grads = g.gradient(local_obj, [Fmu, Fvar])

        # turn into gradient wrt μ, σ² + μ²
        grads = gradient_transformation_mean_var_to_expectation([Fmu, Fvar], grads)

        return local_obj, grads

    def local_objective(self, Fmu, Fvar, Y):
        """
        local loss in CVI
        :param Fmu: means [..., latent_dim]
        :param Fvar: variances [..., latent_dim]
        :param Y: observations [..., observation_dim]
        :return: local objective [...]
        """
        return self._likelihood.variational_expectations(Fmu, Fvar, Y)

    def classic_elbo(self, input_data: Tuple[tf.Tensor, tf.Tensor]):
        """
        Computes the ELBO the classic way:
            ℒ(q) = Σᵢ ∫ log(p(yᵢ | f)) q(f) df - KL[q(f) ‖ p(f)]

        Note: this is mostly for testing purposes and not to be used for optimization

        :param input_data: A tuple of time points and observations
        :return: A scalar tensor representing the ELBO.
        """
        time_points, observations = input_data
        # s ~ q(s) = N(μ, P)
        # Project to function space, fₓ = H*s ~ q(fₓ)
        fx_mus, fx_covs = self.posterior.predict_f(time_points)
        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(fx_mus, fx_covs, observations)
        )
        # KL[q(sₓ) || p(sₓ)]

        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))
        # Return ELBO(fₓ) = VE(fₓ) - KL[q(sₓ) || p(sₓ)]
        return ve_fx - kl_fx

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel of the GP.
        """
        return self._kernel

    @property
    def dist_p(self) -> StateSpaceModel:
        """
        Return the prior `GaussMarkovDistribution`.
        """
        return self._kernel.state_space_model(self.inducing_inputs)

    @property
    def likelihood(self) -> Likelihood:
        """
        Return the likelihood of the GP.
        """
        return self._likelihood
