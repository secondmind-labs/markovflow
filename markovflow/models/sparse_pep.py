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
"""Module containing a model for CVI"""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow import Parameter, default_float

from markovflow.base import ordered
from markovflow.conditionals import (
    _conditional_statistics,
    base_conditional_predict,
    conditional_statistics,
    pairwise_marginals,
)
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import SDEKernel
from markovflow.likelihoods import PEPGaussian, PEPScalarLikelihood
from markovflow.mean_function import MeanFunction, ZeroMeanFunction
from markovflow.models.models import MarkovFlowSparseModel
from markovflow.models.pep import gradient_correction
from markovflow.models.variational_cvi import back_project_nats
from markovflow.posterior import ConditionalProcess, PosteriorProcess
from markovflow.ssm_gaussian_transformations import naturals_to_ssm_params
from markovflow.state_space_model import StateSpaceModel


class SparsePowerExpectationPropagation(MarkovFlowSparseModel):
    """
    This is the  Sparse Power Expectation Propagation Algorithm

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

        t(v) = exp(𝜽ᵀφ(v) - A(𝜽)), where 𝜽=[θ₁, θ₂] and 𝛗(u)=[v, vᵀv]

    with 𝛗(v) are the sufficient statistics and 𝜽 the natural parameters
    """

    def __init__(
        self,
        kernel: SDEKernel,
        inducing_points: tf.Tensor,
        likelihood: PEPScalarLikelihood,
        mean_function: Optional[MeanFunction] = None,
        learning_rate=1.0,
        alpha=1.0,
    ) -> None:
        """
        :param kernel: A kernel that defines a prior over functions.
        :param inducing_points: The points in time on which inference should be performed,
            with shape ``batch_shape + [num_inducing]``.
        :param likelihood: A likelihood.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param learning_rate: the learning rate
        :param alpha: power as in Power Expectation Propagation
        """
        super().__init__(self.__class__.__name__)

        self._kernel = kernel
        if mean_function is None:
            mean_function = ZeroMeanFunction(obs_dim=1)
        self._mean_function = mean_function

        self.learning_rate = learning_rate
        self.alpha = alpha

        self.likelihood = likelihood
        self.inducing_inputs = Parameter(inducing_points, transform=ordered(), trainable=False)

        # initialize sites
        num_inducing = inducing_points.shape[0]
        state_dim = self._kernel.state_dim
        eye = tf.eye(2 * state_dim, dtype=default_float())
        zeros1 = tf.zeros((num_inducing + 1, 2 * state_dim), dtype=default_float())
        eye2 = tf.tile(eye[None], (num_inducing + 1, 1, 1)) * -1e-10

        self.log_norm = Parameter(tf.zeros((num_inducing + 1, 1), dtype=default_float()))
        self.nat1 = Parameter(zeros1)
        self.nat2 = Parameter(eye2)

    def posterior(self):
        """ Posterior Process """
        return ConditionalProcess(
            posterior_dist=self.dist_q,
            kernel=self._kernel,
            conditioning_time_points=self.inducing_inputs,
        )

    def mask_indices(self, exclude_indices):
        """
        Binary mask to exclude data indices
        :param exclude_indices:
        """
        if exclude_indices is None:
            return tf.zeros_like(self._time_points)
        else:
            updates = tf.ones(exclude_indices.shape[0], dtype=default_float())
            shape = self._observations.shape[:1]
            return tf.scatter_nd(exclude_indices, updates, shape)

    def back_project_nats(self, nat1, nat2, time_points):
        """
        back project natural gradient associated to time points to their associated
        inducing sites.
        """
        H = self._kernel.generate_emission_model(time_points=time_points).emission_matrix
        P, _ = conditional_statistics(time_points, self.inducing_inputs, self._kernel)
        HP = tf.matmul(H, P)
        return back_project_nats(nat1, nat2, HP)

    def local_objective(self, Fmu, Fvar, Y):
        """ Local objective of the PEP algorithm : log E_q(f) p(y|f)ᵃ """
        return self._likelihood.predict_log_density(Fmu, Fvar, Y, alpha=self.alpha)

    def local_objective_gradients(self, fx_mus, fx_covs, observations, alpha=1.0):
        """ Gradients of the local objective of the PEP algorithm wrt to the predictive mean """
        obj, grads = self.likelihood.grad_log_expected_density(
            fx_mus, fx_covs, observations, alpha=alpha
        )
        grads_expectation_param = gradient_correction([fx_mus, fx_covs], grads)
        return obj, grads_expectation_param

    def fraction_sites(self, time_points):
        """
        for all segment indexed m of consecutive inducing points [z_m, z_m+1[,
        this counts the time points t falling in that segment:
        c(m) = #{t, z_m <= t < z_m+1} and returns 1/c(m) or 0 when c(m)=0

        :param time_points: tensor of shape batch_shape + [num_data]
        :return: tensor of shape batch_shape + [num_data]
        """
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        num_partition = self.inducing_inputs.shape[0] + 1
        fraction = tf.stack(
            [
                tf.math.reciprocal_no_nan(tf.cast(tf.reduce_sum(l, axis=0), default_float()))
                for l in tf.dynamic_partition(
                    tf.ones_like(indices), indices, num_partitions=num_partition
                )
            ]
        )
        return tf.cast(fraction, default_float())

    def compute_posterior_ssm(self, nat1, nat2):
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

        summed_lik_nat1_diag = nat1[..., 1:, :sd] + nat1[..., :-1, sd:]

        summed_lik_nat2_diag = nat2[..., 1:, :sd, :sd] + nat2[..., :-1, sd:, sd:]
        summed_lik_nat2_subdiag = nat2[..., 1:-1, sd:, :sd]

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

    @property
    def dist_q(self):
        """
        Computes the variational posterior distribution on the vector of inducing states
        """
        return self.compute_posterior_ssm(self.nat1, self.nat2)

    def compute_marginals(self):
        """
        Compute pairwise marginals
        """
        batch_shape = ()
        return pairwise_marginals(
            self.dist_q,
            initial_mean=self._kernel.initial_mean(batch_shape),
            initial_covariance=self._kernel.initial_covariance(tf.constant([0.0])),
        )

    def remove_cavity_from_marginals(self, time_points, marginals):
        """
        Remove cavity from marginals
        :param time_points:
        :param marginals: pairwise mean and covariance tensors
        """
        pw_means, pw_covs = marginals
        pw_chol_covs = tf.linalg.cholesky(pw_covs)

        I = tf.eye(2 * self.kernel.state_dim, dtype=default_float())

        pw_nat2 = -0.5 * tf.linalg.cholesky_solve(pw_chol_covs, I)
        pw_nat1 = tf.linalg.cholesky_solve(pw_chol_covs, pw_means[..., None])[..., 0]

        indices = tf.searchsorted(self.inducing_inputs, time_points)
        batch_dims = len(time_points.shape[:-1])
        pairwise_nat2 = tf.gather(pw_nat2, indices, batch_dims=batch_dims)
        pairwise_nat1 = tf.gather(pw_nat1, indices, batch_dims=batch_dims)

        # site fraction
        fraction = self.fraction_sites(time_points)
        fractions = tf.gather(fraction, indices, batch_dims=batch_dims)
        frac_nat1 = (
            tf.gather(self.nat1, indices, batch_dims=batch_dims, axis=0) * fractions[..., None]
        )
        frac_nat2 = (
            tf.gather(self.nat2, indices, batch_dims=batch_dims) * fractions[..., None, None]
        )
        cav_nat2 = pairwise_nat2 - frac_nat2 * self.alpha
        cav_nat1 = pairwise_nat1 - frac_nat1 * self.alpha

        # TODO chol before tiling?
        cav_chol_nat2 = tf.linalg.cholesky(-cav_nat2)
        cav_means = 0.5 * tf.linalg.cholesky_solve(cav_chol_nat2, cav_nat1[..., None])[..., 0]
        cav_covs = 0.5 * tf.linalg.cholesky_solve(cav_chol_nat2, I)

        P, T, indices = _conditional_statistics(time_points, self.inducing_inputs, self.kernel)
        # projection and marginalization (if Sₜ given)
        sx_mus, sx_covs = base_conditional_predict(
            P, T, cav_means, pairwise_state_covariances=cav_covs
        )

        return sx_mus, sx_covs

    def compute_cavity_state(self, time_points):
        """
        The cavity distributions for data points at input time_points.
        This corresponds to the marginal distribution qᐠⁿ(fₙ) of qᐠⁿ(s) = q(s)/tₘ(vₘ)ᵝᵃ,
        where β = a * (1 / #time points `touching` site tₘ)
        """
        marginals = self.compute_marginals()
        return self.remove_cavity_from_marginals(time_points, marginals)

    def compute_cavity(self, time_points):
        """
        Cavity on f
        :param time_points: time points
        """
        sx_mus, sx_covs = self.compute_cavity_state(time_points)
        emission_model = self.kernel.generate_emission_model(time_points)
        fx_mus, fx_covs = emission_model.project_state_marginals_to_f(
            sx_mus, sx_covs, full_output_cov=False
        )
        return fx_mus, fx_covs

    def compute_new_sites(self, input_data):
        """
        Compute the site updates and perform one update step.
        :param input_data: A tuple of time points and observations containing the data from which
            to calculate the the updates:
            a tensor of inputs with shape ``batch_shape + [num_data]``,
            a tensor of observations with shape ``batch_shape + [num_data, observation_dim]``.
        """

        time_points, observations = input_data
        emission_model = self.kernel.generate_emission_model(time_points)

        # build the 2d x 2d covariance on inducing
        s_marg_mus, s_marg_covs = self.compute_marginals()
        sx_mus, sx_covs = self.remove_cavity_from_marginals(time_points, (s_marg_mus, s_marg_covs))

        #        sx_mus, sx_covs = self.compute_cavity_state(time_points)
        fx_mus, fx_covs = emission_model.project_state_marginals_to_f(
            sx_mus, sx_covs, full_output_cov=False
        )

        # get gradient of variational expectations wrt mu, sigma
        _, grads = self.local_objective_gradients(fx_mus, fx_covs, observations)

        theta_linear, lik_nat2 = self.back_project_nats(grads[0], grads[1], time_points)

        # sum sites together
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        num_partition = self.inducing_inputs.shape[0] + 1

        batch_dims = len(time_points.shape[:-1])
        fraction = self.fraction_sites(time_points)
        fractions = tf.gather(fraction, indices, batch_dims=batch_dims)

        summed_theta_linear = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(
                    theta_linear + 0 * fractions[..., None], indices, num_partitions=num_partition
                )
            ]
        )
        summed_lik_nat2 = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(
                    lik_nat2 + 0 * fractions[..., None, None], indices, num_partitions=num_partition
                )
            ]
        )

        # update
        a = self.alpha
        lr = self.learning_rate

        unchanged_nat1 = self.nat1
        unchanged_nat2 = self.nat2

        pep_nat1 = unchanged_nat1 * (1 - a) + summed_theta_linear * a
        pep_nat2 = unchanged_nat2 * (1 - a) + summed_lik_nat2 * a

        changed_nat1 = unchanged_nat1 * (1 - lr) + pep_nat1 * lr
        changed_nat2 = unchanged_nat2 * (1 - lr) + pep_nat2 * lr

        return changed_nat1, changed_nat2

    def compute_log_norm(self, input_data):
        """
        Compute the site updates and perform one update step.
        :param input_data: A tuple of time points and observations containing the data from which
            to calculate the the updates:
            a tensor of inputs with shape ``batch_shape + [num_data]``,
            a tensor of observations with shape ``batch_shape + [num_data, observation_dim]``.
        """

        time_points, observations = input_data
        emission_model = self.kernel.generate_emission_model(time_points)

        # build the 2d x 2d covariance on inducing
        s_marg_mus, s_marg_covs = self.compute_marginals()
        sx_mus, sx_covs = self.remove_cavity_from_marginals(time_points, (s_marg_mus, s_marg_covs))

        fx_mus, fx_covs = emission_model.project_state_marginals_to_f(
            sx_mus, sx_covs, full_output_cov=False
        )

        # get gradient of variational expectations wrt mu, sigma
        obj, _ = self.local_objective_gradients(fx_mus, fx_covs, observations, alpha=self.alpha)

        # sum sites together
        num_partition = self.inducing_inputs.shape[0] + 1

        # compute the site normalizer:
        # z = obj * G(cav) - G(cav + new)

        # normalizer of the distribution with all sites included
        log_norm_marg = self.dist_q.normalizer()
        # compute total neighbours
        neighbours = self.compute_num_data_per_interval(time_points)
        # compute sites for all intervals with 1 neighbour out
        fraction_one_neighbour = tf.math.reciprocal_no_nan(neighbours)
        # compute nats with one out for all
        nat1 = tf.tile(self.nat1[None, :], [num_partition, 1, 1])
        nat2 = tf.tile(self.nat2[None, :], [num_partition, 1, 1, 1])
        diag = tf.linalg.diag(fraction_one_neighbour * self.alpha)
        nat1 = nat1 * (1 - diag[..., None])
        nat2 = nat2 * (1 - diag[..., None, None])
        # compute marginals with one out
        log_norm_cav = tf.stack(
            [
                self.compute_posterior_ssm(nat1[m], nat2[m]).normalizer()
                for m in range(num_partition)
            ]
        )
        # dispatch to each data point
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        batch_dims = len(time_points.shape[:-1])
        log_norm_cav = tf.gather(log_norm_cav, indices, batch_dims=batch_dims)

        # normalizer
        log_norm = obj + (tf.squeeze(log_norm_cav) - log_norm_marg)

        # sum back to inducing point
        summed_log_z = tf.stack(
            [
                tf.reduce_sum(l, axis=0)
                for l in tf.dynamic_partition(
                    (log_norm)[..., None], indices, num_partitions=num_partition
                )
            ]
        )

        return summed_log_z / self.alpha

    def compute_num_data_per_interval(self, time_points):
        """ compute fraction of site per data point """
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        num_partition = self.inducing_inputs.shape[0] + 1
        neighbours = tf.stack(
            [
                tf.cast(tf.reduce_sum(l, axis=0), default_float())
                for l in tf.dynamic_partition(
                    tf.ones_like(indices), indices, num_partitions=num_partition
                )
            ]
        )
        return neighbours

    def compute_fraction(self, time_points):
        """ compute fraction of site per data point """
        # sum sites together
        indices = tf.searchsorted(self.inducing_inputs, time_points)
        batch_dims = len(time_points.shape[:-1])
        fraction = self.fraction_sites(time_points)
        fractions = tf.gather(fraction, indices, batch_dims=batch_dims)
        return fractions

    def update_sites(self, input_data):
        """ apply updates """

        changed_nat1, changed_nat2 = self.compute_new_sites(input_data)

        self.nat1.assign(changed_nat1)
        self.nat2.assign(changed_nat2)

        a = self.alpha
        lr = self.learning_rate
        log_norm = self.compute_log_norm(input_data)
        pep_log_norm = self.log_norm * (1 - a) + log_norm * a
        changed_log_norm = self.log_norm * (1 - lr) + pep_log_norm * lr

        self.log_norm.assign(changed_log_norm)

    def energy(self, input_data):
        """
        The PEP energy : ∫ ds p(s) 𝚷_m t_m(v_m)
        :param input_data: input data
        """
        log_norm = self.compute_log_norm(input_data)
        return self.dist_q.normalizer() - self.dist_p.normalizer() + tf.reduce_sum(log_norm)

    def loss(self, input_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Return the loss, which is the negative evidence lower bound (ELBO).

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model.
        """
        return -self.elbo(input_data)

    @property
    def dist_p(self) -> GaussMarkovDistribution:
        """
        Return the prior `GaussMarkovDistribution`.
        """
        return self._kernel.build_finite_distribution(self.inducing_inputs)

    @property
    def kernel(self) -> SDEKernel:
        """
        Return the kernel of the GP.
        """
        return self._kernel

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
        fx_mus, fx_covs = self.posterior().predict_f(time_points)
        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self.likelihood.variational_expectations(fx_mus, fx_covs, observations)
        )
        # KL[q(sₓ) || p(sₓ)]

        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))
        # Return ELBO(fₓ) = VE(fₓ) - KL[q(sₓ) || p(sₓ)]
        return ve_fx - kl_fx

    def predict_log_density(
        self, input_data: Tuple[tf.Tensor, tf.Tensor], full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.

        :param input_data: A tuple of time points and observations containing the data at which
            to calculate the loss for training the model:
            a tensor of inputs with shape ``batch_shape + [num_data]``,
            a tensor of observations with shape ``batch_shape + [num_data, observation_dim]``.
        :param full_output_cov: Either full output covariance (`True`) or marginal
            variances (`False`).
        """
        X, Y = input_data
        f_mean, f_var = self.posterior().predict_f(X)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)
