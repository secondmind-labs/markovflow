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
from gpflow import default_float

from markovflow.kernels import SDEKernel
from markovflow.likelihoods import PEPScalarLikelihood
from markovflow.mean_function import MeanFunction
from markovflow.models.variational_cvi import GaussianProcessWithSitesBase, back_project_nats


class PowerExpectationPropagation(GaussianProcessWithSitesBase):
    """
    This is an approximate inference called Power Expectation Propagation.

    Approximates a the posterior of a model with GP prior and a general likelihood
    using a Gaussian posterior parameterized with Gaussian sites.

    The following notation is used:

        * x - the time points of the training data.
        * y - observations corresponding to time points x.
        * s(.) - the latent state of the Markov chain
        * f(.) - the noise free predictions of the model
        * p(y | f) - the likelihood
        * t(f) - a site (indices will refer to the associated data point)
        * p(.) the prior distribution
        * q(.) the variational distribution

    We use the state space formulation of Markovian Gaussian Processes that specifies:
    the conditional density of neighbouring latent states: p(xₖ₊₁| xₖ)
    how to read out the latent process from these states: fₖ = H xₖ

    The likelihood links data to the latent process and p(yₖ | fₖ).
    We would like to approximate the posterior over the latent state space model of this model.

    We parameterize the joint posterior using sites tₖ(fₖ)

        p(x, y) = p(x) ∏ₖ tₖ(fₖ)

    where tₖ(fₖ) are univariate Gaussian sites parameterized in the natural form

        t(f) = exp(𝞰ᵀφ(f) - A(𝞰)), where 𝞰=[η₁,η₂] and 𝛗(f)=[f,f²]

    (note: the subscript k has been omitted for simplicity)

    The site update of the sites are given by the classic EP update rules as described in:

    @techreport{seeger2005expectation,
      title={Expectation propagation for exponential families},
      author={Seeger, Matthias},
      year={2005}
    }

    """

    def __init__(
        self,
        kernel: SDEKernel,
        input_data: Tuple[tf.Tensor, tf.Tensor],
        likelihood: PEPScalarLikelihood,
        mean_function: Optional[MeanFunction] = None,
        learning_rate=1.0,
        alpha=1.0,
    ) -> None:
        """
        :param kernel: A kernel that defines a prior over functions.
        :param input_data: A tuple of ``(time_points, observations)`` containing the observed data:
            time points of observations, with shape ``batch_shape + [num_data]``,
            observations with shape ``batch_shape + [num_data, observation_dim]``.
        :param likelihood: A likelihood.
            with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param learning_rate: the learning rate of the algorithm
        :param alpha: the power as in Power Expectation propagation
        """
        super().__init__(
            input_data=input_data, kernel=kernel, likelihood=likelihood, mean_function=mean_function
        )
        self.learning_rate = learning_rate
        self.alpha = alpha

    def local_objective(self, Fmu, Fvar, Y):
        """ Local objective of the PEP algorithm : log E_q(f) p(y|f)ᵃ """
        return self._likelihood.log_expected_density(Fmu, Fvar, Y, alpha=self.alpha)

    def local_objective_gradients(self, Fmu, Fvar):
        """ Gradients of the local objective of the PEP algorithm wrt to the predictive mean """
        obj, grads = self.likelihood.grad_log_expected_density(
            Fmu, Fvar, self._observations, alpha=self.alpha
        )
        grads_expectation_param = gradient_correction([Fmu, Fvar], grads)
        return obj, grads_expectation_param

    def mask_indices(self, exclude_indices):
        """ Binary mask (cast to float), 0 for the excluded indices, 1 for the rest """
        if exclude_indices is None:
            return tf.zeros_like(self.time_points)
        else:
            updates = tf.ones(exclude_indices.shape[0], dtype=default_float())
            shape = self.observations.shape[:1]
            return tf.scatter_nd(exclude_indices, updates, shape)

    def compute_cavity_from_marginals(self, marginals):
        """
        Compute cavity from marginals
        :param marginals: list of tensors
        """
        means, covs = marginals
        # compute the natural form of the posterior marginals
        chol_covs = tf.linalg.cholesky(covs)
        I = tf.eye(self.kernel.state_dim, dtype=default_float())
        nat2 = -0.5 * tf.linalg.cholesky_solve(chol_covs, I)
        nat1 = tf.linalg.cholesky_solve(chol_covs, means[..., None])[..., 0]

        # remove fraction of sites from the posterior to get natural gradients of cavity
        H = self.kernel.generate_emission_model(self.time_points).emission_matrix
        bp_nat1, bp_nat2 = back_project_nats(self.sites.nat1, self.sites.nat2[..., 0], H)
        cav_nat2 = nat2 - bp_nat2 * self.alpha
        cav_nat1 = nat1 - bp_nat1 * self.alpha

        # get cavity stats in mean / cov form
        cav_chol_nat2 = tf.linalg.cholesky(-cav_nat2)
        cav_means = 0.5 * tf.linalg.cholesky_solve(cav_chol_nat2, cav_nat1[..., None])[..., 0]
        cav_covs = 0.5 * tf.linalg.cholesky_solve(cav_chol_nat2, I)

        # project to observation f
        emission_model = self.kernel.generate_emission_model(self.time_points)
        fx_mus, fx_covs = emission_model.project_state_marginals_to_f(
            cav_means, cav_covs, full_output_cov=False
        )
        return fx_mus, fx_covs

    def compute_cavity(self):
        """
        The cavity distributions for all data points.
        This corresponds to the marginal distribution qᐠⁿ(fₙ) of qᐠⁿ(f) = q(f)/tₙ(fₙ)ᵃ
        """
        # compute the posterior on data points (including all sites)
        marginals = self.dist_q.marginals
        return self.compute_cavity_from_marginals(marginals)

    def compute_log_norm(self):
        """
        Compute log normalizer
        """
        marginals = self.dist_q.marginals
        emission_model = self.kernel.generate_emission_model(self.time_points)
        fx_marg_mus, fx_marg_covs = emission_model.project_state_marginals_to_f(
            *marginals, full_output_cov=False
        )

        fx_mus, fx_covs = self.compute_cavity_from_marginals(marginals)
        # get gradient of variational expectations wrt mu, sigma
        obj, _ = self.local_objective_gradients(fx_marg_mus, fx_marg_covs)

        # log normalizer of cavity
        log_norm_cav = 0.5 * (tf.math.log(fx_covs) + fx_mus ** 2 / fx_covs)
        log_norm_marg = 0.5 * (tf.math.log(fx_marg_covs) + fx_marg_mus ** 2 / fx_marg_covs)
        # normalizer
        return obj + tf.squeeze(log_norm_cav) - tf.squeeze(log_norm_marg)

    def update_sites(self, site_indices=None):
        """
        Compute the site updates and perform one update step
        :param site_indices: list of indices to be updated
        """
        marginals = self.dist_q.marginals
        emission_model = self.kernel.generate_emission_model(self.time_points)
        fx_marg_mus, fx_marg_covs = emission_model.project_state_marginals_to_f(
            *marginals, full_output_cov=False
        )

        fx_mus, fx_covs = self.compute_cavity_from_marginals(marginals)
        # get gradient of variational expectations wrt mu, sigma
        obj, grads = self.local_objective_gradients(fx_mus, fx_covs)

        # log normalizer of cavity
        log_norm_cav = 0.5 * (tf.math.log(fx_covs) + fx_mus ** 2 / fx_covs)
        log_norm_marg = 0.5 * (tf.math.log(fx_marg_covs) + fx_marg_mus ** 2 / fx_marg_covs)
        # normalizer
        log_norm = obj + tf.squeeze(log_norm_cav) - tf.squeeze(log_norm_marg)

        # PEP update
        a = self.alpha
        pep_nat1 = (1 - a) * self.sites.nat1 + grads[0]
        pep_nat2 = ((1 - a) * self.sites.nat2[..., 0] + grads[1])[..., None]
        pep_log_norm = (1 - a) * self.sites.log_norm + log_norm[..., None]
        # Additional damping

        lr = self.learning_rate
        new_nat1 = (1 - lr) * self.sites.nat1 + lr * pep_nat1
        new_nat2 = (1 - lr) * self.sites.nat2 + lr * pep_nat2
        new_log_norm = (1 - lr) * self.sites.log_norm + lr * pep_log_norm

        mask = self.mask_indices(exclude_indices=site_indices)[..., None]
        self.sites.nat2.assign(self.sites.nat2 * (1 - mask)[..., None] + new_nat2 * mask[..., None])
        self.sites.nat1.assign(self.sites.nat1 * (1 - mask) + new_nat1 * mask)
        self.sites.log_norm.assign(self.sites.log_norm * (1 - mask) + new_log_norm * mask)

    def elbo(self) -> tf.Tensor:
        """
        Computes the marginal log marginal likelihood of the approximate  joint p(s, y)
        """
        return self.posterior_kalman.log_likelihood()

    def energy(self):
        """ PEP Energy """
        log_norm = self.compute_log_norm()
        return (
            self.dist_q.normalizer()
            - self.dist_p.normalizer()
            + 1.0 / self.alpha * tf.reduce_sum(log_norm)
        )

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
        f_mean, f_var = self.posterior.predict_f(X, full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)


def gradient_correction(inputs, grads):
    """
    Transforms vectors g=[g1,g2] and i=[i1,i2] into h=[h1, h2]
    where h2 = 1/2 * 1/(i2 + 1/g2) and h1 = 2 * h2 * (g1/g2 - i1)

    :param inputs: a tensor of inputs with shape ``batch_shape + [num_data]``,
    :param grads: a tensor of gradients with shape ``batch_shape + [num_data]``,
    :return: a tensor of modified gradients with shape ``batch_shape + [num_data]``,
    """
    L2 = 0.5 * (inputs[1] + 1.0 / grads[1]) ** -1
    L1 = 2 * L2 * (1.0 * grads[0] / grads[1] - inputs[0])
    return L1, L2
