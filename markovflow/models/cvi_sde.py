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
"""Module containing a model for CVI in SDE models"""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.base import Parameter
from gpflow import default_float

from markovflow.kalman_filter import UnivariateGaussianSitesNat
from markovflow.models.variational_cvi import CVIGaussianProcess
from markovflow.mean_function import MeanFunction
from markovflow.sde.sde import SDE, OrnsteinUhlenbeckSDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import linearize_sde
from markovflow.emission_model import EmissionModel
from markovflow.kalman_filter import KalmanFilterWithSites
from markovflow.sde.sde_utils import KL_sde


class SDESSM(CVIGaussianProcess):
    """
    """

    def __init__(
            self,
            prior_sde: SDE,
            grid: tf.Tensor,
            input_data: Tuple[tf.Tensor, tf.Tensor],
            likelihood: Likelihood,
            mean_function: Optional[MeanFunction] = None,
            learning_rate=0.1,
            prior_params_lr: float = 0.01
    ) -> None:
        """
        :param prior_sde: Prior SDE over the latent states, x.
        :param grid: Grid over time with shape ``batch_shape + [grid_size]``
        :param input_data: A tuple containing the observed data:

            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``

        :param likelihood: A likelihood with shape ``batch_shape + [num_inducing]``.
        :param mean_function: The mean function for the GP. Defaults to no mean function.
        :param learning_rate: The learning rate of the algorithm.
        """
        # TODO: check passing of kernel=None
        super().__init__(
            input_data=input_data, kernel=None, likelihood=likelihood, mean_function=mean_function,
            learning_rate=learning_rate, initialize_sites=False
        )

        self.grid = grid
        self.prior_sde = prior_sde

        # initialize data sites
        self.observations_time_points = self._time_points
        self.data_sites = UnivariateGaussianSitesNat(
            nat1=Parameter(tf.zeros_like(self.observations)),
            nat2=Parameter(tf.ones_like(self.observations)[..., None] * -1e-10),
            log_norm=Parameter(tf.zeros_like(self.observations)),
        )
        self.output_dim = 1
        self.state_dim = 1

        self._initialize_mean_statistic()
        self.sites_nat2 = tf.ones_like(self.grid, dtype=self.observations.dtype)[..., None, None] * -1e-20

        self.sites_lr = learning_rate
        self.prior_sde_optimizer = tf.optimizers.SGD(lr=prior_params_lr)
        self.elbo_vals = []

        self.dist_p_ssm = None
        self._linearize_prior()

        self.prior_params = {}
        for i, param in enumerate(self.prior_sde.trainable_variables):
            self.prior_params[i] = [param.numpy().item()]

    def _store_prior_param_vals(self):
        """Update the list storing the prior sde parameter values"""
        for i, param in enumerate(self.prior_sde.trainable_variables):
            self.prior_params[i].append(param.numpy().item())

    def _linearize_prior(self):
        """
            Set the :class:`~markovflow.state_space_model.StateSpaceModel` representation of the prior process.

            Here, we approximate (linearize) the prior SDE based on the grid.
        """
        # FIXME: check this. because of timepoints[:-1]?
        fx_mus = self.fx_mus[:, :-1, :]
        fx_covs = self.fx_covs[:, :-1, :, :]

        self.dist_p_ssm = linearize_sde(sde=self.prior_sde, transition_times=self.time_points, q_mean=fx_mus,
                                        q_covar=fx_covs, initial_mean=self.initial_mean,
                                        initial_chol_covariance=self.initial_chol_cov,
                                        )

    def _initialize_mean_statistic(self):
        """Simulate initial mean from the prior SDE."""

        self.initial_mean = tf.zeros((1, self.state_dim), dtype=self.observations.dtype)
        self.initial_chol_cov = tf.linalg.cholesky(tf.reshape(self.prior_sde.q, (1, self.state_dim, self.state_dim)))

        self.fx_mus = tf.zeros((1, self.grid.shape[0], self.state_dim), dtype=self.observations.dtype)
        self.fx_covs = tf.square(self.initial_chol_cov) * tf.ones_like(self.fx_mus[..., None])

    @property
    def sites(self) -> UnivariateGaussianSitesNat:
        """
        Sites over a finer grid where at the observation timepoints the parameters are replaced by that
        of the data sites.
        """
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0][..., None]

        nat1 = tf.scatter_nd(indices, self.data_sites.nat1, self.grid[..., None].shape)
        nat2 = tf.tensor_scatter_nd_update(self.sites_nat2, indices, self.data_sites.nat2)

        log_norm = tf.scatter_nd(indices, self.data_sites.log_norm, self.grid[..., None].shape)

        return UnivariateGaussianSitesNat(
            nat1=Parameter(nat1),
            nat2=Parameter(nat2),
            log_norm=Parameter(log_norm),
        )

    @property
    def time_points(self) -> tf.Tensor:
        """
        Return the time points of the observations. For SDE CVI, it is the time-grid.

        :return: A tensor with shape ``batch_shape + [grid_size]``.
        """
        return self.grid

    @property
    def dist_q(self) -> StateSpaceModel:
        """
        Construct the :class:`~markovflow.state_space_model.StateSpaceModel` representation of
        the posterior process indexed at the time points.
        """
        return self.posterior_kalman.posterior_state_space_model()

    def generate_emission_model(self, time_points: tf.Tensor) -> EmissionModel:
        """
        Generate the :class:`~markovflow.emission_model.EmissionModel` that maps from the
        latent :class:`~markovflow.state_space_model.StateSpaceModel` to the observations.

        :param time_points: The time points over which the emission model is defined, with shape
            ``batch_shape + [num_data]``.
        """
        # create 2D matrix
        emission_matrix = tf.concat(
            [
                tf.ones((self.output_dim, 1), dtype=default_float()),
                tf.zeros((self.output_dim, self.state_dim - 1), dtype=default_float()),
            ],
            axis=-1,
        )
        # tile for each time point
        # expand emission_matrix from [output_dim, state_dim], to [1, 1 ... output_dim, state_dim]
        # where there is a singleton dimension for the dimensions of time points
        batch_shape = time_points.shape[:-1]  # num_data may be undefined so skip last dim
        shape = tf.concat(
            [tf.ones(len(batch_shape) + 1, dtype=tf.int32), tf.shape(emission_matrix)], axis=0
        )
        emission_matrix = tf.reshape(emission_matrix, shape)

        # tile the emission matrix into shape batch_shape + [num_data, output_dim, state_dim]
        repetitions = tf.concat([tf.shape(time_points), [1, 1]], axis=0)
        return EmissionModel(tf.tile(emission_matrix, repetitions))

    @property
    def posterior_kalman(self) -> KalmanFilterWithSites:
        """Build the Kalman filter object from the prior state space models and the sites."""
        return KalmanFilterWithSites(
            state_space_model=self.dist_p_ssm,
            emission_model=self.generate_emission_model(tf.reshape(self.time_points, (-1))),
            sites=self.sites,
        )

    def update_sites(self, convergence_tol=1e-4) -> bool:
        """
        Perform one joint update of the Gaussian sites. That is:

        .. math:: 𝜽 ← ρ𝜽 + (1-ρ)𝐠

        Note: We update the data sites and not the full sites.
        """

        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0][..., None]
        fx_mus = tf.gather_nd(tf.reshape(self.fx_mus, (-1, 1)), indices)
        fx_covs = tf.gather_nd(tf.reshape(self.fx_covs, (-1, 1)), indices)

        # get gradient of variational expectations wrt the expectation parameters μ, σ² + μ²
        _, grads = self.local_objective_and_gradients(fx_mus, fx_covs)

        # update
        new_nat1 = (1 - self.sites_lr) * self.data_sites.nat1 + self.sites_lr * grads[0]
        new_nat2 = (1 - self.sites_lr) * self.data_sites.nat2 + self.sites_lr * grads[1][..., None]

        nat1_sq_norm = tf.reduce_sum(tf.square(self.data_sites.nat1 - new_nat1))
        nat2_sq_norm = tf.reduce_sum(tf.square(self.data_sites.nat2 - new_nat2))

        if (nat1_sq_norm < convergence_tol) & (nat2_sq_norm < convergence_tol):
            has_converged = True
        else:
            has_converged = False

        self.data_sites.nat2.assign(new_nat2)
        self.data_sites.nat1.assign(new_nat1)

        # Done this way as dist_q -> dist_p_ssm -> fx_mus, fx_covs
        dist_q = self.dist_q
        self.fx_mus, self.fx_covs = dist_q.marginals

        return has_converged

    def kl(self, dist_p: StateSpaceModel) -> tf.Tensor:
        r"""
        KL between dist_q and dist_p modified for stopping gradient.

        .. math::
            dist₁ = 𝓝(μ₁, P⁻¹₁)\\
            dist₂ = 𝓝(μ₂, P⁻¹₂)

        The KL divergence is thus given by:

        .. math::
            KL(dist₁ ∥ dist₂) = ½(tr(P₂P₁⁻¹) + (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁) - N - log(|P₂|) + log(|P₁|))

        """

        dist_q = self.dist_q

        batch_shape = dist_q.batch_shape

        marginal_covs_1 = dist_q.marginal_covariances
        precision_2 = dist_p.precision

        marginal_covs_1 = tf.stop_gradient(marginal_covs_1)

        # trace term, we use that for any trace tr(AᵀB) = Σᵢⱼ Aᵢⱼ Bᵢⱼ
        # and since the P₂ is symmetric block tri diagonal, we only need the block diagonal and
        # block sub diagonals from from P₁⁻¹
        # this is the sub diagonal of P₁⁻¹, [..., num_transitions, state_dim, state_dim]
        subsequent_covs_1 = dist_q.subsequent_covariances(marginal_covs_1)
        subsequent_covs_1 = tf.stop_gradient(subsequent_covs_1)

        # trace_sub_diag must be added twice as the matrix is symmetric, [...]
        trace = tf.reduce_sum(
            input_tensor=precision_2.block_diagonal * marginal_covs_1, axis=[-3, -2, -1]
        ) + 2.0 * tf.reduce_sum(
            input_tensor=precision_2.block_sub_diagonal * subsequent_covs_1, axis=[-3, -2, -1]
        )
        tf.debugging.assert_equal(tf.shape(trace), batch_shape)

        # (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁)
        # [... num_transitions + 1, state_dim]
        marginal_mean_q = dist_q.marginal_means
        marginal_mean_q = tf.stop_gradient(marginal_mean_q)

        mean_diff = dist_p.marginal_means - marginal_mean_q
        # if P₂ = LLᵀ, calculate [Lᵀ(μ₂ - μ₁)] [... num_transitions + 1, state_dim]
        l_mean_diff = precision_2.cholesky.dense_mult(mean_diff, transpose_left=True)
        mahalanobis = tf.reduce_sum(input_tensor=l_mean_diff * l_mean_diff, axis=[-2, -1])  # [...]
        tf.debugging.assert_equal(tf.shape(mahalanobis), batch_shape)

        dim = (dist_q.num_transitions + 1) * dist_q.state_dim
        dim = tf.cast(dim, default_float())

        q_log_det_precision = dist_q.log_det_precision()
        p_log_det_precision = dist_p.log_det_precision()

        q_log_det_precision = tf.stop_gradient(q_log_det_precision)
        # p_log_det_precision = tf.stop_gradient(p_log_det_precision)

        k_l = 0.5 * (
                trace + mahalanobis - dim - p_log_det_precision + q_log_det_precision
        )

        tf.debugging.assert_equal(tf.shape(k_l), batch_shape)

        return k_l

    def update_prior_sde(self, convergence_tol=1e-4):

        # @property
        def dist_p() -> StateSpaceModel:
            fx_mus = self.fx_mus[:, :-1, :]
            fx_covs = self.fx_covs[:, :-1, :, :]

            return linearize_sde(sde=self.prior_sde, transition_times=self.time_points, q_mean=fx_mus,
                                 q_covar=fx_covs, initial_mean=self.initial_mean,
                                 initial_chol_covariance=self.initial_chol_cov,
                                 )

        def loss():
            return self.kl(dist_p=dist_p()) + self.loss_lin(dist_p=dist_p())
            # return -1. * self.posterior_kalman.log_likelihood() #+ self.loss_lin()

        old_val = self.prior_sde.trainable_variables
        self.prior_sde_optimizer.minimize(loss, self.prior_sde.trainable_variables)
        new_val = self.prior_sde.trainable_variables

        diff_sq_norm = tf.reduce_sum(tf.square(old_val[0] - new_val[0]))
        if diff_sq_norm < convergence_tol:
            has_converged = True
        else:
            has_converged = False

        return has_converged

    def loss_lin(self, dist_p: StateSpaceModel = None):
        """
        KL[p || p_{lin}]
        """

        if dist_p is None:
            dist_p = self.dist_p_ssm

        m, S = self.dist_q.marginals
        S = tf.stop_gradient(S)
        m = tf.stop_gradient(m)

        # removing batch and the last m and S. FIXME: check last point removal
        m = tf.squeeze(m, axis=0)[:-1]
        S = tf.squeeze(S, axis=0)[:-1]

        # convert from A and b to SDE Q's drift and offset
        A = tf.squeeze(dist_p.state_transitions, axis=0)
        b = tf.squeeze(dist_p.state_offsets, axis=0)
        A = (A - tf.eye(self.state_dim, dtype=A.dtype))/(self.grid[1] - self.grid[0])
        b = b / (self.grid[1] - self.grid[0])

        A = tf.stop_gradient(A)
        b = tf.stop_gradient(b)

        # -1 * A as the function expects A without the negative sign i.e. drift = - A*x + b
        lin_loss = KL_sde(self.prior_sde, -1 * A, b, m, S, dt=(self.grid[1] - self.grid[0]))
        return lin_loss

    def classic_elbo(self) -> tf.Tensor:
        """
            Compute the ELBO.
        """
        # s ~ q(s) = N(μ, P)
        fx_mus, fx_covs = self.dist_q.marginals
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0][..., None]
        fx_mus = tf.gather_nd(tf.reshape(fx_mus, (-1, 1)), indices)
        fx_covs = tf.gather_nd(tf.reshape(fx_covs, (-1, 1)), indices)

        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(
                fx_mus, fx_covs, self._observations
            )
        )
        # KL[q(sₓ) || p(sₓ)]
        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p_ssm))

        lin_loss = self.loss_lin()

        print(f"kl_fx = {kl_fx}; ve_fx = {ve_fx}; lin_loss = {lin_loss}")
        return ve_fx - kl_fx - lin_loss

    def run(self, update_prior: bool = False) -> [list, dict]:
        """
        Run inference and (if required) update prior till convergence.
        """
        self.elbo_vals.append(self.classic_elbo().numpy().item())
        print(f"SSM: Starting ELBO {self.elbo_vals[-1]};")

        while len(self.elbo_vals) < 2 or tf.math.abs(self.elbo_vals[-2] - self.elbo_vals[-1]) > 1e-4:
            sites_converged = False
            while not sites_converged:
                if isinstance(self.likelihood, Gaussian):
                    # FIXME: Testing it for Gaussian. Clean later if works and is stable.
                    print("SSM : As Likelihood is Gaussian, performing a single site update with LR=1.")
                    orig_lr = self.sites_lr
                    self.sites_lr = 1.
                    self.update_sites()
                    sites_converged = True
                    if update_prior:
                        self.sites_lr = orig_lr
                    else:
                        self.sites_lr = 1e-3
                else:
                    sites_converged = self.update_sites()

                self.elbo_vals.append(self.classic_elbo().numpy().item())
                print(f"SSM: ELBO {self.elbo_vals[-1]}!")

            if update_prior:
                for _ in range(4):
                    self.update_prior_sde()
                    # Linearize the prior
                    self._linearize_prior()
                    self._store_prior_param_vals()

                    # FIXME: ONLY FOR OU: Steady state covariance
                    if isinstance(self.prior_sde, OrnsteinUhlenbeckSDE):
                        self.initial_chol_cov = tf.linalg.cholesky(
                            (self.prior_sde.q / (2 * (-1 * self.prior_sde.decay))) * tf.ones_like(self.initial_chol_cov))
            else:
                # only linearize the prior
                self._linearize_prior()

            print(f"SSM: ELBO {self.elbo_vals[-1]}; Decaying LR!!!")
            self.sites_lr = self.sites_lr / 2
            self.prior_sde_optimizer.learning_rate = self.prior_sde_optimizer.learning_rate / 2

        return self.elbo_vals, self.prior_params

