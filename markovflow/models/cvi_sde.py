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
from gpflow.likelihoods import Likelihood
from gpflow.base import Parameter
from gpflow import default_float
from gpflow.quadrature import NDiagGHQuadrature

from markovflow.kalman_filter import UnivariateGaussianSitesNat
from markovflow.models.variational_cvi import CVIGaussianProcess
from markovflow.mean_function import MeanFunction
from markovflow.sde.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import linearize_sde
from markovflow.emission_model import EmissionModel
from markovflow.kalman_filter import KalmanFilterWithSites
from markovflow.kernels import OrnsteinUhlenbeck


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
            learning_rate=0.1
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

        self._intialize_mean_statistic()
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.1, 2, decay_rate=0.8, staircase=True)
        self.itr = 0
        self.prior_sde_optimizer = tf.optimizers.Adam()

        self.sites_nat2 = tf.ones_like(self.grid, dtype=self.observations.dtype)[..., None, None] * -1e-10

    def _intialize_mean_statistic(self):
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

    @property
    def dist_p(self) -> StateSpaceModel:
        """
        Return the :class:`~markovflow.state_space_model.StateSpaceModel` representation of the prior process.

        Here, we approximate (linearize) the prior SDE based on the grid.
        """
        # FIXME: check this. because of timepoints[:-1]?
        fx_mus = self.fx_mus[:, :-1, :]
        fx_covs = self.fx_covs[:, :-1, :, :]

        return linearize_sde(sde=self.prior_sde, transition_times=self.time_points, q_mean=fx_mus,
                             q_covar=fx_covs, initial_mean=self.initial_mean,
                             initial_chol_covariance=self.initial_chol_cov,
                            )

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
            state_space_model=self.dist_p,
            emission_model=self.generate_emission_model(tf.reshape(self.time_points, (-1))),
            sites=self.sites,
        )

    def update_sites(self) -> None:
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
        lr = self.learning_rate
        new_nat1 = (1 - lr) * self.data_sites.nat1 + lr * grads[0]
        new_nat2 = (1 - lr) * self.data_sites.nat2 + lr * grads[1][..., None]

        self.data_sites.nat2.assign(new_nat2)
        self.data_sites.nat1.assign(new_nat1)

        dist_q = self.dist_q
        fx_mus, fx_covs = dist_q.marginal_means, dist_q.marginal_covariances

        self.fx_mus = fx_mus
        self.fx_covs = fx_covs

    def kl(self) -> tf.Tensor:
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
        dist_p = self.dist_p

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
        p_log_det_precision = tf.stop_gradient(p_log_det_precision)

        k_l = 0.5 * (
                trace + mahalanobis - dim - p_log_det_precision + q_log_det_precision
        )

        tf.debugging.assert_equal(tf.shape(k_l), batch_shape)

        return k_l

    def update_prior_sde(self):

        # def loss():
        #     return self.kl() #+ self.loss_lin()
        def loss():
            return -1. * self.posterior_kalman.log_likelihood() #+ self.loss_lin()

        lr = self.lr_scheduler(self.itr)
        self.prior_sde_optimizer.learning_rate = lr
        self.prior_sde_optimizer.minimize(loss, self.prior_sde.trainable_variables)
        self.itr += 1

    def loss_lin(self):
        def E_sde(m: tf.Tensor, S: tf.Tensor):
            """
            E_sde = 0.5 * <(f-f_L)^T \sigma^{-1} (f-f_L)>_{q_t}.

            Apply Gaussian quadrature method to approximate the integral.
            """
            assert self.state_dim == 1
            quadrature_pnts = 20

            def func(x, t=None):
                # Adding N information
                x = tf.transpose(x, perm=[1, 0, 2])
                n_pnts = x.shape[1]
                A = tf.squeeze(self.dist_p.state_transitions, axis=0)
                b = tf.squeeze(self.dist_p.state_offsets, axis=0)

                # convert from A and b to SDE drift and offset
                A = (A - tf.eye(self.state_dim, dtype=A.dtype))/(self.grid[1] - self.grid[0])
                b = b / (self.grid[1] - self.grid[0])

                A = tf.repeat(A, n_pnts, axis=1)
                b = tf.repeat(b, n_pnts, axis=1)
                b = tf.expand_dims(b, axis=-1)

                tmp = self.prior_sde.drift(x=x, t=t) - ((x * A) - b)
                tmp = tmp * tmp

                sigma = self.prior_sde.q

                val = tmp * (1 / sigma)

                return tf.transpose(val, perm=[1, 0, 2])

            diag_quad = NDiagGHQuadrature(self.state_dim, quadrature_pnts)
            e_sde = diag_quad(func, m, tf.squeeze(S, axis=-1))

            return 0.5 * tf.reduce_sum(e_sde)

        m, S = self.dist_q.marginals
        S = tf.stop_gradient(S)
        m = tf.stop_gradient(m)

        # removing batch
        m = tf.squeeze(m, axis=0)[:-1]
        S = tf.squeeze(S, axis=0)[:-1]
        return E_sde(m, S) * (self.grid[1] - self.grid[0]) # Riemann sum

    def classic_elbo(self) -> tf.Tensor:
        """
            Compute the ELBO the classic way. That is:

            .. math:: ℒ(q) = Σᵢ ∫ log(p(yᵢ | f)) q(f) df - KL[q(f) ‖ p(f)]

            .. note:: This is mostly for testing purposes and should not be used for optimization.

            :return: A scalar tensor representing the ELBO.
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
        kl_fx = tf.reduce_sum(self.dist_q.kl_divergence(self.dist_p))

        lin_loss = self.loss_lin()

        # print(f"kl_fx = {kl_fx}; ve_fx = {ve_fx}; lin_loss = {lin_loss}")
        return ve_fx - kl_fx - lin_loss
