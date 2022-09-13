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
"""Module containing a model for Variational-CVI in SDE models"""
from typing import Tuple

import tensorflow as tf
from gpflow.likelihoods import Likelihood
from gpflow.base import Parameter
from gpflow import default_float
from gpflow.quadrature import NDiagGHQuadrature
from gpflow.base import TensorType
from gpflow.probability_distributions import Gaussian

from markovflow.models import MarkovFlowModel
from markovflow.kalman_filter import UnivariateGaussianSitesNat
from markovflow.sde.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import linearize_sde
from markovflow.emission_model import EmissionModel
from markovflow.kalman_filter import KalmanFilterWithSparseSites
from markovflow.sde.sde_utils import squared_drift_difference_along_Gaussian_path, LinearDrift
from markovflow.models.variational_cvi import gradient_transformation_mean_var_to_expectation


class CVISDESparseSites(MarkovFlowModel):
    """
    Provides a site-based parameterization to the variational posterior over the state trajectory of a dynamical system.

    The prior SDE is an arbitrary SDE of the form:
                    p : dx(t) = f_p(x_t, t) dt + Σ dB(t).

    The prior SDE is linearized along the current posterior and is represented by
                    p_{L} : dx(t) = f_L(x_t, t) dt + Σ dB(t) where f_L(x_t, t) = A_{pl}(t)x(t) + b_{pl}(t) ,
    where the optimal parameters of the linearized drift is calculated using Girsanov theorem.

    The posterior is defined by the linear SDE of the form:
                    q : dx(t) = f_q(x_t, t) dt + Σ dB(t) where f_q(x_t, t) = A_q(t)x(t) + b_q(t).

    ELBO of the model is defined as:
    E_{q}[log p(Y|X)] - ( KL[q||p_{L}] + 0.5 * E_{q}[||f_L - f_p||^2_{Σ^{-1}}] + E_{q}[(f_q - f_L){Σ^{-1}}(f_L - f_p)]).

    The term E_{q}[||f_L - f_p||^2_{Σ^{-1}}] is termed as linearization loss as it can be interpreted as the metric for
    the difference between the prior SDE and the linearized SDE along the current posterior.

    The term E_{q}[(f_q - f_L){Σ^{-1}}(f_L - f_p)] is termed as the cross term.

    """

    def __init__(
            self,
            prior_sde: SDE,
            time_grid: TensorType,
            input_data: Tuple[TensorType, TensorType],
            likelihood: Likelihood,
            learning_rate=0.1,
            prior_initial_state: Gaussian = None,
            initial_posterior_path: Gaussian = None,
    ) -> None:
        """
        :param prior_sde: Prior SDE over the latent states, x.
        :param time_grid: Grid over time with shape ``batch_shape + [grid_size]``
        :param input_data: A tuple containing the observed data:

            * Time points of observations with shape ``batch_shape + [num_data]``
            * Observations with shape ``batch_shape + [num_data, observation_dim]``

        :param likelihood: A likelihood for the observations of the model.
        :param learning_rate: The learning rate of the algorithm.
        :param prior_initial_state: A Gaussian prior on the initial state.
        :param initial_posterior_path: A Gaussian prior on the initial posterior path.

        Note: Currently, only batch shape 1 is supported.
        """
        super().__init__()

        self._likelihood = likelihood
        self.time_grid = time_grid
        self.prior_sde = prior_sde

        self._observations_time_points = input_data[0]
        self._observations = input_data[1]

        self.output_dim = 1
        self.state_dim = self._observations.shape[-1]

        self.sites = UnivariateGaussianSitesNat(
            nat1=Parameter(tf.zeros_like(self._observations_time_points)[..., None]),
            nat2=Parameter(tf.ones_like(self._observations_time_points)[..., None, None] * -1e-10),
            log_norm=Parameter(tf.zeros_like(self._observations_time_points)[..., None]),
        )
        self.sites_lr = learning_rate

        # Initialize the prior on the initial state
        self.prior_initial_state = self._initialize_initial_state_prior(prior_initial_state)

        # Initialize posterior path
        initial_posterior_path = self._initialize_posterior_path(initial_posterior_path)
        self.fx_mus = tf.cast(tf.reshape(initial_posterior_path.mu, (1, self.time_grid.shape[0], self.state_dim)),
                              dtype=self._observations.dtype)
        self.fx_covs = tf.cast(
            tf.reshape(initial_posterior_path.cov, (1, self.time_grid.shape[0], self.state_dim, self.state_dim)),
            dtype=self._observations.dtype)

        # Linearize prior
        self._update_linearize_path()
        self._linearize_prior()

        self.obs_sites_indices = tf.where(tf.equal(self.time_grid[..., None], self._observations_time_points))[:, 0][
            ..., None]
        self.dt = self.time_grid[1] - self.time_grid[0]

    def _initialize_initial_state_prior(self, prior_initial_state: Gaussian):
        """Initialize the prior on the initial state."""
        if prior_initial_state is None:
            prior_initial_state = Gaussian(mu=tf.zeros((1, self.state_dim), dtype=self._observations.dtype),
                                           cov=self.prior_sde.q * tf.ones((1, self.state_dim, self.state_dim),
                                                                          dtype=self._observations.dtype))

        return prior_initial_state

    def _initialize_posterior_path(self, initial_posterior_path: Gaussian):
        """Initialize posterior path."""
        if initial_posterior_path is None:
            initial_posterior_path = Gaussian(mu=tf.ones((self.time_grid.shape[0], self.state_dim),
                                                         dtype=self._observations.dtype),
                                              cov=tf.ones((self.time_grid.shape[0], self.state_dim, self.state_dim),
                                                          dtype=self._observations.dtype))

        return initial_posterior_path

    def _update_linearize_path(self):
        """
        Store the linearization path i.e. the current posterior path as a Gaussian.
        """
        self.linearization_path = Gaussian(mu=tf.identity(self.fx_mus[:, :-1, :]),
                                           cov=tf.identity(self.fx_covs[:, :-1, :, :]))

    def _linearize_prior(self):
        """
            Set the :class:`~markovflow.state_space_model.StateSpaceModel` representation of the prior process.
            Here, we approximate (linearize) the prior SDE based on the grid.
        """
        self.dist_p = linearize_sde(sde=self.prior_sde, transition_times=self.time_points,
                                    linearization_path=self.linearization_path,
                                    initial_state=self.prior_initial_state
                                    )

    @property
    def time_points(self) -> TensorType:
        """
        Return the time points of the observations.
        :return: A tensor with shape ``batch_shape + [grid_size]``.
        """
        return self.time_grid

    @property
    def dist_q(self) -> StateSpaceModel:
        """
        Construct the :class:`~markovflow.state_space_model.StateSpaceModel` representation of
        the posterior process indexed at the time points.
        """
        return self.posterior_kalman.posterior_state_space_model()

    def generate_emission_model(self, time_points: TensorType) -> EmissionModel:
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
    def posterior_kalman(self) -> KalmanFilterWithSparseSites:
        """Build the Kalman filter object from the prior state space models and the sites."""
        return KalmanFilterWithSparseSites(
            state_space_model=self.dist_p,
            emission_model=self.generate_emission_model(tf.reshape(self.time_points, (-1))),
            sites=self.sites,
            num_grid_points=self.time_grid.shape[0],
            observations_index=self.obs_sites_indices,
            observations=self._observations
        )

    @property
    def posterior(self):
        raise NotImplementedError

    def local_objective(self, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> TensorType:
        """
        Calculate local loss in CVI.

        :param Fmu: Means with shape ``[..., latent_dim]``.
        :param Fvar: Variances with shape ``[..., latent_dim]``.
        :param Y: Observations with shape ``[..., observation_dim]``.
        :return: A local objective with shape ``[...]``.
        """
        return self._likelihood.variational_expectations(Fmu, Fvar, Y)

    def local_objective_and_gradients(self, Fmu: TensorType, Fvar: TensorType) -> [TensorType, TensorType]:
        """
        Return the local objective and its gradients with regard to the expectation parameters.

        :param Fmu: Means :math:`μ` with shape ``[..., latent_dim]``.
        :param Fvar: Variances :math:`σ²` with shape ``[..., latent_dim]``.
        :return: A local objective and gradient with regard to :math:`[μ, σ² + μ²]`.
        """
        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(
                input_tensor=self.local_objective(Fmu, Fvar, self._observations)
            )
        grads = g.gradient(local_obj, [Fmu, Fvar])
        # turn into gradient wrt μ, σ² + μ²
        grads = gradient_transformation_mean_var_to_expectation((Fmu, Fvar), grads)

        return local_obj, grads

    def kl_cross_term(self, dist_q: StateSpaceModel = None) -> TensorType:
        """
        Calculates the cross term of the ELBO,
                    E_{q}[(f_q - f_L){Σ^{-1}}(f_L - f_p)]
        """
        # convert from state transitions of the SSM to SDE's drift and offset
        if dist_q is None:
            dist_q = self.dist_q

        linear_drift_q = LinearDrift()
        linear_drift_q.from_ssm(dist_q, self.dt)

        linear_drift_pl = LinearDrift()
        linear_drift_pl.from_ssm(self.dist_p, self.dt)

        def func(x, t=None, A_q=linear_drift_q.A, b_q=linear_drift_q.b, A_p_l=linear_drift_pl.A,
                 b_p_l=linear_drift_pl.b, sde_p=self.prior_sde):
            # Adding N information
            x = tf.transpose(x, perm=[1, 0, 2])
            n_pnts = x.shape[1]

            A_q = tf.repeat(A_q, n_pnts, axis=1)
            b_q = tf.repeat(b_q, n_pnts, axis=1)
            b_q = tf.expand_dims(b_q, axis=-1)

            A_p_l = tf.repeat(A_p_l, n_pnts, axis=1)
            b_p_l = tf.repeat(b_p_l, n_pnts, axis=1)
            b_p_l = tf.expand_dims(b_p_l, axis=-1)

            prior_drift = sde_p.drift(x=x, t=t)

            fq_fL = ((x * A_q) + b_q) - ((x * A_p_l) + b_p_l)  # (f_q - f_L)
            fl_fp = ((x * A_p_l) + b_p_l) - prior_drift  # (f_l - f_p)

            sigma = sde_p.q
            val = fq_fL * (1 / sigma) * fl_fp

            return tf.transpose(val, perm=[1, 0, 2])

        diag_quad = NDiagGHQuadrature(self.prior_sde.state_dim, 20)
        q_mean = tf.squeeze(self.fx_mus, axis=0)[:-1]
        q_covar = tf.squeeze(self.fx_covs, axis=0)[:-1]

        val = diag_quad(func, q_mean, tf.squeeze(q_covar, axis=-1))
        val = tf.reduce_sum(val) * self.dt

        return val

    def kl_linearization_loss(self, dist_p: StateSpaceModel = None, m: TensorType = None, S: TensorType = None):
        """
        Calculates the linearization loss term of the ELBO which can also be interpreted as the metric for
        the difference between the prior SDE and the linearized SDE along the current posterior,
                                E_{q}[||f_L - f_p||^2_{Σ^{-1}}] .
        """
        if dist_p is None:
            dist_p = self.dist_p

        if m is None:
            m, S = self.dist_q.marginals

        # removing batch and the last m and S.
        m = tf.squeeze(m, axis=0)[:-1]
        S = tf.squeeze(S, axis=0)[:-1]
        q = Gaussian(mu=m, cov=S)

        linear_drift = LinearDrift()
        linear_drift.from_ssm(dist_p, self.dt)

        lin_loss = squared_drift_difference_along_Gaussian_path(self.prior_sde, linear_drift=linear_drift, q=q,
                                                                dt=self.dt)
        return lin_loss

    def update_sites(self):
        """Update of the data-sites following CVI"""
        fx_mus = tf.gather_nd(tf.reshape(self.fx_mus, (-1, 1)), self.obs_sites_indices)
        fx_covs = tf.gather_nd(tf.reshape(self.fx_covs, (-1, 1)), self.obs_sites_indices)

        # get gradient of variational expectations wrt the expectation parameters μ, σ² + μ²
        _, grads = self.local_objective_and_gradients(fx_mus, fx_covs)

        # update
        new_data_nat1 = (1 - self.sites_lr) * self.sites.nat1 + self.sites_lr * grads[0]
        new_data_nat2 = (1 - self.sites_lr) * self.sites.nat2 + self.sites_lr * grads[1][..., None]

        self.sites.nat1.assign(new_data_nat1)
        self.sites.nat2.assign(new_data_nat2)

    def variational_expectation(self, fx_mus: TensorType = None, fx_covs: TensorType = None) -> TensorType:
        """Expected log-likelihood under the current variational posterior"""

        if fx_mus is None or fx_covs is None:
            fx_mus, fx_covs = self.dist_q.marginals

        indices = tf.where(tf.equal(self.time_grid[..., None], self._observations_time_points))[:, 0][..., None]
        fx_mus_obs = tf.gather_nd(tf.reshape(fx_mus, (-1, 1)), indices)
        fx_covs_obs = tf.gather_nd(tf.reshape(fx_covs, (-1, 1)), indices)

        # VE(fₓ) = Σᵢ ∫ log(p(yᵢ | fₓ)) q(fₓ) dfₓ
        ve_fx = tf.reduce_sum(
            input_tensor=self._likelihood.variational_expectations(
                fx_mus_obs, fx_covs_obs, self._observations
            )
        )
        return ve_fx

    def log_likelihood(self) -> TensorType:
        """
        Calculate :math:`log p(y)`.

        :return: A scalar tensor representing the ELBO.
        """
        return self.posterior_kalman.log_likelihood()

    def elbo(self) -> TensorType:
        """
        Calculate the evidence lower bound (ELBO) :math:`log p(y)`.

        This is done by computing the marginal of the model in which the likelihood terms were
        replaced by the Gaussian sites.

        :return: A scalar tensor representing the ELBO.
        """
        return self.log_likelihood()

    def loss(self) -> TensorType:
        """
        Return the loss, which is the negative ELBO.
        """
        return -self.log_likelihood()

    def classic_elbo(self) -> TensorType:
        """
        Compute the ELBO.

        ELBO of the model is defined as:
        E_{q}[log p(Y|X)] - (KL[q||p_{L}] + 0.5 * E_{q}[||f_L - f_p||^2_{Σ^{-1}}] + E_{q}[(f_q-f_L){Σ^{-1}}(f_L-f_p)]).

        """
        # s ~ q(s) = N(μ, P)
        dist_q = self.dist_q
        fx_mus, fx_covs = dist_q.marginals

        ve_fx = self.variational_expectation(fx_mus, fx_covs)
        kl_fx = tf.reduce_sum(dist_q.kl_divergence(self.dist_p))
        lin_loss = self.kl_linearization_loss(m=fx_mus, S=fx_covs)
        cross_term = self.kl_cross_term(dist_q=dist_q)

        return ve_fx - kl_fx - lin_loss - cross_term
