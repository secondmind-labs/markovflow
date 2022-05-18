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
"""
    Module containing a model for variational inference in SDE.

    @inproceedings{pmlr-v1-archambeau07a,
      title = 	 {Gaussian Process Approximations of Stochastic Differential Equations},
      author = 	 {Archambeau, Cedric and Cornford, Dan and Opper, Manfred and Shawe-Taylor, John},
      booktitle = 	 {Gaussian Processes in Practice},
      pages = 	 {1--16},
      year = 	 {2007},
      editor = 	 {Lawrence, Neil D. and Schwaighofer, Anton and Quiñonero Candela, Joaquin},
      volume = 	 {1},
      series = 	 {Proceedings of Machine Learning Research},
      address = 	 {Bletchley Park, UK},
      month = 	 {12--13 Jun},
      publisher =    {PMLR},
      pdf = 	 {http://proceedings.mlr.press/v1/archambeau07a/archambeau07a.pdf},
      url = 	 {https://proceedings.mlr.press/v1/archambeau07a.html},
    }


    @inproceedings{NIPS2007_818f4654,
         author = {Archambeau, C\'{e}dric and Opper, Manfred and Shen, Yuan and Cornford, Dan and Shawe-taylor, John},
         booktitle = {Advances in Neural Information Processing Systems},
         editor = {J. Platt and D. Koller and Y. Singer and S. Roweis},
         pages = {},
         publisher = {Curran Associates, Inc.},
         title = {Variational Inference for Diffusion Processes},
         url = {https://proceedings.neurips.cc/paper/2007/file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf},
         volume = {20},
         year = {2007}
    }
"""

import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions

from gpflow.likelihoods import Likelihood

from markovflow.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from gpflow.quadrature import NDiagGHQuadrature


class VariationalMarkovGP:
    """
        Variational approximation to a non-linear SDE by a time-varying Gauss-Markov Process of the form
        dx(t) = - A(t) dt + b(t) dW(t)
    """
    def __init__(self, input_data: [tf.Tensor, tf.Tensor], prior_sde: SDE, grid: tf.Tensor, likelihood: Likelihood,
                 lr: float = 0.5):
        """
        Initialize the model.

        """
        assert prior_sde.state_dim == 1, "Currently only 1D is supported."

        self._time_points,  self.observations = input_data
        self.prior_sde = prior_sde
        self.likelihood = likelihood
        self.state_dim = self.prior_sde.state_dim
        self.DTYPE = self.observations.dtype

        self.grid = grid
        self.N = self.grid.shape[0]
        self.dt = float(self.grid[1] - self.grid[0])
        self.observations_time_points = self._time_points

        self.A = tf.ones((self.N, self.state_dim, self.state_dim), dtype=self.DTYPE)  # [N, D, D]
        self.b = tf.zeros((self.N, self.state_dim), dtype=self.DTYPE)  # [N, D]

        # p(x0)
        self.p_initial_mean = tf.zeros(self.state_dim, dtype=self.DTYPE)
        self.p_initial_cov = (self.prior_sde.q.numpy()/(2 * self.prior_sde.decay.numpy())) * tf.ones((self.state_dim, self.state_dim), dtype=self.DTYPE)

        # q(x0)
        self.q_initial_mean = tf.zeros(self.state_dim, dtype=self.DTYPE)
        self.q_initial_cov = tf.ones((self.state_dim, self.state_dim), dtype=self.DTYPE)

        self.lambda_lagrange = tf.zeros_like(self.b)  # [N, D]
        self.psi_lagrange = tf.zeros_like(self.b)   # [N, D]

        self.lr = lr
        self.prior_sde_optimizer = tf.optimizers.Adam()

    @property
    def forward_pass(self) -> (tf.Tensor, tf.Tensor):
        """
        Computes the mean and variance of the SDE using SSM.

        The returned m and S have initial values appended too.
        """

        state_transition = tf.eye(self.state_dim, dtype=self.A.dtype) - self.A * self.dt
        state_offset = self.b * self.dt
        q = tf.repeat(tf.reshape(self.prior_sde.q * self.dt, (1, 1, 1)), state_transition.shape[0], axis=0)

        ssm = StateSpaceModel(initial_mean=self.q_initial_mean,
                              chol_initial_covariance=tf.linalg.cholesky(self.q_initial_cov),
                              state_transitions=state_transition,
                              state_offsets=state_offset,
                              chol_process_covariances=tf.linalg.cholesky(q)
                              )

        return ssm.marginal_means[:-1], ssm.marginal_covariances[:-1]

    def E_sde(self, m: tf.Tensor, S: tf.Tensor):
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

            A = tf.repeat(self.A, n_pnts, axis=1)
            b = tf.repeat(self.b, n_pnts, axis=1)
            b = tf.expand_dims(b, axis=-1)

            tmp = self.prior_sde.drift(x=x, t=t) + ((x * A) - b)
            tmp = tmp * tmp

            sigma = self.prior_sde.q

            val = tmp * (1 / sigma)

            return tf.transpose(val, perm=[1, 0, 2])

        diag_quad = NDiagGHQuadrature(self.state_dim, quadrature_pnts)
        e_sde = diag_quad(func, m, tf.squeeze(S, axis=-1))

        return 0.5 * tf.reduce_sum(e_sde)

    def grad_E_sde(self, m: tf.Tensor, S: tf.Tensor):
        """
        Gradient of E_sde wrt m and S.
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch([m, S])
            E_sde = self.E_sde(m, S)

        dE_dm, dE_dS = g.gradient(E_sde, [m, S])
        dE_dS = tf.squeeze(dE_dS, axis=-1)

        return dE_dm, dE_dS

    def update_initial_statistics(self, lr=0.1):
        """
        Update the initial statistics.
        """
        q_initial_mean = self.p_initial_mean - tf.reshape(self.lambda_lagrange[0], self.q_initial_mean.shape) * tf.reshape(self.p_initial_cov, self.q_initial_mean.shape)
        q_initial_cov = 1/(2 * tf.reshape(self.psi_lagrange[0], self.q_initial_cov.shape) + 1/self.p_initial_cov)

        self.q_initial_mean = (1-lr) * self.q_initial_mean + lr * q_initial_mean
        self.q_initial_cov = (1-lr) * self.q_initial_cov + lr * q_initial_cov

    def _jump_conditions(self, m: tf.Tensor, S: tf.Tensor):
        """
        Declare jump conditions on a bigger grid with values only where the observations are available.
        """
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0]
        m_obs_t = tf.gather(m, indices, axis=0)
        S_obs_t = tf.gather(S, indices, axis=0)

        _, grads = self.local_objective_and_gradients(m_obs_t, tf.squeeze(S_obs_t, axis=-1))

        # Put grads in a bigger grid back
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0][..., None]
        d_obs_m = tf.scatter_nd(indices, grads[0], self.lambda_lagrange.shape)
        d_obs_S = tf.scatter_nd(indices, grads[1], self.psi_lagrange.shape)

        return d_obs_m, d_obs_S

    def update_lagrange(self, m: tf.Tensor, S: tf.Tensor):
        """
        Backward pass incorporating jump conditions.
        This function updates the lagrange multiplier values.

        d psi(t)/dt = 2 * psi(t) * A(t) - dE_{sde}/ds(t)
        d lambda(t)/dt = A(t).T * lambda(t) - dE_{sde}/dm(t)

        At the time of observation, jump conditions:
            psi(t) = psi(t) + dE_{obs}/dS(t)
            lambda(t) = lambda(t) + dE_{obs}/dm(t)

        """
        dEdm, dEdS = self.grad_E_sde(m, S)

        d_obs_m, d_obs_S = self._jump_conditions(m, S)

        lambda_lagrange = np.zeros_like(self.lambda_lagrange)
        psi_lagrange = np.zeros_like(self.psi_lagrange)

        for t in range(self.N-1, 0, -1):
            d_psi = psi_lagrange[t] * self.A[t] + psi_lagrange[t] * self.A[t] - dEdS[t]
            d_lambda = self.A[t] * lambda_lagrange[t] - dEdm[t]

            psi_lagrange[t - 1] = psi_lagrange[t] - self.dt * d_psi - d_obs_S[t-1]
            lambda_lagrange[t - 1] = lambda_lagrange[t] - self.dt * d_lambda - d_obs_m[t-1]

        self.psi_lagrange = tf.convert_to_tensor(psi_lagrange)
        self.lambda_lagrange = tf.convert_to_tensor(lambda_lagrange)

    def local_objective(self, Fmu: tf.Tensor, Fvar: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Calculate local loss..

        :param Fmu: Means with shape ``[..., latent_dim]``.
        :param Fvar: Variances with shape ``[..., latent_dim]``.
        :param Y: Observations with shape ``[..., observation_dim]``.
        :return: A local objective with shape ``[...]``.
        """
        return self.likelihood.variational_expectations(Fmu, Fvar, Y)

    def local_objective_and_gradients(self, Fmu: tf.Tensor, Fvar: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Return the local objective and its gradients.

        :param Fmu: Means :math:`μ` with shape ``[..., latent_dim]``.
        :param Fvar: Variances :math:`σ²` with shape ``[..., latent_dim]``.
        :return: A local objective and gradient with regard to :math:`[μ, σ²]`.
        """
        with tf.GradientTape() as g:
            g.watch([Fmu, Fvar])
            local_obj = tf.reduce_sum(
                input_tensor=self.local_objective(Fmu, Fvar, self.observations)
            )
        grads = g.gradient(local_obj, [Fmu, Fvar])

        return local_obj, grads

    def update_param(self, m: tf.Tensor, S: tf.Tensor):
        """
        Update the params A(t) and b(t).

        \tilde{A(t)} = - <\frac{df}{dx}>_{qt} + 2 Q \psi(t)
        \tilde{b(t)} = <f(x)>_{qt} + \tilde{A(t)} * m(t) - Q \lambda(t)

        A(t) = (1 - lr) * A(t) + lr * \tilde{A(t)}
        b(t) = (1 - lr) * b(t) + lr *  \tilde{b(t)}

        """
        var = self.prior_sde.q

        A_tilde = tf.squeeze(-self.prior_sde.expected_gradient_drift(m[..., None], S), axis=-1) + 2. * var * self.psi_lagrange
        b_tilde = tf.squeeze(self.prior_sde.expected_drift(m[..., None], S), axis=-1) + A_tilde * m - self.lambda_lagrange * var

        self.A = (1 - self.lr) * self.A + self.lr * A_tilde[..., None]
        self.b = (1 - self.lr) * self.b + self.lr * b_tilde

    def KL_initial_state(self):
        """
        KL[q(x0) || p(x0)]
        """
        dist_qx0 = distributions.Normal(self.q_initial_mean, tf.linalg.cholesky(self.q_initial_cov))
        dist_px0 = distributions.Normal(self.p_initial_mean, tf.linalg.cholesky(self.p_initial_cov))
        return distributions.kl_divergence(dist_qx0, dist_px0)

    def elbo(self) -> float:
        """ Variational lower bound to the marginal likelihood """
        m, S = self.forward_pass
        E_sde = tf.reduce_sum(self.E_sde(m, S))
        E_sde = E_sde * self.dt

        #
        KL_q0_p0 = self.KL_initial_state()

        # E_obs
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0]
        m_obs_t = tf.gather(m, indices, axis=0)
        S_obs_t = tf.gather(S, indices, axis=0)

        E_obs = self.likelihood.variational_expectations(m_obs_t, tf.squeeze(S_obs_t, axis=-1), self.observations)
        E_obs = tf.reduce_sum(E_obs)

        E = E_obs - E_sde - KL_q0_p0

        return E.numpy().item()

    def _update_prior(self, m: tf.Tensor, S: tf.Tensor):
        """Internal function to calculate the prior SDE."""
        def func():
            return self.E_sde(m, S) * self.dt #+ self.KL_initial_state()

        self.prior_sde_optimizer.minimize(func, self.prior_sde.trainable_variables)

    def update_prior(self):
        """
        Function to update the prior SDE.
        """
        m, S = self.forward_pass
        # lr = self.lr_scheduler_prior(itr)
        # self.prior_sde_optimizer.learning_rate = lr
        self._update_prior(m, S)

    def run_inference(self):
        """
        Run a single loop of inference.
        """
        m, S = self.forward_pass
        self.update_lagrange(m, S)
        self.update_param(m, S)

