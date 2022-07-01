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
import wandb
from gpflow.likelihoods import Likelihood

from markovflow.sde import SDE
from markovflow.sde.sde import PriorOUSDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import KL_sde
from markovflow.sde.sde_utils import gaussian_log_predictive_density


class VariationalMarkovGP:
    """
        Variational approximation to a non-linear SDE by a time-varying Gauss-Markov Process of the form
        dx(t) = - A(t) dt + b(t) dW(t)
    """
    def __init__(self, input_data: [tf.Tensor, tf.Tensor], prior_sde: SDE, grid: tf.Tensor, likelihood: Likelihood,
                 lr: float = 0.5, prior_params_lr: float = 0.01, test_data: [tf.Tensor, tf.Tensor] = None,
                 initial_state_lr: float = 0.01):
        """
        Initialize the model.

        """
        assert prior_sde.state_dim == 1, "Currently only 1D is supported."

        self._time_points,  self.observations = input_data
        self.prior_sde = prior_sde
        self.likelihood = likelihood
        self.state_dim = self.prior_sde.state_dim
        self.DTYPE = self.observations.dtype
        self.test_data = test_data

        self.orig_grid = grid
        self.grid = grid[1:]
        self.N = grid.shape[0]

        self.dt = float(self.grid[1] - self.grid[0])
        self.observations_time_points = self._time_points

        self.A = tf.ones((self.N, self.state_dim, self.state_dim), dtype=self.DTYPE)  # [N, D, D]
        self.b = tf.zeros((self.N, self.state_dim), dtype=self.DTYPE)  # [N, D]

        # p(x0)
        self.p_initial_mean = tf.zeros(self.state_dim, dtype=self.DTYPE)
        self.p_initial_cov = self.prior_sde.q.numpy() * tf.ones((self.state_dim, self.state_dim), dtype=self.DTYPE)

        # q(x0)
        self.q_initial_mean = tf.identity(self.p_initial_mean)
        self.q_initial_cov = tf.identity(self.p_initial_cov)

        self.lambda_lagrange = tf.zeros_like(self.b)  # [N, D]
        self.psi_lagrange = tf.ones_like(self.b) * -1e-10   # [N, D]

        self.q_lr = lr  # lr for variational parameters A and b
        self.p_lr = prior_params_lr  # lr for prior parameters
        self.x_lr = initial_state_lr  # lr for initial statistics

        self.prior_sde_optimizer = tf.optimizers.SGD(lr=self.p_lr)

        self.elbo_vals = []
        self.dist_q_ssm = None

        self.prior_params = {}
        for i, param in enumerate(self.prior_sde.trainable_variables):
            self.prior_params[i] = [param.numpy().item()]

    def _store_prior_param_vals(self):
        """Update the list storing the prior sde parameter values"""
        for i, param in enumerate(self.prior_sde.trainable_variables):
            self.prior_params[i].append(param.numpy().item())

    @property
    def forward_pass(self) -> (tf.Tensor, tf.Tensor):
        """
        Computes the mean and variance of the SDE using SSM.

        The returned m and S have initial values appended too.
        """
        A = self.A[:-1]
        b = self.b[:-1]
        state_transition = tf.eye(self.state_dim, dtype=self.A.dtype) - A * self.dt
        state_offset = b * self.dt

        q = tf.repeat(tf.reshape(self.prior_sde.q * self.dt, (1, 1, 1)), state_transition.shape[0], axis=0)

        self.dist_q_ssm = StateSpaceModel(initial_mean=self.q_initial_mean,
                                          chol_initial_covariance=tf.linalg.cholesky(self.q_initial_cov),
                                          state_transitions=state_transition,
                                          state_offsets=state_offset,
                                          chol_process_covariances=tf.linalg.cholesky(q)
                                          )

        return self.dist_q_ssm.marginal_means, self.dist_q_ssm.marginal_covariances

    def grad_E_sde(self, m: tf.Tensor, S: tf.Tensor):
        """
        Gradient of E_sde wrt m and S.
        """
        with tf.GradientTape(persistent=True) as g:
            g.watch([m, S])
            E_sde = KL_sde(self.prior_sde, self.A, self.b, m, S, self.dt)

        dE_dm, dE_dS = g.gradient(E_sde, [m, S])
        dE_dS = tf.squeeze(dE_dS, axis=-1)

        return dE_dm/self.dt, dE_dS/self.dt  # Due to Reimann sum

    def update_initial_statistics(self, convergence_tol=1e-4):
        """
        Update the initial statistics.
        """
        q_initial_mean = self.p_initial_mean - tf.reshape(self.lambda_lagrange[0],
                                                          self.q_initial_mean.shape) * tf.reshape(self.p_initial_cov,
                                                                                                  self.q_initial_mean.shape)
        q_initial_cov = 1/(2 * tf.reshape(self.psi_lagrange[0], self.q_initial_cov.shape) + 1/self.p_initial_cov)

        # compute criterion for convergence
        q_mean_diff_sq_norm = tf.reduce_sum(tf.square(self.q_initial_mean - q_initial_mean))
        q_cov_diff_sq_norm = tf.reduce_sum(tf.square(self.q_initial_cov - q_initial_cov))

        self.q_initial_mean = (1 - self.x_lr) * self.q_initial_mean + self.x_lr * q_initial_mean
        self.q_initial_cov = (1 - self.x_lr) * self.q_initial_cov + self.x_lr * q_initial_cov

        if (q_mean_diff_sq_norm < convergence_tol) & (q_cov_diff_sq_norm < convergence_tol):
            has_converged = True
        else:
            has_converged = False

        return has_converged

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

    def update_param(self, m: tf.Tensor, S: tf.Tensor, convergence_tol: float = 1e-4):
        """
        Update the params A(t) and b(t).

        \tilde{A(t)} = - <\frac{df}{dx}>_{qt} + 2 Q \psi(t)
        \tilde{b(t)} = <f(x)>_{qt} + \tilde{A(t)} * m(t) - Q \lambda(t)

        A(t) = (1 - lr) * A(t) + lr * \tilde{A(t)}
        b(t) = (1 - lr) * b(t) + lr *  \tilde{b(t)}

        """
        assert m.shape[0] == self.A.shape[0]

        var = self.prior_sde.q

        A_tilde = tf.squeeze(-self.prior_sde.expected_gradient_drift(m[..., None], S), axis=-1) + 2. * var * self.psi_lagrange
        b_tilde = tf.squeeze(self.prior_sde.expected_drift(m[..., None], S), axis=-1) + A_tilde * m - self.lambda_lagrange * var

        # compute criterion for convergence
        A_diff_sq_norm = tf.reduce_sum(tf.square(self.A - A_tilde[..., None]))
        b_diff_sq_norm = tf.reduce_sum(tf.square(self.b - b_tilde))

        self.A = (1 - self.q_lr) * self.A + self.q_lr * A_tilde[..., None]
        self.b = (1 - self.q_lr) * self.b + self.q_lr * b_tilde

        if (A_diff_sq_norm < convergence_tol) & (b_diff_sq_norm < convergence_tol):
            has_converged = True
        else:
            has_converged = False

        return has_converged

    def KL_initial_state(self):
        """
        KL[q(x0) || p(x0)]
        """
        q0_mean = tf.stop_gradient(self.q_initial_mean)
        q0_cov = tf.stop_gradient(self.q_initial_cov)

        dist_qx0 = distributions.Normal(q0_mean, tf.linalg.cholesky(q0_cov))
        dist_px0 = distributions.Normal(self.p_initial_mean, tf.linalg.cholesky(self.p_initial_cov))
        return distributions.kl_divergence(dist_qx0, dist_px0)

    def elbo(self) -> float:
        """ Variational lower bound to the marginal likelihood """
        m, S = self.forward_pass

        # remove the final state and the final A and b
        E_sde = KL_sde(self.prior_sde, self.A[:-1], self.b[:-1], m[:-1], S[:-1], self.dt)

        KL_q0_p0 = self.KL_initial_state()

        # E_obs
        indices = tf.where(tf.equal(self.grid[..., None], self.observations_time_points))[:, 0] + 1
        m_obs_t = tf.gather(m, indices, axis=0)
        S_obs_t = tf.gather(S, indices, axis=0)

        E_obs = self.likelihood.variational_expectations(m_obs_t, tf.squeeze(S_obs_t, axis=-1), self.observations)
        E_obs = tf.reduce_sum(E_obs)

        E = E_obs - E_sde - KL_q0_p0
        wandb.log({"VGP-EObs": E_obs})
        wandb.log({"VGP-ESDE": E_sde})
        wandb.log({"VGP-KL-X0": KL_q0_p0})
        # print(f"E_obs: {E_obs}; E_sde: {E_sde}; KL_x0 : {KL_q0_p0}")

        return E.numpy().item()

    def update_prior_sde(self, convergence_tol=1e-4):
        """
        Function to update the prior SDE.
        """
        def func():
            m, S = self.forward_pass

            # remove the final state
            m = tf.stop_gradient(m[:-1])
            S = tf.stop_gradient(S[:-1])

            A = self.A[:-1]
            b = self.b[:-1]

            return KL_sde(self.prior_sde, A, b, m, S, self.dt) + self.KL_initial_state()

        # FIXME: check all variables and not only the first one
        old_val = self.prior_sde.trainable_variables[0].numpy().item()
        self.prior_sde_optimizer.minimize(func, self.prior_sde.trainable_variables)
        new_val = self.prior_sde.trainable_variables[0].numpy().item()

        # FIXME: Only for OU: Steady state covariance
        if isinstance(self.prior_sde, PriorOUSDE):
            self.p_initial_cov = (self.prior_sde.q / (2 * -1 * self.prior_sde.decay)) * tf.ones_like(
                self.p_initial_cov)

        diff_sq_norm = tf.reduce_sum(tf.square(old_val - new_val))
        if diff_sq_norm < convergence_tol:
            has_converged = True
        else:
            has_converged = False

        return has_converged

    def calculate_nlpd(self) -> float:
        """
            Calculate NLPD on the test set
            FIXME: Only in case of Gaussian
        """
        if self.test_data is None:
            return 0.

        m, S = self.forward_pass
        s_std = tf.linalg.cholesky(S + self.likelihood.variance)

        pred_idx = list((tf.where(self.orig_grid == self.test_data[0][..., None])[:, 1]).numpy())
        s_std = tf.reshape(tf.gather(s_std, pred_idx, axis=0), (-1, 1, 1))
        lpd = gaussian_log_predictive_density(mean=tf.gather(m, pred_idx, axis=0), chol_covariance=s_std,
                                              x=tf.reshape(self.test_data[1], (-1,)))
        nlpd = -1 * tf.reduce_mean(lpd)

        return nlpd.numpy().item()

    def run_single_inference(self):
        """
        Run a single loop of inference.
        """
        m, S = self.forward_pass
        self.update_lagrange(m, S)
        converged = self.update_param(m, S)

        return converged

    def run_inference_till_convergence(self, update_prior: bool, update_initial_statistics: bool = True):
        """
        Run inference till convergence
        """
        self.elbo_vals.append(self.elbo())
        print(f"VGP: Starting ELBO {self.elbo_vals[-1]}")
        wandb.log({"VGP-ELBO": self.elbo_vals[-1]})

        while len(self.elbo_vals) < 2 or tf.math.abs(self.elbo_vals[-2] - self.elbo_vals[-1]) > 1e-4:
            itr = 0  # Minimum iterations
            while itr < 5 or (self.elbo_vals[-1] - self.elbo_vals[-2]) > 1e-4:
                self.run_single_inference()
                if update_initial_statistics:
                    self.update_initial_statistics()

                self.elbo_vals.append(self.elbo())

                print(f"VGP: ELBO {self.elbo_vals[-1]}")
                wandb.log({"VGP-ELBO": self.elbo_vals[-1]})
                wandb.log({"VGP-NLPD": self.calculate_nlpd()})

                itr += 1
                if self.elbo_vals[-1] - self.elbo_vals[-2] < 0:
                    print("ELBO increased!!! Breaking the loop")
                    break

            if update_prior:
                prior_converged = False
                while not prior_converged:
                    prior_converged = self.update_prior_sde()
                    self._store_prior_param_vals()

                    for k in self.prior_params.keys():
                        v = self.prior_params[k][-1]
                        wandb.log({"VGP-learning-" + str(k): v})

                self.elbo_vals.append(self.elbo())

            # print(f"VGP: ELBO {self.elbo_vals[-1]}; Decaying LR!!!")
            wandb.log({"VGP-ELBO": self.elbo_vals[-1]})

            if self.elbo_vals[-1] - self.elbo_vals[-2] < 0:
                print("ELBO increased!!! Breaking the main loop too!")
                break
            # self.q_lr = self.q_lr / 2
            # self.x_lr = self.x_lr / 2
            # self.prior_sde_optimizer.learning_rate = self.prior_sde_optimizer.learning_rate / 2

    def run(self, update_prior: bool = False, update_initial_statistics: bool = True) -> [list, dict]:
        """
        Run inference and (if required) update prior till convergence.
        """
        self.run_inference_till_convergence(update_prior, update_initial_statistics)

        return self.elbo_vals, self.prior_params
