from typing import Tuple
import wandb
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.sde_utils import linearize_sde
from markovflow.models.cvi_sde import SDESSM
from markovflow.sde.sde_utils import gaussian_log_predictive_density
from markovflow.sde.sde import PriorOUSDE


class tVGPTrainer:
    def __init__(self, observation_data, likelihood, time_grid, prior_sde, data_sites_lr: float = 0.5,
                 all_sites_lr: float = 0.1, prior_sde_lr: float = 0.1, test_data: Tuple[tf.Tensor, tf.Tensor] = None,
                 update_all_sites: bool = False):
        self.tvgp_model = SDESSM(input_data=observation_data, prior_sde=prior_sde, grid=time_grid,
                                likelihood=likelihood, learning_rate=data_sites_lr, all_sites_lr=all_sites_lr)

        # Initialize the initial statistics of the model
        # FIXME: Find a better way
        if isinstance(prior_sde, PriorOUSDE):
            initial_cov = prior_sde.q / (2 * -1 * prior_sde.decay)  # Steady covariance
            self.tvgp_model.initial_mean = tf.zeros_like(self.tvgp_model.initial_mean)
            self.tvgp_model.initial_chol_cov = tf.linalg.cholesky(tf.reshape(initial_cov, self.tvgp_model.initial_chol_cov.shape))
            self.tvgp_model.fx_covs = self.tvgp_model.initial_chol_cov.numpy().item() ** 2 + 0 * self.tvgp_model.fx_covs
            self.tvgp_model._linearize_prior()
        else:
            self.tvgp_model.initial_mean = observation_data[1][0] + 0. * self.tvgp_model.initial_mean
            self.tvgp_model.initial_chol_cov = 0.5 ** (1 / 2) + 0. * self.tvgp_model.initial_chol_cov
            self.tvgp_model.fx_mus = self.tvgp_model.initial_mean + 0. * self.tvgp_model.fx_mus
            self.tvgp_model.fx_covs = 1. + 0. * self.tvgp_model.fx_covs

        self.update_all_sites = update_all_sites
        self.test_data = test_data
        self.elbo_vals = []
        self.prior_sde_optimizer = tf.optimizers.SGD(lr=prior_sde_lr)
        self.m_step_data = {}
        self.prior_params = {}
        for i, param in enumerate(self.tvgp_model.prior_sde.trainable_variables):
            self.prior_params[i] = [param.numpy().item()]
            self.m_step_data[i] = [param.numpy().item()]

    def _store_prior_param_vals(self):
        """Update the list storing the prior sde parameter values"""
        for i, param in enumerate(self.tvgp_model.prior_sde.trainable_variables):
            self.prior_params[i].append(param.numpy().item())

    def calculate_nlpd(self) -> float:
        """
            Calculate NLPD on the test set
        """
        if self.test_data is None:
            return 0.

        if not isinstance(self.tvgp_model.likelihood, Gaussian):
            raise Exception("NLPD for non-Gaussian likelihood is not supported!")

        m, S = self.tvgp_model.dist_q.marginals
        s_std = tf.linalg.cholesky(S + self.tvgp_model.likelihood.variance)

        pred_idx = list((tf.where(self.tvgp_model.grid == self.test_data[0][..., None])[:, 1]).numpy())
        s_std = tf.reshape(tf.gather(s_std, pred_idx, axis=1), (-1, 1, 1))
        lpd = gaussian_log_predictive_density(mean=tf.gather(m, pred_idx, axis=1), chol_covariance=s_std,
                                              x=tf.reshape(self.test_data[1], (-1,)))
        nlpd = -1 * tf.reduce_mean(lpd)

        return nlpd.numpy().item()

    def single_e_step(self, max_itr: int = 500) -> list:
        """
        Perform inference.
        """
        elbo_vals = []
        i = 0
        while i < max_itr:
            elbo_before = self.tvgp_model.classic_elbo().numpy().item()

            self.tvgp_model.update_sites(update_all_sites=self.update_all_sites)

            # print(self.tvgp_model.data_sites.nat1.numpy()[0])
            # print(self.tvgp_model.observations[0] / self.tvgp_model.likelihood.variance)

            elbo_vals.append(self.tvgp_model.classic_elbo().numpy().item())

            print(f"t-VGP: ELBO {elbo_vals[-1]}!!!")
            wandb.log({"t-VGP-ELBO": elbo_vals[-1]})
            wandb.log({"t-VGP-NLPD": self.calculate_nlpd()})

            lin_converged = False
            max_lin_itr = 20
            j = 0
            while not lin_converged:
                lin_before_elbo = self.tvgp_model.classic_elbo().numpy().item()
                print(f"t-VGP: ELBO before linearization {lin_before_elbo}!!!")

                self.tvgp_model.fx_mus, self.tvgp_model.fx_covs = self.tvgp_model.dist_q.marginals

                self.tvgp_model.linearization_pnts = (tf.identity(self.tvgp_model.fx_mus[:, :-1, :]),
                                                      tf.identity(self.tvgp_model.fx_covs[:, :-1, :, :]))
                self.tvgp_model._linearize_prior()

                lin_after_elbo = self.tvgp_model.classic_elbo().numpy().item()
                print(f"t-VGP: ELBO after linearization {lin_after_elbo}!!!")

                if lin_before_elbo > lin_after_elbo:
                    break

                if tf.math.abs(lin_before_elbo - lin_after_elbo) < 1e-4:
                    lin_converged = True

                j = j + 1

                if j == max_lin_itr:
                    break
                
            elbo_after = self.tvgp_model.classic_elbo().numpy().item()
            if tf.math.abs(elbo_before - elbo_after) < 1e-4:
                break

            if elbo_before > elbo_after:
                print("ELBO decreasing! Decaying lr!")
                self.tvgp_model.all_sites_lr = self.tvgp_model.all_sites_lr / 2

            i = i + 1

        if i == max_itr:
            print("t-VGP: maximum iterations reached!!!")

        wandb.log({"t-VGP-E-Step": elbo_vals[-1]})

        print("t-VGP: Sites Converged!!!")

        return elbo_vals

    def update_prior_sde(self, max_itr=50):
        elbo_vals = []

        def dist_p() -> StateSpaceModel:
            fx_mus = self.tvgp_model.fx_mus[:, :-1, :]
            fx_covs = self.tvgp_model.fx_covs[:, :-1, :, :]

            return linearize_sde(sde=self.tvgp_model.prior_sde, transition_times=self.tvgp_model.time_points,
                                 q_mean=fx_mus, q_covar=fx_covs, initial_mean=self.tvgp_model.initial_mean,
                                 initial_chol_covariance=self.tvgp_model.initial_chol_cov,
                                 )

        def loss():
            return self.tvgp_model.kl(dist_p=dist_p()) + self.tvgp_model.loss_lin(dist_p=dist_p())
            # return -1. * self.posterior_kalman.log_likelihood() + self.loss_lin()

        i = 0
        while i < max_itr:
            elbo_before = self.tvgp_model.classic_elbo().numpy().item()
            self.prior_sde_optimizer.minimize(loss, self.tvgp_model.prior_sde.trainable_variables)

            # FIXME: ONLY FOR OU: Steady state covariance
            if isinstance(self.tvgp_model.prior_sde, PriorOUSDE):
                self.tvgp_model.initial_chol_cov = tf.linalg.cholesky(
                    (self.tvgp_model.prior_sde.q / (2 * (-1 * self.tvgp_model.prior_sde.decay))) * tf.ones_like(self.tvgp_model.initial_chol_cov))

            # Linearize the prior
            self.linearization_pnts = (tf.identity(self.tvgp_model.fx_mus[:, :-1, :]),
                                       tf.identity(self.tvgp_model.fx_covs[:, :-1, :, :]))
            self.tvgp_model._linearize_prior()

            elbo_after = self.tvgp_model.classic_elbo().numpy().item()
            elbo_vals.append(elbo_after)
            self._store_prior_param_vals()

            # Plotting
            for k in self.prior_params.keys():
                v = self.prior_params[k][-1]
                wandb.log({"t-VGP-learning-" + str(k): v})
                print(f"t-VGP-learning-{str(k)} : {v}")

            # Check convergence
            converged = True
            for i, param in enumerate(self.tvgp_model.prior_sde.trainable_variables):
                old_val = self.prior_params[i][-2]
                new_val = self.prior_params[i][-1]

                diff = tf.reduce_sum(tf.math.abs(old_val - new_val))

                if diff < 1e-4:
                    converged = converged & True
                else:
                    converged = False

            if converged:
                print("t-VGP: Learning; ELBO converged!!!")
                break

            if elbo_before > elbo_after:
                print("t-VGP: Learning; ELBO decreasing. Breaking the loop")
                break

            i = i + 1

        return elbo_vals

    def inference_and_learning(self, max_itr: int = 50, convergence_tol: float = 1e-4):
        """Perform inference and learning"""
        elbo_vals = []
        i = 0

        while i < max_itr:
            elbo_before = self.tvgp_model.classic_elbo().numpy().item()
            inference_elbo = self.single_e_step()
            elbo_vals = elbo_vals + inference_elbo

            learn_elbo_vals = self.update_prior_sde()
            elbo_vals = elbo_vals + learn_elbo_vals

            elbo_vals.append(self.tvgp_model.classic_elbo().numpy().item())
            print(f"t-VGP: Prior SDE (learnt and) re-linearized: ELBO {elbo_vals[-1]};!!!")
            wandb.log({"t-VGP-ELBO": elbo_vals[-1]})

            for k in self.prior_params.keys():
                v = self.prior_params[k][-1]
                self.m_step_data[k].append(v)
                wandb.log({"t-VGP-M-Step-" + str(k): v})

            elbo_after = self.tvgp_model.classic_elbo().numpy().item()

            if elbo_before > elbo_after:
                print("t-VGP: ELBO increasing! Decaying LR!")
                self.prior_sde_optimizer.learning_rate = self.prior_sde_optimizer.learning_rate / 2

            if tf.math.abs(elbo_before - elbo_after) < convergence_tol:
                print("t-VGP: ELBO converged!!!")
                break

            i = i + 1
        return elbo_vals

    def run(self, update_prior: bool = False) -> [list, dict]:
        """
        Run inference and (if required) update prior till convergence.
        """
        self.elbo_vals.append(self.tvgp_model.classic_elbo().numpy().item())
        print(f"t-VGP: Starting ELBO {self.elbo_vals[-1]};")
        wandb.log({"t-VGP-ELBO": self.elbo_vals[-1]})

        if not update_prior:
            inf_elbo_vals = self.single_e_step()
            self.elbo_vals = self.elbo_vals + inf_elbo_vals

            return self.elbo_vals, self.prior_params, self.m_step_data

        else:
            learning_elbo_vals = self.inference_and_learning()
            self.elbo_vals = self.elbo_vals + learning_elbo_vals

            # One last site update for the updated prior
            print("Performing last update sites!!!")
            sites_converged = False
            while not sites_converged:
                sites_converged = self.tvgp_model.update_sites()
                self.elbo_vals.append(self.tvgp_model.classic_elbo().numpy().item())

                print(f"t-VGP: ELBO {self.elbo_vals[-1]};!!!")
                wandb.log({"t-VGP-ELBO": self.elbo_vals[-1]})
                wandb.log({"t-VGP-E-Step": self.elbo_vals[-1]})

        return self.elbo_vals, self.prior_params, self.m_step_data
