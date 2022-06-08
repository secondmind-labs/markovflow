"""
Script for common utility functions needed for SDE experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Likelihood
from gpflow.utilities.bijectors import positive

from markovflow.sde.sde import OrnsteinUhlenbeckSDE, DoubleWellSDE
from markovflow.sde.sde_utils import euler_maruyama
from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.models.variational_cvi import CVIGaussianProcess, CVIGaussianProcessTaylorKernel
from markovflow.kernels import SDEKernel


def generate_ou_data(decay: float, q: float, x0: float, t0: float, t1: float, simulation_dt: float,
                     noise_stddev: np.ndarray, n_observations: int, dtype: tf.DType = tf.float64,
                     state_dim: int = 1, num_batch: int = 1):

    decay = decay * tf.ones((1, 1), dtype=dtype)
    q = q * tf.ones((1, 1), dtype=dtype)
    sde = OrnsteinUhlenbeckSDE(decay=decay, q=q)

    x0_shape = (num_batch, state_dim)
    x0 = x0 + tf.zeros(x0_shape, dtype=dtype)

    time_grid = tf.cast(np.arange(t0, t1+simulation_dt, simulation_dt), dtype=dtype)

    # Observation at every even place

    # Don't observe on 10% of both the time ends
    N = time_grid.shape[0]
    l1 = int(N * 0.05)
    l2 = int(N * 0.95)
    observation_idx = list(tf.cast(np.linspace(l1, l2, n_observations), dtype=tf.int32))
    observation_grid = tf.gather(time_grid, observation_idx)

    latent_process = euler_maruyama(sde, x0, time_grid)

    # Pick observations from the latent process
    latent_states = tf.gather(latent_process, observation_idx, axis=1)

    # Adding observation noise
    simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=dtype)

    return simulated_values, observation_grid, latent_process, time_grid


def generate_dw_data(q: float, x0: float, t0: float, t1: float, simulation_dt: float,
                     noise_stddev: np.ndarray, n_observations: int, dtype: tf.DType = tf.float64,
                     state_dim: int = 1, num_batch: int = 1):

    q = q * tf.ones((1, 1), dtype=dtype)
    sde = DoubleWellSDE(q=q)

    x0_shape = (num_batch, state_dim)
    x0 = x0 + tf.zeros(x0_shape, dtype=dtype)

    time_grid = tf.cast(np.arange(t0, t1+simulation_dt, simulation_dt), dtype=dtype)

    # Observation at every even place
    observation_idx = list(tf.cast(np.linspace(2, time_grid.shape[0]-2, n_observations), dtype=tf.int32))
    observation_grid = tf.gather(time_grid, observation_idx)

    latent_process = euler_maruyama(sde, x0, time_grid)

    # Pick observations from the latent process
    latent_states = tf.gather(latent_process, observation_idx, axis=1)

    # Adding observation noise
    simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=dtype)

    return simulated_values, observation_grid, latent_process, time_grid, sde


def plot_posterior(m: np.ndarray, S_std: np.ndarray, time_grid: np.ndarray, model_type):
    """
    Plot posterior, this function doesn't do show(), label setting, etc.
    """
    if model_type == "GPR":
        plt.plot(time_grid.reshape(-1), m.reshape(-1), linestyle='dashed', color="black")
        plt.fill_between(
            time_grid,
            y1=(m.reshape(-1) - 2 * S_std.reshape(-1)).reshape(-1, ),
            y2=(m.reshape(-1) + 2 * S_std.reshape(-1)).reshape(-1, ),
            # alpha=1.,
            label=model_type,
            edgecolor="black",
            facecolor=(0, 0, 0, 0.),
            linestyle='dashed'
        )

    elif model_type == "VGP":
        alpha = 0.2
        plt.plot(time_grid.reshape(-1), m.reshape(-1), linestyle='dotted')
        plt.fill_between(
            time_grid,
            y1=(m.reshape(-1) - 2 * S_std.reshape(-1)).reshape(-1, ),
            y2=(m.reshape(-1) + 2 * S_std.reshape(-1)).reshape(-1, ),
            alpha=alpha,
            label=model_type,
            linestyle='dashed'
        )

    elif model_type == "SDE-SSM":
        alpha = 0.2
        plt.plot(time_grid.reshape(-1), m.reshape(-1), linestyle='-.')
        plt.fill_between(
            time_grid,
            y1=(m.reshape(-1) - 2 * S_std.reshape(-1)).reshape(-1, ),
            y2=(m.reshape(-1) + 2 * S_std.reshape(-1)).reshape(-1, ),
            alpha=alpha,
            label=model_type,
        )


def plot_observations(observation_t: np.ndarray, observation_y: np.ndarray):
    """
    Plot observations; this function doesn't do show(), label setting, etc.
    """
    plt.plot(observation_t.reshape(-1), observation_y.reshape(-1), 'kx', ms=8, mew=2, label="Observations (Y)")


def predict_vgp(model: VariationalMarkovGP, noise_stddev: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Predict mean and std-dev for VGP model.
    """
    m, S = model.forward_pass
    S = tf.reshape(S, (-1)).numpy()

    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S) + noise_stddev

    return m, S_std


def predict_ssm(model: SDESSM, noise_stddev: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Predict mean and std-dev for SSM model.
    """
    posterior_ssm = model.posterior_kalman.posterior_state_space_model()
    m, S = posterior_ssm.marginal_means, posterior_ssm.marginal_covariances

    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S) + noise_stddev

    return m, S_std


def predict_cvi_gpr(model: CVIGaussianProcess, time_grid: np.ndarray, noise_stddev: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for CVI-GPR model.
    """
    m, S = model.predict_f(time_grid)
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S) + noise_stddev

    return m, S_std


def predict_cvi_gpr_taylor(model: CVIGaussianProcessTaylorKernel, noise_stddev: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for CVI-GPR model.
    """
    m, S = model.dist_q.marginals
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S) + noise_stddev

    return m, S_std


def predict_gpr(model: GaussianProcessRegression, time_grid: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for GPR model.
    """
    m, S = model.posterior.predict_y(time_grid)
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S)

    return m, S_std


def get_cvi_gpr(input_data: [tf.Tensor, tf.Tensor], kernel: SDEKernel, likelihood: Likelihood,
                train: bool = False, sites_lr: float = 0.9):
    cvi_model = CVIGaussianProcess(input_data=input_data, kernel=kernel, likelihood=likelihood,
                                   learning_rate=sites_lr)

    opt = tf.optimizers.Adam(lr=0.1)

    prior_params = {}
    for i, param in enumerate(kernel.trainable_variables):
        prior_params[i] = [positive().forward(param).numpy().item()]

    @tf.function
    def opt_step():
        opt.minimize(cvi_model.loss, cvi_model.kernel.trainable_variables)

    elbo_vals = [cvi_model.classic_elbo()]
    while len(elbo_vals) < 10 or elbo_vals[-2] - elbo_vals[-1] > 1e-2:
        for _ in range(2):
            cvi_model.update_sites()

        if train:
            for _ in range(5):
                opt_step()

            for i, param in enumerate(kernel.trainable_variables):
                prior_params[i].append(positive().forward(param).numpy().item())

        elbo_vals.append(cvi_model.classic_elbo())

    return cvi_model, prior_params


def get_cvi_gpr_taylor(input_data: [tf.Tensor, tf.Tensor], kernel: SDEKernel, time_grid: tf.Tensor,
                       likelihood: Likelihood, train: bool = False, sites_lr: float = 0.9):
    cvi_model = CVIGaussianProcessTaylorKernel(input_data=input_data, kernel=kernel, likelihood=likelihood,
                                               learning_rate=sites_lr, time_grid=time_grid)

    if train:
        raise Exception("Currently training isn't supported!")

    prior_params = {}
    for i, param in enumerate(kernel.trainable_variables):
        prior_params[i] = [positive().forward(param).numpy().item()]

    elbo_vals = [cvi_model.classic_elbo()]
    while len(elbo_vals) < 5 or elbo_vals[-2] - elbo_vals[-1] > 1e-2:
        for _ in range(1):
            cvi_model.update_sites()
            elbo_vals.append(cvi_model.classic_elbo())

    return cvi_model, prior_params, elbo_vals


def get_gpr(input_data: [tf.Tensor, tf.Tensor], kernel: SDEKernel, train: bool, noise_stddev: np.ndarray,
            iterations: int = 50, state_dim: int = 1) -> GaussianProcessRegression:
    """
    Train a GPR model.
    """
    gpr_model = GaussianProcessRegression(input_data=input_data,
                                          kernel=kernel,
                                          chol_obs_covariance=noise_stddev * tf.eye(state_dim,
                                                                                    dtype=input_data[1].dtype)
                                         )
    if train:
        opt = tf.optimizers.Adam()

        @tf.function
        def opt_step():
            opt.minimize(gpr_model.loss, gpr_model.trainable_variables)

        for _ in range(iterations):
            opt_step()

    return gpr_model
