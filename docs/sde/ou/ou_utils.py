"""
Script for common utility functions needed for OU process experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.gaussian_process_regression import GaussianProcessRegression
from markovflow.kernels import SDEKernel
from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from markovflow.sde.sde_utils import euler_maruyama


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
    S = (tf.reshape(S, (-1)) + noise_stddev ** 2).numpy()

    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S)

    return m, S_std


def predict_ssm(model: SDESSM, noise_stddev: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Predict mean and std-dev for SSM model.
    """
    posterior_ssm = model.posterior_kalman.posterior_state_space_model()
    m, S = posterior_ssm.marginal_means, posterior_ssm.marginal_covariances
    S = S + noise_stddev ** 2

    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S)

    return m, S_std


def predict_gpr(model: GaussianProcessRegression, time_grid: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for GPR model.
    """
    m, S = model.posterior.predict_y(time_grid)
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S)

    return m, S_std


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


def generate_data(decay: float, q: float, x0: float, t0: float, t1: float, simulation_dt: float, noise_stddev: np.ndarray,
                  n_observations: int, dtype: tf.DType=tf.float64, state_dim: int = 1, num_batch: int = 1):

    decay = decay * tf.ones((1, 1), dtype=dtype)
    q = q * tf.ones((1, 1), dtype=dtype)
    sde = OrnsteinUhlenbeckSDE(decay=decay, q=q)

    x0_shape = (num_batch, state_dim)
    x0 = x0 + tf.zeros(x0_shape, dtype=dtype)

    time_grid = tf.cast(np.arange(t0, t1+simulation_dt, simulation_dt), dtype=dtype)

    # Observation at every even place
    observation_idx = list(tf.cast(np.linspace(10, time_grid.shape[0]-2, n_observations), dtype=tf.int32))
    observation_grid = tf.gather(time_grid, observation_idx)

    latent_process = euler_maruyama(sde, x0, time_grid)

    # Pick observations from the latent process
    latent_states = tf.gather(latent_process, observation_idx, axis=1)

    # Adding observation noise
    simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=dtype)

    return simulated_values, observation_grid, latent_process, time_grid

