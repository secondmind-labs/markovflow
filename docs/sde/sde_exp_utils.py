"""
Script for common utility functions needed for SDE experiments.
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Likelihood
from gpflow.utilities.bijectors import positive
from scipy.linalg import solve_continuous_lyapunov

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

    time_grid = tf.cast(np.linspace(t0, t1, int((t1-t0)//simulation_dt) + 2), dtype=dtype)

    # Don't observe on 10% of both the time ends
    N = time_grid.shape[0]
    l1 = int(N * 0.05)
    l2 = int(N * 0.95)
    observation_idx = list(tf.cast(tf.linspace(l1, l2, n_observations), dtype=tf.int32))
    observation_grid = tf.gather(time_grid, observation_idx)

    n_test = int(0.1 * n_observations)
    test_idx = sorted(list(np.random.randint(0, time_grid.shape[0], n_test)))
    if len(test_idx) > 0:
        test_grid = tf.gather(time_grid, test_idx)
    else:
        test_grid = None

    latent_process = euler_maruyama(sde, x0, time_grid)

    # Pick observations from the latent process
    latent_states = tf.gather(latent_process, observation_idx, axis=1)

    # Adding observation noise
    simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=dtype)

    # Pick test observations
    if len(test_idx) > 0:
        test_latent_states = tf.gather(latent_process, test_idx, axis=1)
        test_values = test_latent_states + tf.random.normal(test_latent_states.shape, stddev=noise_stddev, dtype=dtype)
    else:
        test_values = None

    return simulated_values, observation_grid, latent_process, time_grid, test_values, test_grid


def generate_dw_data(q: float, x0: float, t0: float, t1: float, simulation_dt: float,
                     noise_stddev: np.ndarray, n_observations: int, dtype: tf.DType = tf.float64,
                     state_dim: int = 1, num_batch: int = 1):

    q = q * tf.ones((1, 1), dtype=dtype)
    sde = DoubleWellSDE(q=q)

    x0_shape = (num_batch, state_dim)
    x0 = x0 + tf.zeros(x0_shape, dtype=dtype)

    time_grid = tf.cast(np.linspace(t0, t1, int((t1-t0)//simulation_dt) + 2), dtype=dtype)

    # Observation at every even place
    observation_idx = list(tf.cast(np.linspace(2, time_grid.shape[0]-2, n_observations), dtype=tf.int32))
    observation_grid = tf.gather(time_grid, observation_idx)

    n_test = int(0.1 * n_observations)
    test_idx = sorted(list(np.random.randint(0, time_grid.shape[0], n_test)))
    test_grid = tf.gather(time_grid, test_idx)

    latent_process = euler_maruyama(sde, x0, time_grid)

    # Pick observations from the latent process
    latent_states = tf.gather(latent_process, observation_idx, axis=1)

    # Adding observation noise
    simulated_values = latent_states + tf.random.normal(latent_states.shape, stddev=noise_stddev, dtype=dtype)

    # Pick test observations
    test_latent_states = tf.gather(latent_process, test_idx, axis=1)
    test_values = test_latent_states + tf.random.normal(test_latent_states.shape, stddev=noise_stddev, dtype=dtype)

    return simulated_values, observation_grid, latent_process, time_grid, test_values, test_grid, sde


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
    S_std = np.sqrt(S + noise_stddev**2)

    return m, S_std


def predict_ssm(model: SDESSM, noise_stddev: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Predict mean and std-dev for SSM model.
    """
    posterior_ssm = model.posterior_kalman.posterior_state_space_model()
    m, S = posterior_ssm.marginal_means, posterior_ssm.marginal_covariances

    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S + noise_stddev**2)

    return m, S_std


def predict_cvi_gpr(model: CVIGaussianProcess, time_grid: np.ndarray, noise_stddev: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for CVI-GPR model.
    """
    m, S = model.predict_f(time_grid)
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S + noise_stddev**2)

    return m, S_std


def predict_cvi_gpr_taylor(model: CVIGaussianProcessTaylorKernel, noise_stddev: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict mean and std-dev for CVI-GPR model.
    """
    m, S = model.dist_q.marginals
    m = m.numpy().reshape(-1)
    S_std = np.sqrt(S + noise_stddev**2)

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

    opt = tf.optimizers.Adam(lr=0.01)

    prior_params = {}
    for i, param in enumerate(kernel.trainable_variables):
        prior_params[i] = [positive().forward(param).numpy().item()]

    @tf.function
    def opt_step():
        opt.minimize(cvi_model.loss, cvi_model.kernel.trainable_variables)

    elbo_vals = [cvi_model.classic_elbo()]
    while len(elbo_vals) < 10 or elbo_vals[-2] - elbo_vals[-1] > 1e-4:
        for _ in range(2):
            cvi_model.update_sites()

        if train:
            for _ in range(50):
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
        print("CVI-GPR (Taylor) : Not training!!!")
        # raise Exception("Currently training isn't supported!")

    prior_params = {}
    for i, param in enumerate(kernel.trainable_variables):
        prior_params[i] = [positive().forward(param).numpy().item()]

    elbo_vals = [cvi_model.classic_elbo()]
    while len(elbo_vals) < 5 or (elbo_vals[-2] - elbo_vals[-1]) > 1e-4:
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


def calculate_inv(A: tf.Tensor) -> tf.Tensor:
    """Calculate inverse of a matrix using Cholesky."""
    A_chol = tf.linalg.cholesky(A)
    A_inv = tf.linalg.cholesky_solve(A_chol, tf.eye(A.shape[0], dtype=A.dtype))

    return A_inv


def get_steady_state(A: tf.Tensor, b: tf.Tensor, Q: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    """
    Calculate the steady state mean and covariance for the SDE

    Input SDE: d x_t = (-A_t x_t + b_t) dt + dB_t; Q

    """
    if A.shape[0] != A.shape[1]:
       A = -1 * tf.linalg.diag(tf.reshape(A, (-1)))

    A_inv = calculate_inv(A)
    steady_m = A_inv @ b

    Q = tf.linalg.diag(tf.reshape(Q, (-1)))
    steady_P = solve_continuous_lyapunov(A, Q)

    return steady_m, steady_P


def bitmappify(ax, dpi=None):
    """
    Convert vector axes content to raster (bitmap) images
    """
    fig = ax.figure
    # safe plot without axes
    ax.set_axis_off()
    fig.savefig('temp.png', dpi=dpi, transparent=False)
    ax.set_axis_on()

    # remember geometry
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xb = ax.bbox._bbox.corners()[:, 0]
    xb = (min(xb), max(xb))
    yb = ax.bbox._bbox.corners()[:, 1]
    yb = (min(yb), max(yb))

    # compute coordinates to place bitmap image later
    xb = (- xb[0] / (xb[1] - xb[0]), (1 - xb[0]) / (xb[1] - xb[0]))
    xb = (xb[0] * (xl[1] - xl[0]) + xl[0], xb[1] * (xl[1] - xl[0]) + xl[0])
    yb = (- yb[0] / (yb[1] - yb[0]), (1 - yb[0]) / (yb[1] - yb[0]))
    yb = (yb[0] * (yl[1] - yl[0]) + yl[0], yb[1] * (yl[1] - yl[0]) + yl[0])

    # replace the dots by the bitmap
    del ax.collections[:]
    del ax.lines[:]
    ax.imshow(imread('temp.png'), origin='upper',
              aspect='auto', extent=(xb[0], xb[1], yb[0], yb[1]), label='_nolegend_')

    # reset view
    ax.set_xlim(xl)
    ax.set_ylim(yl)
