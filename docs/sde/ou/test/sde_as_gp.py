"""
Stratis implementation of the VGP (Archambeau's method.)
    https://random-walks.org/content/misc/sde-as-gp/sde-as-gp.html
"""


import numpy as np
import matplotlib.pyplot as plt
import gpflow

from markovflow.models.vi_sde import VariationalMarkovGP
import tensorflow as tf
from gpflow.likelihoods import Gaussian
from markovflow.sde.sde import OrnsteinUhlenbeckSDE
import sys
sys.path.append("../..")
from sde_exp_utils import plot_posterior, predict_vgp, generate_ou_data

DTYPE = gpflow.config.default_float()


def forward(m, S, b, A, Sigma, dt):
    for i in range(len(b) - 1):
        # Euler step for m and S ODEs
        m[i + 1] = m[i] - (np.dot(A[i], m[i]) - b[i]) * dt
        S[i + 1] = S[i] - (np.dot(A[i], S[i]) + np.dot(S[i], A[i].T) - Sigma) * dt

    return m, S


def backward(t_grid, A, b, m, S, Sigma, gamma, r, psi, lamda, t_dict, x, dt):
    # Arrays for storing the updates for A and b
    A_ = np.zeros_like(A)
    b_ = np.zeros_like(b)

    for i in range(len(b) - 1, 0, -1):

        # Compute dEdS and dEdm
        coeff = (A[i] - gamma) ** 2 / Sigma
        dEdS = 0.5 * coeff
        dEdm = coeff * m[i] - b[i] * (A[i] - gamma) / Sigma

        # Euler step for lambda and psi ODEs
        lamda[i - 1] = lamda[i] - (np.dot(A[i].T, lamda[i]) - dEdm) * dt
        psi[i - 1] = psi[i] - (2 * np.dot(psi[i], A[i]) - dEdS) * dt

        # Handle jump conditions at locations of the data
        if t_grid[i - 1] in t_dict:
            psi[i - 1] = psi[i - 1] + 0.5 / r ** 2
            lamda[i - 1] = lamda[i - 1] - (x[t_dict[t_grid[i - 1]]] - m[i - 1])/r ** 2

    for i in range(len(b) - 1, -1, -1):
        A_[i] = gamma + 2 * np.dot(Sigma, psi[i])
        b_[i] = - gamma * m[i] + np.dot(A_[i], m[i]) - np.dot(Sigma, lamda[i])

    return psi, lamda, b_, A_


def smoothing(t_obs, t_grid, y_obs, num_passes, omega, Sigma, gamma, r, dt, m0, S0):
    grid_size = t_grid.shape[0]

    # Dictionary mapping from times to indices for array x
    t_dict = dict(zip(t_obs, np.arange(0, len(t_obs))))

    b = np.zeros((grid_size, 1))
    A = np.zeros((grid_size, 1, 1))

    for i in range(num_passes):
        lamda = np.zeros((grid_size, 1))
        psi = np.zeros((grid_size, 1, 1))

        m = m0 * np.ones((grid_size, 1))
        S = S0 * np.ones((grid_size, 1, 1))

        # Forward pass to compute m, S
        m, S = forward(m=m, S=S, b=b, A=A, Sigma=Sigma, dt=dt)

        # Backward pass to compute psi, lamda, b_, A_
        psi, lamda, b_, A_ = backward(t_grid=t_grid,
                                      A=A,
                                      b=b,
                                      m=m,
                                      S=S,
                                      Sigma=Sigma,
                                      gamma=gamma,
                                      r=r,
                                      psi=psi,
                                      lamda=lamda,
                                      t_dict=t_dict,
                                      x=y_obs,
                                      dt=dt)

        b = b + omega * (b_ - b)
        A = A + omega * (A_ - A)

    return b, A, m, S, psi, lamda


def ornstein_uhlenbeck(sigma, gamma, t, t_):
    coeff = 0.5 * sigma ** 2 / gamma
    exp = np.exp(- gamma * np.abs(t[..., :, None] - t_[..., None, :]))

    return coeff * exp


if __name__ == '__main__':
    decay = 1.5
    q = .8
    t0 = 0
    t1 = 5
    dt = 0.01
    m0 = 2.
    S0 = 1e-2
    num_passes = 50
    lr = 0.5
    noise_var = 0.5
    noise_std = np.sqrt(noise_var)

    likelihood = Gaussian(variance=noise_std ** 2)

    sde = OrnsteinUhlenbeckSDE(decay=tf.reshape(tf.constant(decay, dtype=DTYPE), (1, 1)),
                               q=tf.reshape(tf.constant(q, dtype=DTYPE), (1, 1)))

    t_grid = np.arange(t0, t1, dt)

    observation_vals, observation_grid, latent_process, time_grid = generate_ou_data(decay=decay, q=q, x0=m0, t0=t0,
                                                                                     t1=t1,
                                                                                     simulation_dt=dt,
                                                                                     noise_stddev=noise_std,
                                                                                     n_observations=10,
                                                                                     dtype=DTYPE)

    input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))

    # Run the smoothing algorithm
    b, A, m, S, psi, lamda = smoothing(t_obs=observation_grid.numpy().reshape(-1),
                                       t_grid=t_grid,
                                       y_obs=observation_vals.numpy().reshape(-1),
                                       num_passes=num_passes,
                                       omega=lr,
                                       Sigma=q,
                                       gamma=decay,
                                       r=noise_std,
                                       dt=dt,
                                       m0=m0,
                                       S0=S0)

    m, S = forward(m=m, S=S, b=b, A=A, Sigma=q, dt=dt)

    # Plot data, approximate and exact posterior
    plt.figure(figsize=(8, 3))

    # Observed data
    plt.scatter(observation_grid, observation_vals, marker='x', color='red', zorder=3)

    # Approximate posterior
    plt.plot(t_grid, m[:, 0], color='k', zorder=2)
    plt.fill_between(t_grid,
                     m[:, 0] - 2 * S[:, 0, 0] ** 0.5,
                     m[:, 0] + 2 * S[:, 0, 0] ** 0.5,
                     color='gray',
                     alpha=0.5,
                     zorder=1,
                     label='Approximate posterior')

    """
    VGP
    """
    prior_sde = OrnsteinUhlenbeckSDE(decay=tf.constant(decay, shape=(1, 1), dtype=observation_vals.dtype),
                                     q=tf.constant(q, shape=(1, 1), dtype=observation_vals.dtype))

    steady_cov = S0  # 0.5 * sigma ** 2 / gamma

    vgp_model = VariationalMarkovGP(input_data=input_data, prior_sde=prior_sde, grid=tf.constant(t_grid),
                                    likelihood=likelihood,
                                    lr=0.5)

    vgp_model.p_initial_cov = tf.reshape(tf.constant(steady_cov, dtype=observation_vals.dtype),
                                         vgp_model.p_initial_cov.shape)
    vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
    vgp_model.p_initial_mean = tf.reshape(tf.constant(m0, dtype=observation_vals.dtype), vgp_model.p_initial_mean.shape)
    vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)
    vgp_model.A = 0. * vgp_model.A
    vgp_model.b = 0. * vgp_model.b
    vgp_model.lambda_lagrange = 0. * vgp_model.lambda_lagrange
    vgp_model.psi_lagrange = 0. * vgp_model.psi_lagrange

    # Perform steps
    for _ in range(num_passes):
        vgp_model.run_single_inference()

    m_vgp, S_std_vgp = predict_vgp(vgp_model, 0)
    plot_posterior(m_vgp, S_std_vgp, t_grid, "VGP")

    # Format plot
    plt.title('Exact and approximate posterior', fontsize=18)
    plt.xlabel('$t$', fontsize=16)
    plt.ylabel('$x$', fontsize=16)
    plt.xlim([t0, t1])
    plt.legend()
    plt.show()

    """VGP==VGP-Stratis"""

    np.testing.assert_array_almost_equal(lamda.reshape(-1), vgp_model.lambda_lagrange.numpy().reshape(-1))
    np.testing.assert_array_almost_equal(psi.reshape(-1), vgp_model.psi_lagrange.numpy().reshape(-1))

    np.testing.assert_array_almost_equal(m_vgp, m.reshape(-1))
    np.testing.assert_array_almost_equal(S_std_vgp, S.reshape(-1)**0.5)
