"""
This script compares the outputs of functions of VGP model and Stratis implementation's output.
"""

import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Gaussian
from gpflow.config import default_float

from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from sde_as_gp import forward, backward
from markovflow.models.vi_sde import VariationalMarkovGP
from docs.sde.sde_exp_utils import generate_ou_data

DTYPE = default_float()


def forward_test(vgp_model: VariationalMarkovGP):
    m_vgp, S_vgp = vgp_model.forward_pass

    m = np.zeros_like(m_vgp)
    S = np.zeros_like(S_vgp)
    m[0] = m0
    S[0] = S0
    A = vgp_model.A.numpy()
    b = vgp_model.b.numpy()
    Sigma = vgp_model.prior_sde.q

    m, S = forward(m, S, b, A, Sigma, dt)

    np.testing.assert_array_almost_equal(m, m_vgp)
    np.testing.assert_array_almost_equal(S, S_vgp)


def update_params_test(vgp_model: VariationalMarkovGP):
    A = vgp_model.A.numpy()
    b = vgp_model.b.numpy()

    m_vgp, S_vgp = vgp_model.forward_pass
    vgp_model.update_lagrange(m_vgp, S_vgp)
    vgp_model.update_param(m_vgp, S_vgp)

    vgp_A = vgp_model.A
    vgp_b = vgp_model.b

    m = np.zeros_like(m_vgp)
    S = np.zeros_like(S_vgp)
    m[0] = m0
    S[0] = S0
    Sigma = vgp_model.prior_sde.q
    m, S = forward(m, S, b, A, Sigma, dt)

    # Dictionary mapping from times to indices for array x
    t_obs = list(observation_grid.numpy().reshape(-1))
    t_dict = dict(zip(t_obs, np.arange(0, len(t_obs))))
    x = list(observation_vals.numpy().reshape(-1))

    lamda = np.zeros((t_grid.shape[0], 1))
    psi = np.zeros((t_grid.shape[0], 1, 1))
    psi, lamda, b_, A_ = backward(t_grid, A, b, m, S, Sigma, decay, noise_std, psi, lamda, t_dict, x, dt)
    lr = 0.5
    A = (1 - lr) * A + lr * A_
    b = (1 - lr) * b + lr * b_

    np.testing.assert_array_almost_equal(vgp_A.numpy().reshape(-1), A.reshape(-1))
    np.testing.assert_array_almost_equal(vgp_b.numpy().reshape(-1), b.reshape(-1))


def lagrange_multipler_test(vgp_model: VariationalMarkovGP):
    m_vgp, S_vgp = vgp_model.forward_pass
    vgp_model.update_lagrange(m_vgp, S_vgp)

    vgp_psi = vgp_model.psi_lagrange
    vgp_lambda = vgp_model.lambda_lagrange

    m = np.zeros_like(m_vgp)
    S = np.zeros_like(S_vgp)
    m[0] = m0
    S[0] = S0
    A = vgp_model.A.numpy()
    b = vgp_model.b.numpy()
    Sigma = vgp_model.prior_sde.q
    m, S = forward(m, S, b, A, Sigma, dt)

    # Dictionary mapping from times to indices for array x
    t_obs = list(observation_grid.numpy().reshape(-1))
    t_dict = dict(zip(t_obs, np.arange(0, len(t_obs))))
    x = list(observation_vals.numpy().reshape(-1))

    lamda = np.zeros((t_grid.shape[0], 1))
    psi = np.zeros((t_grid.shape[0], 1, 1))
    psi, lamda, b_, A_ = backward(t_grid, A, b, m, S, Sigma, decay, noise_std, psi, lamda, t_dict, x, dt)

    np.testing.assert_array_almost_equal(vgp_psi.numpy().reshape(-1), psi.reshape(-1))
    np.testing.assert_array_almost_equal(vgp_lambda.numpy().reshape(-1), lamda.reshape(-1))


if __name__ == '__main__':

    decay = 1.5
    q = .8
    t0 = 0
    t1 = 5
    dt = 0.01
    m0 = 2.
    S0 = 1e-2

    noise_std = np.sqrt(0.5)
    likelihood = Gaussian(variance=noise_std**2)

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

    vgp_model = VariationalMarkovGP(input_data=input_data, prior_sde=sde, grid=tf.constant(t_grid),
                                    likelihood=likelihood,
                                    lr=0.5)

    vgp_model.p_initial_cov = tf.reshape(tf.constant(S0, dtype=observation_vals.dtype), vgp_model.p_initial_cov.shape)
    vgp_model.q_initial_cov = tf.identity(vgp_model.p_initial_cov)
    vgp_model.p_initial_mean = tf.reshape(tf.constant(m0, dtype=observation_vals.dtype), vgp_model.p_initial_mean.shape)
    vgp_model.q_initial_mean = tf.identity(vgp_model.p_initial_mean)
    vgp_model.A = tf.convert_to_tensor(np.random.randn(vgp_model.A.shape[0]).reshape(vgp_model.A.shape))

    """Testing"""
    forward_test(vgp_model)
    lagrange_multipler_test(vgp_model)
    update_params_test(vgp_model)
