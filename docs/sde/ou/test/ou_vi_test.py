"""
Test function of VGP: Archambeau's method for OU process vs closed form updates.
"""
import os

import numpy as np
import tensorflow as tf
from gpflow.likelihoods import Gaussian
from gpflow.config import default_float

from markovflow.models.vi_sde import VariationalMarkovGP
from markovflow.sde.sde_utils import KL_sde
from markovflow.sde.sde import OrnsteinUhlenbeckSDE


DTYPE = default_float()


def forward_pass_expected(vgp_model: VariationalMarkovGP):
    m = np.zeros_like(vgp_model.b)
    S = np.zeros_like(vgp_model.A)

    m[0] = vgp_model.q_initial_mean
    S[0] = vgp_model.q_initial_cov
    dt = vgp_model.dt
    A = -vgp_model.A
    b = vgp_model.b
    q = vgp_model.prior_sde.q

    for i, t in enumerate(vgp_model.grid[1:]):
        dmdt = A[i] * m[i] + b[i]
        dSdt = A[i] * S[i] + S[i] * A[i] + q
        m[i+1] = m[i] + dt * dmdt
        S[i+1] = S[i] + dt * dSdt

    return m, S


def E_sde_closed_form(vgp_model: VariationalMarkovGP):
    A = vgp_model.A
    b = vgp_model.b
    sde = vgp_model.prior_sde
    dt = vgp_model.dt
    m, S = vgp_model.forward_pass

    A = tf.squeeze(A, axis=-1)
    S = tf.squeeze(S, axis=-1)
    expected_e_sde = tf.square(A - sde.decay) * (tf.square(m) + S) + tf.square(b) - 2 * (A - sde.decay) * b * m
    expected_e_sde = 0.5 * tf.reduce_sum(expected_e_sde) / q
    expected_e_sde = expected_e_sde.numpy() * dt

    return expected_e_sde


def grad_esde_closed_form(vgp_model: VariationalMarkovGP):
    A = vgp_model.A
    b = vgp_model.b
    sde = vgp_model.prior_sde
    m, _ = vgp_model.forward_pass

    A = tf.squeeze(A, axis=-1)
    expected_dEdm = (tf.square(A - sde.decay) * m - (A - sde.decay) * b) / q
    expected_dEdS = 0.5 * (tf.square(A - sde.decay)) / q

    return expected_dEdm, expected_dEdS


def jump_condition_closed_form(vgp_model: VariationalMarkovGP, observation_variance: float, observations: tf.Tensor,
                               observation_grid: tf.Tensor):
    m, _ = vgp_model.forward_pass
    indices = tf.where(tf.equal(vgp_model.grid[..., None], observation_grid))[:, 0]
    expected_d_obs_S = - 0.5 / observation_variance
    m_obs = tf.gather(m, indices)
    expected_d_obs_m = (observations - m_obs) / observation_variance

    expected_d_obs_S = np.repeat(expected_d_obs_S, observations.shape[0], axis=0).reshape(-1, 1)

    # Bigger grid
    indices = tf.where(tf.equal(vgp_model.grid[..., None], observation_grid))[:, 0][..., None]
    expected_d_obs_m = tf.scatter_nd(indices, expected_d_obs_m, vgp_model.lambda_lagrange.shape)
    expected_d_obs_S = tf.scatter_nd(indices, expected_d_obs_S, vgp_model.psi_lagrange.shape)

    return expected_d_obs_m, expected_d_obs_S


def forward_pass_test(vgp_model: VariationalMarkovGP):
    m_vgp, S_vgp = vgp_model.forward_pass
    m_expected, S_expected = forward_pass_expected(vgp_model)

    np.testing.assert_array_almost_equal(m_vgp, m_expected)
    np.testing.assert_array_almost_equal(S_vgp, S_expected)


def E_sde_test(vgp_model: VariationalMarkovGP):
    A = vgp_model.A
    b = vgp_model.b
    sde = vgp_model.prior_sde
    dt = vgp_model.dt
    m, S = vgp_model.forward_pass

    E_sde = KL_sde(sde, A, b, m, S, dt)

    expected_e_sde = E_sde_closed_form(vgp_model)
    np.testing.assert_array_almost_equal(E_sde, expected_e_sde)


def grad_e_sde_test(vgp_model: VariationalMarkovGP):
    m, S = vgp_model.forward_pass
    dEdm, dEdS = vgp_model.grad_E_sde(m, S)

    expected_dEdm, expected_dEdS = grad_esde_closed_form(vgp_model)

    np.testing.assert_array_almost_equal(dEdm, expected_dEdm)
    np.testing.assert_array_almost_equal(dEdS, expected_dEdS)


def jump_condition_test(vgp_model: VariationalMarkovGP, observation_variance: float, observations: tf.Tensor,
                        observation_grid: tf.Tensor):
    m, S = vgp_model.forward_pass
    d_obs_m, d_obs_S = vgp_model._jump_conditions(m, S)

    expected_d_obs_m, expected_d_obs_S = jump_condition_closed_form(vgp_model, observation_variance, observations,
                                                                    observation_grid)

    np.testing.assert_array_almost_equal(d_obs_S, expected_d_obs_S)
    np.testing.assert_array_almost_equal(d_obs_m, expected_d_obs_m)


def update_lambda_lagrange_test(vgp_model: VariationalMarkovGP, observation_variance: float, observations: tf.Tensor,
                                observation_grid: tf.Tensor):
    m, S = vgp_model.forward_pass
    vgp_model.update_lagrange(m, S)
    model_lambda_lagrange = vgp_model.lambda_lagrange

    A = tf.squeeze(vgp_model.A, axis=-1)
    N = vgp_model.N
    dt = vgp_model.dt

    dE_sde_dm, _ = grad_esde_closed_form(vgp_model)
    d_obs_m, _ = jump_condition_closed_form(vgp_model, observation_variance, observations, observation_grid)
    observation_idx = tf.where(tf.equal(vgp_model.grid[..., None], observation_grid))[:, 0]

    expected_lambda_lagrange = np.zeros_like(model_lambda_lagrange)
    for i in range(N-1, 0, -1):
        expected_lambda_lagrange[i-1] = expected_lambda_lagrange[i] + (dE_sde_dm[i] - A[i] *
                                                                       expected_lambda_lagrange[i]) * dt
        if (i-1) in observation_idx.numpy():
            expected_lambda_lagrange[i-1] = expected_lambda_lagrange[i-1] - d_obs_m[i-1]

    np.testing.assert_array_almost_equal(model_lambda_lagrange, expected_lambda_lagrange)


def update_psi_lagrange_test(vgp_model: VariationalMarkovGP, observation_variance: float, observations: tf.Tensor,
                             observation_grid: tf.Tensor):
    m, S = vgp_model.forward_pass
    vgp_model.update_lagrange(m, S)
    model_psi_lagrange = vgp_model.psi_lagrange

    A = tf.squeeze(vgp_model.A, axis=-1)
    N = vgp_model.N
    dt = vgp_model.dt

    _, dE_sde_dS = grad_esde_closed_form(vgp_model)
    _, d_obs_S = jump_condition_closed_form(vgp_model, observation_variance, observations, observation_grid)
    observation_idx = tf.where(tf.equal(vgp_model.grid[..., None], observation_grid))[:, 0]

    expected_psi_lagrange = np.zeros_like(model_psi_lagrange)

    for i in range(N-1, 0, -1):
        expected_psi_lagrange[i-1] = expected_psi_lagrange[i] + (dE_sde_dS[i] - 2 * A[i] *
                                                                 expected_psi_lagrange[i]) * dt
        if (i-1) in observation_idx.numpy():
            expected_psi_lagrange[i-1] = expected_psi_lagrange[i-1] - d_obs_S[i-1]

    np.testing.assert_array_almost_equal(model_psi_lagrange, expected_psi_lagrange)


if __name__ == '__main__':
    data_dir = "../data/786"

    data_path = os.path.join(data_dir, "data.npz")
    data = np.load(data_path)
    decay = data["decay"]
    q = data["q"]
    noise_stddev = data["noise_stddev"]
    x0 = data["x0"]
    observation_vals = data["observation_vals"]
    observation_grid = data["observation_grid"]
    latent_process = data["latent_process"]
    time_grid = data["time_grid"]
    t0 = time_grid[0]
    t1 = time_grid[-1]

    input_data = (observation_grid, tf.constant(tf.squeeze(observation_vals, axis=0)))
    observation_vals = observation_vals.reshape((-1, 1))
    likelihood = Gaussian(variance=noise_stddev**2)
    prior_sde = OrnsteinUhlenbeckSDE(decay=decay * tf.ones((1, 1), dtype=DTYPE), q=q * tf.ones((1, 1), dtype=DTYPE))

    vgp_model = VariationalMarkovGP(input_data=input_data,
                                    prior_sde=prior_sde, grid=time_grid, likelihood=likelihood,
                                    lr=0.001, prior_params_lr=0.01)
    vgp_model.A = tf.convert_to_tensor(np.random.normal(5, 1, size=vgp_model.A.shape))
    vgp_model.b = tf.convert_to_tensor(np.random.normal(0, 1, size=vgp_model.b.shape))

    forward_pass_test(vgp_model)

    E_sde_test(vgp_model)

    grad_e_sde_test(vgp_model)

    jump_condition_test(vgp_model, noise_stddev**2, observation_vals, observation_grid)

    update_lambda_lagrange_test(vgp_model, noise_stddev**2, observation_vals, observation_grid)

    update_psi_lagrange_test(vgp_model, noise_stddev**2, observation_vals, observation_grid)
