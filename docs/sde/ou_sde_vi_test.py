import tensorflow as tf
import numpy as np

from gpflow.likelihoods import Gaussian
from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from markovflow.models.vi_sde import VariationalMarkovGP


def forward_pass(model: VariationalMarkovGP):
    model_m, model_S = model.forward_pass
    dt = model.dt
    m = [model.initial_mean.numpy().item()]
    S = [model.initial_cov.numpy().item()]
    for i in range(model.N):
        m.append((m[i] + dt * (-model.A[i] * m[i] + model.b[i])).numpy().item())
        S.append((S[i] + dt * (-model.A[i] * S[i] - S[i] * model.A[i] + model.prior_sde.q)).numpy().item())

    m = np.array(m).reshape((-1, 1))
    S = np.array(S).reshape((-1, 1, 1))

    np.testing.assert_array_equal(model_m, m)
    np.testing.assert_array_almost_equal(model_S, S, 2)


def e_sde(model: VariationalMarkovGP):
    m, S = model.forward_pass
    e_sde = tf.reduce_sum(variational_gp.E_sde(m, S)).numpy().item()
    q = model.prior_sde.q

    A = tf.squeeze(model.A, axis=-1)
    S = tf.squeeze(S, axis=-1)
    expected_e_sde = tf.square(A - model.prior_sde.decay) * (tf.square(m) + S) + tf.square(model.b) - 2 * (A - model.prior_sde.decay) * model.b * m
    expected_e_sde = 0.5 * tf.reduce_sum(expected_e_sde) / q
    expected_e_sde = expected_e_sde.numpy().item()

    np.testing.assert_array_almost_equal(e_sde, expected_e_sde)


def grad_e_sde(model: VariationalMarkovGP):
    m, S = model.forward_pass
    dEdm, dEdS = variational_gp.grad_E_sde(m, S)
    q = model.prior_sde.q

    A = tf.squeeze(model.A, axis=-1)

    expected_dEdm = (tf.square(A - model.prior_sde.decay) * m - (A - model.prior_sde.decay) * model.b) / q
    expected_dEdS = 0.5 * (tf.square(A - model.prior_sde.decay)) / q

    np.testing.assert_array_almost_equal(dEdm, expected_dEdm)
    np.testing.assert_array_almost_equal(tf.squeeze(dEdS, axis=-1), expected_dEdS)


def jump_condition(model: VariationalMarkovGP, observation_variance: float, observations: tf.Tensor,
                   observation_idx: tf.Tensor):
    m, S = model.forward_pass
    d_obs_m, d_obs_S = variational_gp._jump_conditions(m, S)

    expected_d_obs_S = - 0.5 / observation_variance
    m_obs = tf.gather(m, observation_idx)
    expected_d_obs_m = (observations - m_obs) / observation_variance

    np.testing.assert_array_equal(tf.reduce_sum(d_obs_S), tf.reduce_sum(expected_d_obs_S))
    np.testing.assert_array_equal(tf.reduce_sum(d_obs_m), tf.reduce_sum(expected_d_obs_m))


if __name__ == '__main__':

    f = 2. * tf.ones((1, 1), dtype=tf.float64)
    q = .5 * tf.ones((1, 1), dtype=tf.float64)
    ou_sde = OrnsteinUhlenbeckSDE(decay=f, q=q)
    observation_variance = 0.4

    time_grid = tf.linspace(0, 1, 10)
    observation_grid = time_grid[5:6]

    simulated_values = tf.cast(tf.ones_like(observation_grid)[..., None], dtype=tf.float64)
    likelihood = Gaussian(variance=observation_variance)

    variational_gp = VariationalMarkovGP(input_data=(observation_grid,
                                                     tf.constant(simulated_values)),
                                         prior_sde=ou_sde, grid=time_grid, likelihood=likelihood)

    forward_pass(variational_gp)

    e_sde(variational_gp)

    grad_e_sde(variational_gp)

    indices = tf.where(tf.equal(time_grid[..., None], observation_grid))[:, 0]
    jump_condition(variational_gp, observation_variance, simulated_values, indices)
