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
"""Module containing the unit tests for the SDE related functions and classes"""
import pytest

import tensorflow as tf
import numpy as np
import gpflow
from gpflow.probability_distributions import Gaussian

from markovflow.state_space_model import state_space_model_from_covariances
from markovflow.sde.sde_utils import euler_maruyama, linearize_sde
from markovflow.sde import OrnsteinUhlenbeckSDE
from markovflow.sde.sde_utils import squared_drift_difference_along_Gaussian_path
from markovflow.sde.drift import LinearDrift

tf.random.set_seed(33)

DTYPE = gpflow.config.default_float()


@pytest.fixture(
    name="sde_batch_shape", params=[tf.TensorShape([3, ]), tf.TensorShape([2, 1]), tf.TensorShape([1, ])],
)
def _sde_batch_shape_fixture(request):
    return request.param


@pytest.fixture(name="setup")
def _setup(sde_batch_shape):
    """ Construct an Ornstein Uhlenbeck SDE and a grid"""

    # construct a time grid
    t0, t1 = 0.0, 1.0
    num_transitions = 1000
    dt = float((t1 - t0) / num_transitions)
    time_grid = tf.cast(tf.linspace(t0 + dt, t1, num_transitions), dtype=DTYPE)

    # sample initial state
    state_dim = 1
    num_batch = sde_batch_shape.dims[0]
    x0_shape = (num_batch, state_dim)
    x0 = tf.random.normal(x0_shape, dtype=DTYPE)

    # construct OU sde
    decay = tf.random.normal((1, 1), dtype=DTYPE)
    q = tf.eye(state_dim, dtype=DTYPE)
    ou_sde = OrnsteinUhlenbeckSDE(decay, q)

    return ou_sde, x0, time_grid


def test_linearize_sde_ou_statedim_1(setup):
    """
    Test for checking the linearize sde method for Ornstein-Uhlenbeck SDE with state_dim=1.
    """

    ou_sde, x0, time_grid = setup
    decay = ou_sde.decay
    state_dim = ou_sde.state_dim
    dt = time_grid[1] - time_grid[0]

    # construct arbitrary marginal statistics
    q_mean = euler_maruyama(ou_sde, x0, time_grid)
    q_mean = tf.convert_to_tensor(q_mean, dtype=DTYPE)
    q_mean = q_mean[:, :-1, :]  # [num_batch, num_transitions, state_dim]

    q_covar = tf.zeros((q_mean.shape + state_dim), dtype=DTYPE)
    covar_diag = 1e-4 * tf.ones_like(q_mean)
    q_covar = tf.linalg.set_diag(
        q_covar, covar_diag
    )  # [num_batch, num_transitions, state_dim, state_dim]

    x0_covar = q_covar[:, 0, :, :]  # [num_batch, state_dim, state_dim]

    q = Gaussian(mu=q_mean, cov=q_covar)
    initial_state = Gaussian(mu=x0, cov=x0_covar)
    # linearize the sde around the path distribution provided by the marginal statistics
    linearized_ssm = linearize_sde(
        ou_sde, time_grid, q, initial_state
    )

    # ground true linearization for OU
    expected_A = tf.zeros_like(q_covar)  # [num_batch, num_transitions, state_dim, state_dim]
    expected_b = tf.zeros_like(q_mean) * dt  # [num_batch, num_transitions, state_dim]
    expected_A = tf.linalg.set_diag(expected_A, -decay + expected_b) * dt + tf.eye(ou_sde.state_dim, dtype=DTYPE)
    expected_chol_Q = ou_sde.diffusion(q_mean, t=None) * tf.sqrt(dt)

    np.testing.assert_allclose(linearized_ssm.state_transitions, expected_A, atol=1e-3)
    np.testing.assert_allclose(linearized_ssm.state_offsets, expected_b, atol=1e-3)
    np.testing.assert_allclose(
        linearized_ssm.cholesky_process_covariances, expected_chol_Q, atol=1e-3
    )


def test_euler_maruyama_shapes(setup):
    """Test checking the shape of the simulated values by Euler-Maruyama for Ornstein-Uhlenbeck SDE."""
    ou_sde, x0, time_grid = setup
    state_dim = ou_sde.state_dim

    num_transitions = time_grid.shape[0] - 1

    simulated_values = euler_maruyama(ou_sde, x0, time_grid)

    assert simulated_values.shape[0] == x0.shape[0]
    assert simulated_values.shape[1] == num_transitions + 1
    assert simulated_values.shape[-1] == state_dim


def test_deterministic_euler_maruyama_value(setup):
    """
    Test checking the simulated values by Euler-Maruyama for Ornstein-Uhlenbeck SDE.
    We make the diffusion term zero i.e. there is no noise and the drift value is set as 1.
    """
    ou_sde, x0, time_grid = setup
    decay = ou_sde.decay
    state_dim = ou_sde.state_dim
    dt = time_grid[1] - time_grid[0]

    num_transitions = time_grid.shape[0] - 1
    num_batch = x0.shape[0]

    # make diffusion zero and simulate
    ou_sde.q = 1e-20 * ou_sde.q
    simulated_values = euler_maruyama(ou_sde, x0, time_grid)

    expected_values = tf.zeros((num_transitions, num_batch, state_dim), dtype=DTYPE)
    expected_values = tf.concat([tf.expand_dims(x0, axis=0), expected_values], axis=0)

    expected_values = tf.scan(lambda a, x: a + (-decay * a * dt), expected_values)
    expected_values = tf.transpose(expected_values, [1, 0, 2])

    np.testing.assert_allclose(simulated_values, expected_values, atol=1e-5)


def test_KL_sde(setup):
    """
    Test the KL_sde function which calculates the KL divergence between two SDEs with same diffusion using
    Girsanov theorem.

    We assert with decimal place 2 as KL_SDE is an approximation. 
    With finer grid the tolerance can be made much tighter but it results in high running time.
    """
    ou_sde, x0, time_grid = setup

    # Removing batch as KL_sde only supports single batch
    if x0.shape[0] != 1:
        x0 = x0[:1]

    m = euler_maruyama(ou_sde, x0, time_grid)
    m = tf.squeeze(m, axis=0)

    dt = time_grid[1] - time_grid[0]

    x0 = tf.squeeze(x0, axis=0)

    # convert OU-SDE to SSM
    A_p = -1 * ou_sde.decay * tf.ones_like(m)[..., None] * dt + tf.eye(1, dtype=DTYPE)
    b_p = tf.zeros_like(m) * dt
    p_ssm = state_space_model_from_covariances(initial_mean=x0, initial_covariance=1e-1 * tf.ones_like(x0[..., None]),
                                               state_transitions=A_p,
                                               state_offsets=b_p,
                                               process_covariances=tf.square(ou_sde.diffusion(m, t=None)) * dt)

    # generate a q SSM and q SDE
    A_q_drift = -1 * np.random.random((1, 1)) * tf.ones_like(A_p)
    A_q = tf.eye(1, dtype=DTYPE) + A_q_drift * dt

    b_q = tf.zeros_like(b_p)
    b_q_drift = b_q / dt

    q_ssm = state_space_model_from_covariances(initial_mean=x0, initial_covariance=1e-1 * tf.ones_like(x0[..., None]),
                                               state_transitions=A_q, state_offsets=b_q,
                                               process_covariances=tf.square(ou_sde.diffusion(m, t=None)) * dt)

    kl_expected_val = q_ssm.kl_divergence(p_ssm)
    m, S = q_ssm.marginals
    q = Gaussian(mu=m[:-1], cov=S[:-1])

    linear_drift = LinearDrift(A=A_q_drift, b=b_q_drift)

    kl_val = squared_drift_difference_along_Gaussian_path(sde_p=ou_sde, linear_drift=linear_drift, q=q, dt=dt)

    np.testing.assert_array_almost_equal(kl_expected_val, kl_val, decimal=2)


def test_ssm_to_linear_drift(setup):
    ou_sde, x0, time_grid = setup

    # Removing batch
    if x0.shape[0] != 1:
        x0 = x0[:1]

    m = euler_maruyama(ou_sde, x0, time_grid)
    m = tf.squeeze(m, axis=0)

    dt = time_grid[1] - time_grid[0]

    x0 = tf.squeeze(x0, axis=0)
    A_p = -1 * ou_sde.decay * tf.ones_like(m)[..., None] * dt + tf.eye(1, dtype=DTYPE)
    b_p = tf.zeros_like(m) * dt

    p_ssm = state_space_model_from_covariances(initial_mean=x0, initial_covariance=1e-1 * tf.ones_like(x0[..., None]),
                                               state_transitions=A_p,
                                               state_offsets=b_p,
                                               process_covariances=tf.square(ou_sde.diffusion(m, t=None)) * dt)
    linear_drift = LinearDrift()
    linear_drift.set_from_ssm(p_ssm, dt=dt)

    true_A = -1 * ou_sde.decay * tf.ones_like(m)[..., None]
    true_b = tf.zeros_like(m)
    np.testing.assert_allclose(true_A, linear_drift.A)
    np.testing.assert_allclose(true_b, linear_drift.b)
