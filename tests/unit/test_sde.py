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

from markovflow.sde.sde_utils import euler_maruyama, linearize_sde
from markovflow.sde import OrnsteinUhlenbeckSDE

tf.random.set_seed(33)

DTYPE = tf.float32
gpflow.config.set_default_float(DTYPE)


@pytest.fixture(name="setup")
def _setup(state_dim, batch_shape):
    """"""
    t0 = 0.
    t1 = 1.
    n_transitions = 10
    n_batch = batch_shape.dims[0].value if batch_shape.ndims > 0 else 1
    x0_shape = (n_batch, state_dim)

    dt = float((t1 - t0) / n_transitions)
    x0 = tf.random.normal(x0_shape, dtype=DTYPE)

    decay = tf.random.normal((1, 1), dtype=DTYPE)
    q = tf.eye(state_dim, dtype=DTYPE)
    ou_sde = OrnsteinUhlenbeckSDE(decay, q)

    time_grid = tf.cast(tf.linspace(t0 + dt, t1, n_transitions), dtype=DTYPE)

    return state_dim, decay, ou_sde, x0, dt, time_grid


def test_linearize_sde_ou_statedim_1(setup):
    """
    Test for checking the linearize sde method for Ornstein-Uhlenbeck SDE with state_dim=1.
    """

    state_dim, decay, ou_sde, x0, dt, time_grid = setup

    # As currently only 1 state dim is supported
    if state_dim != 1:
        return True

    q_mean = euler_maruyama(ou_sde, x0, time_grid)
    q_mean = tf.convert_to_tensor(q_mean, dtype=DTYPE)
    q_mean = q_mean[:, :-1, :]  # [n_batch, n_transitions, state_dim]

    q_covar = tf.zeros((q_mean.shape + state_dim), dtype=DTYPE)
    covar_diag = 1e-4 * tf.ones(q_mean.shape)
    q_covar = tf.linalg.set_diag(q_covar, covar_diag)  # [n_batch, n_transitions, state_dim, state_dim]

    x0_covar_chol = tf.linalg.cholesky(q_covar[:, 0, :, :])  # [n_batch, state_dim, state_dim]
    noise_covar = 1e-2 * tf.ones_like(q_covar)  # [n_batch, n_transitions, state_dim, state_dim]

    # Adding t0 to time_grid
    time_grid = tf.concat([tf.zeros(1,), time_grid], axis=0)

    linearized_ssm = linearize_sde(ou_sde, time_grid, q_mean, q_covar, x0, x0_covar_chol, noise_covar)

    expected_A = tf.zeros_like(q_covar)  # [n_batch, n_transitions, state_dim, state_dim]
    expected_b = tf.zeros_like(q_mean) * dt  # [n_batch, n_transitions, state_dim]
    expected_A = tf.linalg.set_diag(expected_A, -decay + expected_b) * dt

    np.testing.assert_allclose(linearized_ssm.state_transitions, expected_A, atol=1e-3)
    np.testing.assert_allclose(linearized_ssm.state_offsets, expected_b, atol=1e-3)


def test_euler_maruyama_shapes(setup):
    """Test checking the shape of the simulated values by Euler-Maruyama for Ornstein-Uhlenbeck SDE."""
    state_dim, decay, ou_sde, x0, dt, t = setup
    n_transitions = t.shape[0]

    simulated_values = euler_maruyama(ou_sde, x0, t)

    assert simulated_values.shape[0] == x0.shape[0]
    assert simulated_values.shape[1] == n_transitions+1
    assert simulated_values.shape[-1] == state_dim


def test_euler_maruyama_value(setup):
    """
    Test checking the simulated values by Euler-Maruyama for Ornstein-Uhlenbeck SDE.
    We make the diffusion term zero i.e. there is no noise and the drift value is set as 1.
    """
    state_dim, decay, ou_sde, x0, dt, t = setup
    n_transitions = t.shape[0]
    n_batch = x0.shape[0]

    # make diffusion zero and simulate
    ou_sde.q = 1e-20 * ou_sde.q
    simulated_values = euler_maruyama(ou_sde, x0, t)

    expected_values = tf.zeros((n_transitions, n_batch, state_dim), dtype=DTYPE)
    expected_values = tf.concat([tf.expand_dims(x0, axis=0), expected_values], axis=0)

    expected_values = tf.scan(lambda a, x: a + (-decay*a*dt), expected_values)
    expected_values = tf.transpose(expected_values, [1, 0, 2])

    np.testing.assert_allclose(simulated_values, expected_values, atol=1e-6)
