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

from markovflow.sde.linearize_sde import LinearizeSDE
from markovflow.sde.sde_utils import euler_maruyama
from markovflow.sde import OrnsteinUhlenbeckSDE

tf.random.set_seed(33)

DTYPE = tf.float32
gpflow.config.set_default_float(DTYPE)


@pytest.fixture(name="setup")
def _setup(state_dim):
    """"""
    t0 = 0.
    t1 = 10.
    n = 10
    dt = float((t1 - t0) / n)
    x0 = tf.random.normal((1, state_dim), dtype=DTYPE)

    decay = tf.random.normal((1, 1), dtype=DTYPE)
    ou_sde = OrnsteinUhlenbeckSDE(decay)

    linearize_points = tf.cast(tf.linspace(t0 + dt, t1, n), dtype=DTYPE)

    return state_dim, decay, ou_sde, x0, dt, linearize_points


def test_ou_update_params_non_sparse(setup):
    """
    Test for checking the update parameter of LinearizedSDE class for Ornstein-Uhlenbeck SDE for non-sparse case i.e.
    inference grid is same as linearization grid.
    """

    dim, decay, ou_sde, x0, dt, linearize_points = setup

    inference_grid = tf.identity(linearize_points)

    linearize_sde = LinearizeSDE(ou_sde, linearize_points, dim=dim)

    q_mean = euler_maruyama(ou_sde, x0, inference_grid)
    q_mean = tf.convert_to_tensor(q_mean, dtype=DTYPE)
    q_covar = 1e-2 * tf.repeat(tf.expand_dims(tf.eye(dim, dtype=DTYPE), axis=0), q_mean.shape[0], axis=0)

    linearize_sde.update_linearization_parameters(q_mean, q_covar, inference_grid)

    expected_A = -decay * tf.ones((linearize_points.shape[0]+1, 1, 1), dtype=DTYPE)
    expected_b = tf.zeros((linearize_points.shape[0]+1, 1), dtype=DTYPE)

    for i in range(dim):
        linearize_sde_A_i = tf.reshape(tf.reduce_sum(linearize_sde.A, axis=-1)[:, i], (-1, 1, 1))
        np.testing.assert_allclose(linearize_sde_A_i, expected_A, atol=1e-4)

    linearize_sde_b = tf.reshape(tf.reduce_sum(linearize_sde.b, axis=-1), (-1, 1))
    np.testing.assert_allclose(linearize_sde_b, expected_b, atol=1e-4)


def test_euler_maruyama_shapes(setup):
    """Test checking the shape of the simulated values by Euler-Maruyama for Ornstein-Uhlenbeck SDE."""
    state_dim, decay, ou_sde, x0, dt, t = setup
    n = t.shape[0]

    simulated_vals = euler_maruyama(ou_sde, x0, t)

    tf.assert_equal(tf.gather(simulated_vals, [0]), x0)

    assert simulated_vals.shape[0] == n+1
    assert simulated_vals.shape[-1] == state_dim

