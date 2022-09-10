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
"""Module containing the integration tests for the `CVISDESparseSites` class."""
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian
from gpflow.config import default_float

from markovflow.sde import OrnsteinUhlenbeckSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.models import GaussianProcessRegression, CVISDESparseSites
from markovflow.sde.sde_utils import euler_maruyama

DECAY = 0.5
Q = 0.8
T0 = 0.
T1 = 1.
DT = 0.001
NUM_DATA = 8
NOISE_STDDEV = 1.
batch_shape = ()
DTYPE = default_float()


@pytest.fixture(name="cvi_sde_gpr_optim_setup")
def _cvi_sde_gpr_optim_setup():
    """
    Creates a GPR model and a matched CVI-SDE sparse sites model, and optimize the later (single step).
    """
    observation_grid, observations, time_grid, kernel, sde = _setup()

    input_data = (tf.constant(observation_grid), tf.constant(observations))

    chol_obs_covariance = tf.eye(1, dtype=DTYPE) * NOISE_STDDEV

    gpr = GaussianProcessRegression(
        input_data=input_data,
        kernel=kernel,
        mean_function=None,
        chol_obs_covariance=chol_obs_covariance,
    )

    likelihood = Gaussian(variance=NOISE_STDDEV ** 2)
    cvi_sde = CVISDESparseSites(
        prior_sde=sde,
        time_grid=time_grid,
        input_data=input_data,
        likelihood=likelihood,
        learning_rate=1.0,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    cvi_sde.update_sites()

    return cvi_sde, gpr


def _setup():
    """Generate a time-series data from Ornstein-Uhlenbeck SDE."""
    ou_sde = OrnsteinUhlenbeckSDE(decay=DECAY * tf.ones((1, 1), dtype=DTYPE), q=Q * tf.ones((1, 1), dtype=DTYPE))
    time_grid = tf.cast(tf.linspace(T0, T1, int((T1 - T0) // DT) + 2), dtype=DTYPE)
    simulated_vals = euler_maruyama(ou_sde, x0=tf.zeros((1, 1), dtype=DTYPE), time_grid=time_grid)
    simulated_vals = tf.squeeze(simulated_vals, axis=0)

    # observations
    observation_grid = tf.convert_to_tensor(
        np.sort(np.random.choice(time_grid, NUM_DATA, replace=False)).reshape((-1,)), dtype=DTYPE)
    observation_idx = tf.where(tf.equal(time_grid[..., None], observation_grid))[:, 0]
    observations = tf.gather(simulated_vals, observation_idx, axis=0)
    observations = observations + tf.random.normal(observations.shape, stddev=NOISE_STDDEV, dtype=DTYPE)

    kernel = OrnsteinUhlenbeck(decay=DECAY, diffusion=Q)

    return observation_grid, observations, time_grid, kernel, ou_sde


def test_elbo_optimal(with_tf_random_seed, cvi_sde_gpr_optim_setup):
    """
    Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood.

    We compare with less tolerance i.e. 4 decimal places as in CVI-SDE model we linearize using Taylor expansion which
    is an approximation to the prior in GPR.

    """
    cvi_sde, gpr = cvi_sde_gpr_optim_setup
    np.testing.assert_array_almost_equal(cvi_sde.elbo(), gpr.log_likelihood(), decimal=4)


def test_unchanged_at_optimum(with_tf_random_seed, cvi_sde_gpr_optim_setup):
    """
    Test that the update does not change sites at the optimum
    """
    cvi_sde, _ = cvi_sde_gpr_optim_setup
    # ELBO at optimum
    optim_elbo = cvi_sde.elbo()
    # site update step
    cvi_sde.update_sites()
    # ELBO after step
    new_elbo = cvi_sde.elbo()

    with tf.GradientTape() as g:
        g.watch(cvi_sde.trainable_variables)
        elbo = cvi_sde.classic_elbo()
    grad_elbo = g.gradient(elbo, cvi_sde.trainable_variables)

    for g in grad_elbo:
        if g is not None:
            np.testing.assert_array_almost_equal(g, tf.zeros_like(g))
    np.testing.assert_array_almost_equal(optim_elbo, new_elbo)


def test_optimal_sites(with_tf_random_seed, cvi_sde_gpr_optim_setup):
    """
    Test that the optimal value of the exact sites match the true sites.
    """
    cvi_sde, gpr = cvi_sde_gpr_optim_setup

    cvi_sde_nat1 = cvi_sde.sites.nat1.numpy()
    cvi_sde_nat2 = cvi_sde.sites.nat2.numpy()

    # manually compute the optimal sites
    s2 = gpr._chol_obs_covariance.numpy()
    gpr_nat1 = gpr.observations / s2
    gpr_nat2 = -0.5 / s2 * np.ones_like(cvi_sde_nat2)

    np.testing.assert_allclose(cvi_sde_nat1, gpr_nat1)
    np.testing.assert_allclose(cvi_sde_nat2, gpr_nat2)


def test_elbo_terms(with_tf_random_seed, cvi_sde_gpr_optim_setup):
    """
    For a linear SDE prior (i.e. a Gaussian Process [GP]), test that the ELBO computed
    using Girsanov theorem or the usual formula for GPs match.
    """
    cvi_sde, gpr = cvi_sde_gpr_optim_setup

    cross_term = cvi_sde.kl_cross_term()
    linearization_loss = cvi_sde.kl_linearization_loss()

    np.testing.assert_array_almost_equal(cross_term, tf.zeros_like(cross_term))
    np.testing.assert_array_almost_equal(linearization_loss, tf.zeros_like(linearization_loss))

    ve_kl = cvi_sde.variational_expectation() - cvi_sde.dist_q.kl_divergence(cvi_sde.dist_p)
    np.testing.assert_array_almost_equal(ve_kl, gpr.log_likelihood(), decimal=4)
