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
"""Module containing the integration tests for the `CVIGaussianProcess` class."""
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import Matern12
from markovflow.models import CVIGaussianProcess, GaussianProcessRegression
from tests.tools.generate_random_objects import generate_random_time_observations

OUT_DIM = 1
LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8
batch_shape = ()
output_dim = 1


@pytest.fixture(name="vgp_gpr_optim_setup")
def _vgp_gpr_optim_setup():
    """ Creates a GPR model and a matched VGP model, and optimize the later (single step)"""

    time_points, observations, kernel, variance = _setup()

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = GaussianProcessRegression(
        input_data=input_data,
        kernel=kernel,
        mean_function=None,
        chol_obs_covariance=chol_obs_covariance,
    )

    likelihood = Gaussian(variance=variance)
    vgp = CVIGaussianProcess(
        input_data=(time_points, observations),
        kernel=kernel,
        likelihood=likelihood,
        learning_rate=1.0,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    vgp.update_sites()

    return vgp, gpr


def _setup():
    """ Data, kernel and likelihood setup """
    time_points, observations = generate_random_time_observations(
        obs_dim=output_dim, num_data=NUM_DATA, batch_shape=batch_shape
    )
    time_points = tf.constant(time_points)
    observations = tf.constant(observations)

    kernel = Matern12(lengthscale=LENGTH_SCALE, variance=VARIANCE, output_dim=output_dim)

    observation_noise = 1.0
    variance = tf.constant(observation_noise, dtype=tf.float64)

    return time_points, observations, kernel, variance


def test_vgp_elbo_optimal(with_tf_random_seed, vgp_gpr_optim_setup):
    """Test that the value of the ELBO at the optimum is the same as the GPR Log Likelihood."""
    vgp, gpr = vgp_gpr_optim_setup
    np.testing.assert_allclose(vgp.elbo(), gpr.log_likelihood())


def test_loss(with_tf_random_seed, vgp_gpr_optim_setup):
    """Test that the loss is equal to the negative ELBO."""
    vgp, _ = vgp_gpr_optim_setup
    np.testing.assert_allclose(-vgp.elbo(), vgp.loss())


def test_vgp_unchanged_at_optimum(with_tf_random_seed, vgp_gpr_optim_setup):
    """Test that the update does not change sites at the optimum"""
    vgp, _ = vgp_gpr_optim_setup
    # ELBO at optimum
    optim_elbo = vgp.elbo()
    # site update step
    vgp.update_sites()
    # ELBO after step
    new_elbo = vgp.elbo()

    with tf.GradientTape() as g:
        g.watch(vgp.trainable_variables)
        elbo = vgp.classic_elbo()
    grad_elbo = g.gradient(elbo, vgp.trainable_variables)

    for g in grad_elbo:
        if g is not None:
            np.testing.assert_allclose(g, tf.zeros_like(g), atol=1e-9)
    np.testing.assert_allclose(optim_elbo, new_elbo, atol=1e-9)


def _test_vgp_vs_gpr(vgp, gpr):
    """Test that the VGP with Gaussian Likelihood gives the same results as GPR."""

    true_likelihood = gpr.log_likelihood()
    for __ in range(10):  # iterations per try
        vgp.update_sites()

    trained_likelihood = vgp.elbo()

    np.testing.assert_allclose(trained_likelihood, true_likelihood, atol=1e-6, rtol=1e-6)


def test_optimal_sites(with_tf_random_seed, vgp_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites """
    vgp, gpr = vgp_gpr_optim_setup

    vgp_nat1 = vgp.sites.nat1.numpy()
    vgp_nat2 = vgp.sites.nat2.numpy()

    # manually compute the optimal sites
    s2 = gpr._chol_obs_covariance.numpy()
    gpr_nat1 = gpr.observations / s2
    gpr_nat2 = -0.5 / s2 * np.ones_like(vgp_nat2)

    np.testing.assert_allclose(vgp_nat1, gpr_nat1)
    np.testing.assert_allclose(vgp_nat2, gpr_nat2)


def test_posterior_ssm(with_tf_random_seed, vgp_gpr_optim_setup):
    """Self consistency test on two ways to compute the posterior ssm """
    cvi, _ = vgp_gpr_optim_setup

    q_ssm = cvi.dist_q
    kf_ssm = cvi.posterior_kalman.posterior_state_space_model()

    np.testing.assert_allclose(q_ssm.marginal_means, kf_ssm.marginal_means)
    np.testing.assert_allclose(q_ssm.marginal_covariances, kf_ssm.marginal_covariances)

    np.testing.assert_allclose(q_ssm.state_transitions, kf_ssm.state_transitions)
    np.testing.assert_allclose(q_ssm.state_offsets, kf_ssm.state_offsets)
