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
"""Module containing the integration tests for the `VariationalGaussianProcess` class."""
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import Matern12
from markovflow.mean_function import LinearMeanFunction
from markovflow.models import GaussianProcessRegression, VariationalGaussianProcess
from tests.tools.generate_random_objects import generate_random_time_observations

LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 8


@pytest.fixture(name="vgp_gpr_optim_setup")
def _vgp_gpr_optim_setup(batch_shape, output_dim):
    time_points, observations, kernel, variance = _setup(batch_shape, output_dim)

    # nasty hack because tensorflow complains about uninitialised variables when passed as params.
    kernel._lengthscale = tf.constant(LENGTH_SCALE, dtype=tf.float64)
    kernel._variance = tf.constant(VARIANCE, dtype=tf.float64)

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = GaussianProcessRegression(
        input_data=input_data, kernel=kernel, chol_obs_covariance=chol_obs_covariance
    )

    likelihood = Gaussian(variance=variance)
    input_data = (tf.constant(time_points), tf.constant(observations))
    vgp = VariationalGaussianProcess(
        input_data=input_data,
        kernel=kernel,
        likelihood=likelihood,
        initial_distribution=gpr.posterior.gauss_markov_model,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables:
        t._trainable = False

    return vgp, gpr


@pytest.fixture(name="vgp_gpr_batch_setup")
def _vgp_gpr_batch_setup(batch_shape, output_dim):
    return _vgp_gpr_setup(batch_shape, output_dim, None)


@pytest.fixture(name="vgp_gpr_mean_setup")
def _vgp_gpr_mean_setup(output_dim):
    return _vgp_gpr_setup(tuple(), output_dim, LinearMeanFunction(1.5, output_dim))


def _vgp_gpr_setup(batch_shape, output_dim, mean_function):
    time_points, observations, kernel, variance = _setup(batch_shape, output_dim)

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (tf.constant(time_points), tf.constant(observations))
    gpr = GaussianProcessRegression(
        input_data=input_data,
        kernel=kernel,
        mean_function=mean_function,
        chol_obs_covariance=chol_obs_covariance,
    )

    likelihood = Gaussian(variance=variance)
    vgp = VariationalGaussianProcess(
        input_data=input_data, kernel=kernel, likelihood=likelihood, mean_function=mean_function
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return vgp, gpr


def _setup(batch_shape, output_dim):
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


def test_vgp_grad_optimal(with_tf_random_seed, vgp_gpr_optim_setup):
    """Test that the gradient of the ELBO at the optimum is zero."""
    vgp, _ = vgp_gpr_optim_setup
    with tf.GradientTape() as tape:
        elbo = vgp.elbo()
    eval_grads = tape.gradient(elbo, vgp.trainable_variables)
    # Check that we've actually computed some gradients
    assert eval_grads is not None
    for grad in eval_grads:
        np.testing.assert_allclose(grad, 0.0, atol=1e-9)


def _test_vgp_vs_gpr(vgp, gpr):
    """Test that the VGP with Gaussian Likelihood gives the same results as GPR."""
    opt = tf.optimizers.Adam(learning_rate=1e-2)

    @tf.function
    def opt_step():
        opt.minimize(vgp.loss, vgp.trainable_variables)

    true_likelihood = gpr.log_likelihood()
    for _ in range(100):  # number of tries
        for __ in range(100):  # iterations per try
            # Only optimise the SSM parameters
            opt_step()
        trained_likelihood = vgp.elbo()
        if np.allclose(trained_likelihood, true_likelihood, atol=1e-6, rtol=1e-6):
            break
    np.testing.assert_allclose(trained_likelihood, true_likelihood, atol=1e-6, rtol=1e-6)


def test_vgp_vs_gpr_batch(with_tf_random_seed, vgp_gpr_batch_setup):
    """
    Test that VGP with a Gaussian Likelihood gives the same results as GPR.

    Tested with different batch shapes and output dimensions.
    """
    _test_vgp_vs_gpr(*vgp_gpr_batch_setup)


def test_vgp_vs_gpr_means(with_tf_random_seed, vgp_gpr_mean_setup):
    """
    Test that VGP with a Gaussian Likelihood gives the same results as GPR.

    Tested with a mean function
    """
    _test_vgp_vs_gpr(*vgp_gpr_mean_setup)
