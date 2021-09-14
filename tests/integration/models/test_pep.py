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
"""Module containing the integration tests for the `SparsePowerExpectationPropagation` class."""
import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import Matern12
from markovflow.likelihoods import PEPGaussian, PEPScalarLikelihood
from markovflow.models import GaussianProcessRegression, PowerExpectationPropagation
from tests.tools.generate_random_objects import generate_random_time_observations

OUT_DIM = 1
LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 2
batch_shape = ()
output_dim = 1


@pytest.fixture(name="pep_gpr_setup")
def _pep_gpr_setup():
    """
    Creates a GPR model and a matched PEP model,
    and optimize the later (single step)
    """

    time_points, observations, kernel, variance = _setup()

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (time_points, observations)
    gpr = GaussianProcessRegression(
        kernel=kernel,
        input_data=input_data,
        chol_obs_covariance=chol_obs_covariance,
        mean_function=None,
    )

    # likelihood = PEPGaussian(Gaussian(variance=variance))
    likelihood = PEPScalarLikelihood(Gaussian(variance=variance))
    pep = PowerExpectationPropagation(
        kernel=kernel, input_data=input_data, likelihood=likelihood, learning_rate=1.0, alpha=1.0
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    return pep, gpr


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


def test_convergence_of_pep_to_optimal(with_tf_random_seed, pep_gpr_setup):
    """Test that PEP converges to the optimal sites """

    pep, gpr = pep_gpr_setup

    batch_size = 1
    for _ in range(20):
        indices = np.random.permutation(NUM_DATA)[:batch_size].reshape(-1, 1)
        pep.update_sites(indices)

    # compute optimal sites p(y|f) = N(y;f,s2)
    s2 = gpr._chol_obs_covariance.numpy() ** 2
    opt_nat1 = gpr.observations / s2
    opt_nat2 = -0.5 / s2 * np.ones_like(pep.sites.nat2.numpy())
    opt_log_norm = -0.5 * gpr.observations.numpy() ** 2 / s2 - 0.5 * np.log(2.0 * np.pi * s2)

    final_nat1 = pep.sites.nat1.numpy()
    final_nat2 = pep.sites.nat2.numpy()
    final_log_norm = pep.sites.log_norm.numpy()

    np.testing.assert_array_almost_equal(final_log_norm, opt_log_norm, decimal=3)
    np.testing.assert_array_almost_equal(final_nat1, opt_nat1, decimal=3)
    np.testing.assert_array_almost_equal(final_nat2, opt_nat2, decimal=3)


def test_PEPlikelihood():
    """
    Test that Gauss Hermite implementation of ``log_expected_density`` and its gradient
    match the analytical one for the Gaussian case.
    """

    lik1 = PEPScalarLikelihood(Gaussian(1.0), num_gauss_hermite_points=10)
    lik2 = PEPGaussian(Gaussian(1.0))

    num_data = 1
    Y = tf.constant(np.random.randn(num_data, 1))
    Fmu = tf.constant(np.random.randn(num_data, 1))
    Fvar = tf.constant(np.random.rand(num_data, 1))

    led1 = lik1.log_expected_density(Fmu, Fvar, Y)
    led2 = lik2.log_expected_density(Fmu, Fvar, Y)

    _, gled1 = lik1.grad_log_expected_density(Fmu, Fvar, Y)
    _, gled2 = lik2.grad_log_expected_density(Fmu, Fvar, Y)

    np.testing.assert_array_almost_equal(led1.numpy(), led2.numpy())
    np.testing.assert_array_almost_equal(gled1[0].numpy(), gled2[0].numpy())
    np.testing.assert_array_almost_equal(gled1[1].numpy(), gled2[1].numpy())


def test_pep_updates():
    """
    Test that the manually computed EP updates at optimum do not change the sites"""

    lik = PEPGaussian(Gaussian(1.0))
    Y = np.random.randn(3, 1) * 0

    # prior on f
    prior_var = np.random.rand(3, 1)
    prior_mu = np.random.rand(3, 1)

    # site on f set to optimum
    s2 = lik.base.variance.numpy()
    site_nat1 = Y / s2
    site_nat2 = -0.5 / s2 * np.ones((3, 1))

    # Manual computation of the PEP update
    # cavity
    prior_nat1 = prior_mu / prior_var
    prior_nat2 = -0.5 / prior_var
    cav_nat1 = prior_nat1 - site_nat1
    cav_nat2 = prior_nat2 - site_nat2
    cav_mean = -0.5 * cav_nat1 / cav_nat2
    cav_var = -0.5 / cav_nat2

    # gradient of objective
    _, grads = lik.grad_log_expected_density(
        tf.constant(cav_mean), tf.constant(cav_var), tf.constant(Y)
    )

    # PEP update computed manually
    new_site_mean = cav_mean - grads[0] / grads[1]
    new_site_cov = -cav_var - 1.0 / grads[1]
    new_site_nat1 = new_site_mean / new_site_cov
    new_site_nat2 = -0.5 / new_site_cov

    # compare
    np.testing.assert_array_almost_equal(new_site_nat1.numpy(), site_nat1, decimal=3)
    np.testing.assert_array_almost_equal(new_site_nat2.numpy(), site_nat2, decimal=3)
