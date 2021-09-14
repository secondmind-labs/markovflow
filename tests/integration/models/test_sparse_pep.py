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
from markovflow.models import (
    GaussianProcessRegression,
    SparseCVIGaussianProcess,
    SparsePowerExpectationPropagation,
)
from tests.tools.generate_random_objects import generate_random_time_observations

OUT_DIM = 1
LENGTH_SCALE = 2.0
VARIANCE = 2.25
NUM_DATA = 2
batch_shape = ()
output_dim = 1


@pytest.fixture(name="spep_gpr_optim_setup")
def _spep_gpr_optim_setup():
    """
    Creates a GPR model and a matched Sparse PEP model (z=x),
    and optimize the later (single step)
    """

    time_points, observations, kernel, variance = _setup()

    chol_obs_covariance = tf.eye(output_dim, dtype=tf.float64) * tf.sqrt(variance)
    input_data = (time_points, observations)
    inducing_points = time_points + 1e-10
    gpr = GaussianProcessRegression(
        kernel=kernel,
        input_data=input_data,
        chol_obs_covariance=chol_obs_covariance,
        mean_function=None,
    )

    likelihood = Gaussian(variance=variance)
    sep = SparsePowerExpectationPropagation(
        kernel=kernel,
        inducing_points=inducing_points,
        likelihood=PEPScalarLikelihood(likelihood),
        learning_rate=0.1,
        alpha=1.0,
    )

    scvi = SparseCVIGaussianProcess(
        kernel=kernel, inducing_points=inducing_points, likelihood=likelihood, learning_rate=1.0,
    )

    # do not train any hyper-parameters for these tests
    for t in likelihood.trainable_variables + kernel.trainable_variables:
        t._trainable = False

    # update sites -> optimal
    scvi.update_sites(input_data)
    sep.nat1.assign(scvi.nat1.numpy())
    sep.nat2.assign(scvi.nat2.numpy())

    return sep, gpr, input_data


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


def test_optimal_sites(with_tf_random_seed, spep_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites """
    spep, gpr, data = spep_gpr_optim_setup

    spep.learning_rate = 1.0
    spep.alpha = 1.0
    spep.update_sites(data)

    sd = spep.kernel.state_dim

    # for z = x, the sites are 2 sd x 2 sd but half empty
    # one part must match the GPR site
    spep_nat1 = spep.nat1.numpy()[:-1, sd:]
    spep_nat2 = spep.nat2.numpy()[:-1, sd:, sd:]
    spep_log_norm = spep.log_norm.numpy()[:-1]
    spep_energy = spep.energy(data).numpy()
    # manually compute the optimal sites
    s2 = gpr._chol_obs_covariance.numpy() ** 2
    gpr_nat1 = gpr.observations / s2
    gpr_nat2 = -0.5 / s2 * np.ones_like(spep_nat2)
    gpr_log_norm = -0.5 * gpr.observations.numpy() ** 2 / s2 - 0.5 * np.log(2.0 * np.pi * s2)
    gpr_llh = gpr.log_likelihood().numpy()

    np.testing.assert_array_almost_equal(spep_nat1, gpr_nat1, decimal=3)
    np.testing.assert_array_almost_equal(spep_nat2, gpr_nat2, decimal=3)
    np.testing.assert_array_almost_equal(gpr_log_norm, spep_log_norm, decimal=4)
    np.testing.assert_array_almost_equal(gpr_llh, spep_energy, decimal=4)


def test_log_norm(with_tf_random_seed, spep_gpr_optim_setup):
    """Test that the optimal value of the exact sites match the true sites """

    # sites are set to optimal  (but not the sites)
    spep, gpr, data = spep_gpr_optim_setup
    a = 1.0
    spep.alpha = a
    spep_log_norm = spep.compute_log_norm(data).numpy()[:-1]

    s2 = gpr._chol_obs_covariance.numpy() ** 2
    gpr_log_norm = -0.5 * gpr.observations.numpy() ** 2 / s2 - 0.5 * np.log(2.0 * np.pi * s2)

    np.testing.assert_array_almost_equal(gpr_log_norm, spep_log_norm, decimal=4)


def test_convergence_of_spep(with_tf_random_seed, spep_gpr_optim_setup):
    """Test that the optimal sites are fixed points of the update """
    spep, gpr, input_data = spep_gpr_optim_setup

    # run EP site optimization
    for _ in range(20):
        spep.update_sites(input_data)

    # run one last step of EP
    old_nat1 = spep.nat1.numpy()
    old_nat2 = spep.nat2.numpy()

    spep.update_sites(input_data)

    new_nat1 = spep.nat1.numpy()
    new_nat2 = spep.nat2.numpy()

    np.testing.assert_array_almost_equal(new_nat1, old_nat1)
    np.testing.assert_array_almost_equal(new_nat2, old_nat2)

    np.testing.assert_array_almost_equal(
        gpr.log_likelihood().numpy(), spep.energy(input_data=input_data)
    )
