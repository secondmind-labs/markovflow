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
"""Tests for the `PosteriorProcess` class."""
from dataclasses import dataclass
from typing import Tuple

import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow import default_float

from markovflow.conditionals import pairwise_marginals
from markovflow.kernels import Matern32
from markovflow.mean_function import LinearMeanFunction
from markovflow.models import GaussianProcessRegression
from tests.tools.state_space_model import StateSpaceModelBuilder


def create_markovflow_gpr(time_points, observations):
    """
    Helper method to create a 'GaussianProcessRegression'
    """
    observation_covariance = 1.0  # Same as GPFlow default
    input_data = (
        tf.constant(time_points, dtype=default_float()),
        tf.constant(observations, dtype=default_float()),
    )
    return GaussianProcessRegression(
        input_data=input_data,
        kernel=Matern32(lengthscale=1.0, variance=1.0, output_dim=observations.shape[-1]),
        mean_function=LinearMeanFunction(1.1),
        chol_obs_covariance=tf.constant([[np.sqrt(observation_covariance)]], dtype=default_float()),
    )


def create_gpflow_gpr(time_points, observations):
    """ Helper method to create a GPFlow GPR """
    return gpflow.models.GPR(
        data=(time_points[..., None], observations),
        kernel=gpflow.kernels.Matern32(1),
        mean_function=gpflow.mean_functions.Linear(1.1),
    )


@dataclass
class TestData:
    time_points: np.ndarray
    observations: np.ndarray
    intermediate_time_points: np.ndarray
    future_time_points: np.ndarray


def as_batch(data: np.ndarray, batch_size: Tuple) -> np.ndarray:
    """ Helper method to tile an array to make batches """
    existing_batch = tuple(1 for _ in data.shape)
    return np.tile(data, reps=batch_size + existing_batch)


def get_test_data(batch_size: Tuple) -> Tuple[TestData, TestData]:
    """
    Returns a tuple of 'TestData'.  The first item is the raw test data and the second item is the
    'batch' version of it.
    """
    time_points = np.linspace(0.0, 10.0, 10)
    observations = np.sin(12 * time_points[..., None]) + np.random.randn(len(time_points), 1) * 0.1
    intermediate_time_points = np.arange(0.0, time_points[-1], 0.5)
    future_time_points = np.arange(time_points[-1] + 0.5, 13.0, 0.5)

    test_data = TestData(
        time_points=time_points,
        observations=observations,
        intermediate_time_points=intermediate_time_points,
        future_time_points=future_time_points,
    )

    batch_test_data = TestData(
        time_points=as_batch(time_points, batch_size),
        observations=as_batch(observations, batch_size),
        intermediate_time_points=as_batch(intermediate_time_points, batch_size),
        future_time_points=as_batch(future_time_points, batch_size),
    )

    return test_data, batch_test_data


def test_posterior_predict_f_with_gpflow(batch_shape, with_tf_random_seed):
    test_data, batch_test_data = get_test_data(batch_shape)
    # Generate a GP Flow GPR and call predict_f for comparison
    gpflow_gpr = create_gpflow_gpr(test_data.time_points, test_data.observations)

    mu, cov = gpflow_gpr.predict_f(test_data.intermediate_time_points[..., None])

    mf_gpr = create_markovflow_gpr(batch_test_data.time_points, batch_test_data.observations)
    mf_mu, mf_cov = mf_gpr.posterior.predict_f(
        tf.constant(batch_test_data.intermediate_time_points, dtype=default_float())
    )

    np.testing.assert_array_almost_equal(
        as_batch(mu.numpy(), batch_shape).squeeze(), mf_mu.numpy().squeeze()
    )
    np.testing.assert_array_almost_equal(
        as_batch(cov.numpy(), batch_shape).squeeze(), mf_cov.numpy().squeeze()
    )


@pytest.mark.skip(reason='too slow and not crucial')
def test_mu_sample_f_posterior_sampling(batch_shape, with_tf_random_seed):
    """
    Tests that using `sample_f` the expectation of samples from the posterior is equal to the
    posterior mean.
    """
    _, batch_test_data = get_test_data(batch_shape)
    gpr_posterior = create_markovflow_gpr(
        batch_test_data.time_points, batch_test_data.observations
    ).posterior
    test_times = np.concatenate(
        [batch_test_data.intermediate_time_points, batch_test_data.future_time_points], axis=-1
    )
    tf_test_times = tf.constant(test_times, dtype=default_float())
    samples = gpr_posterior.sample_f(tf_test_times, sample_shape=10000)
    mu, _ = gpr_posterior.predict_f(tf_test_times)

    np.testing.assert_allclose(mu, samples.numpy().mean(0), atol=1e-1)


@pytest.mark.skip(reason='too slow and not crucial')
def test_var_sample_f_posterior_sampling(batch_shape, with_tf_random_seed):
    """
    Tests that using `sample_f` the expected variance of samples from the posterior is equal to
    the posterior variance.
    """
    _, batch_test_data = get_test_data(batch_shape)
    gpr_posterior = create_markovflow_gpr(
        batch_test_data.time_points, batch_test_data.observations
    ).posterior
    test_times = np.concatenate(
        [batch_test_data.intermediate_time_points, batch_test_data.future_time_points], axis=-1
    )
    tf_test_times = tf.constant(test_times, dtype=default_float())
    tf_samples = gpr_posterior.sample_f(tf_test_times, sample_shape=10000)
    tf_mu, tf_var = gpr_posterior.predict_f(tf_test_times)

    samples_var = tf.square(tf_samples - tf_mu).numpy().mean(0)
    np.testing.assert_allclose(tf_var.numpy(), samples_var, atol=1e-1)


def test_zero_samples_returns_zero_samples(batch_shape, with_tf_random_seed):
    _, batch_test_data = get_test_data(batch_shape)
    mf_gpr = create_markovflow_gpr(batch_test_data.time_points, batch_test_data.observations)
    samples = mf_gpr.posterior.sample_f(
        tf.constant(batch_test_data.future_time_points, dtype=default_float()), sample_shape=0
    )
    assert samples.numpy().size == 0


def test_sample_shapes(batch_shape, with_tf_random_seed):
    _, batch_test_data = get_test_data(batch_shape)
    test_times = tf.constant(batch_test_data.future_time_points, dtype=default_float())
    mf_gpr = create_markovflow_gpr(batch_test_data.time_points, batch_test_data.observations)
    posterior = mf_gpr.posterior
    for sample_shape in [0, 1, 6, (10, 10), (3, 1), (0, 1), (1, 1, 1), (2, 1, 3)]:
        samples = posterior.sample_f(test_times, sample_shape=sample_shape)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        assert samples.shape[:-2] == sample_shape + tuple(batch_shape)


@pytest.fixture(name="ssm_setup")
def _ssm_setup_fixture(batch_shape):
    return _setup(batch_shape)


def _setup(batch_shape, state_dim=3, transitions=5):
    """Create a state space model with a given batch shape."""
    return StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()


def test_pairwise_marginal_means(ssm_setup):
    """Test the pairwise marginal means are correct."""
    ssm, _ = ssm_setup

    initial_mean = np.zeros(shape=tuple(ssm.batch_shape) + (ssm.state_dim,))
    initial_cov = np.zeros(shape=tuple(ssm.batch_shape) + (ssm.state_dim, ssm.state_dim))

    marginal_means = ssm.marginal_means
    pairwise_marginal_means = pairwise_marginals(
        ssm,
        tf.constant(initial_mean, dtype=default_float()),
        tf.constant(initial_cov, dtype=default_float()),
    )[0]
    # create the joint means in numpy
    extended_means = np.concatenate(
        [initial_mean[..., None, :], marginal_means, initial_mean[..., None, :]], axis=-2
    )
    pairwise_means_np = np.concatenate(
        [extended_means[..., :-1, :], extended_means[..., 1:, :]], axis=-1
    )
    np.testing.assert_allclose(pairwise_means_np, pairwise_marginal_means)


def test_pairwise_marginal_covs(ssm_setup):
    """Test the pairwise marginal covariances are correct."""
    ssm, _ = ssm_setup

    initial_mean = np.zeros(shape=tuple(ssm.batch_shape) + (ssm.state_dim,))
    initial_cov = np.zeros(shape=tuple(ssm.batch_shape) + (ssm.state_dim, ssm.state_dim))

    marginal_covs = ssm.marginal_covariances
    subsequent_covs = ssm.subsequent_covariances(marginal_covs)
    pairwise_marginal_covs = pairwise_marginals(
        ssm,
        tf.constant(initial_mean, dtype=default_float()),
        tf.constant(initial_cov, dtype=default_float()),
    )[1]
    # create the joint covariance matrix in numpy
    extended_covs = np.concatenate(
        [initial_cov[..., None, :, :], marginal_covs, initial_cov[..., None, :, :]], axis=-3
    )

    sub_cov_zero = np.zeros_like(initial_cov[..., None, :, :])
    extended_sub_covs = np.concatenate([sub_cov_zero, subsequent_covs, sub_cov_zero], axis=-3)

    joint_cov_0 = np.concatenate(
        [extended_covs[..., :-1, :, :], np.swapaxes(extended_sub_covs, -1, -2)], axis=-1
    )
    joint_cov_1 = np.concatenate([extended_sub_covs, extended_covs[..., 1:, :, :]], axis=-1)
    joint_cov = np.concatenate([joint_cov_0, joint_cov_1], axis=-2)

    np.testing.assert_allclose(joint_cov, pairwise_marginal_covs)
