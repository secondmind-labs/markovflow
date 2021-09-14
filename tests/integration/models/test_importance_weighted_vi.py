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
"""Module containing the integration tests for the `ImportanceWeightedVI` class."""
import warnings

import numpy as np
import pytest
import tensorflow as tf
from gpflow.likelihoods import Gaussian

from markovflow.kernels import IndependentMultiOutputStack, Matern12
from markovflow.mean_function import LinearMeanFunction
from markovflow.models import ImportanceWeightedVI, SparseVariationalGaussianProcess
from tests.tools.check_distributions import (
    assert_samples_close_in_expectation,
    assert_samples_close_to_mean_in_expectation,
)

LENGTH_SCALE = 0.82
VARIANCE = 0.7
NUM_DATA = 3


@pytest.fixture(name="iwvi_svgp_setup")
def _iwvi_svgp_setup(output_dim):
    time_points, observations = _data_setup(output_dim)
    # set every other time_point (starting from the first one) as inducing point location
    Z = time_points[..., ::2]

    # kernel setup
    if output_dim == 1:
        kernel = Matern12(lengthscale=LENGTH_SCALE, variance=VARIANCE, jitter=1e-6)
    else:
        kern_list = [
            Matern12(lengthscale=LENGTH_SCALE, variance=VARIANCE, jitter=1e-6)
            for _ in range(output_dim)
        ]
        kernel = IndependentMultiOutputStack(kern_list)

    # likelihood and mean_function setup
    likelihood = Gaussian(variance=1e-1)
    mean_function = LinearMeanFunction(1.5)

    # models setup
    svgp = SparseVariationalGaussianProcess(
        kernel=kernel, likelihood=likelihood, inducing_points=Z, mean_function=mean_function
    )

    iwvi = ImportanceWeightedVI(
        kernel=kernel,
        inducing_points=Z,
        likelihood=likelihood,
        mean_function=mean_function,
        num_importance_samples=1,
    )

    # set every other point (starting from the second one) as observed data
    return iwvi, svgp, (time_points[..., 1::2], observations[..., 1::2, :])


def _data_setup(output_dim: int):
    time_points = np.linspace(0.0, 1.0, NUM_DATA)
    # setup for batch_shape, where batch_shape is basically the output_dim
    if output_dim > 1:
        time_points = np.broadcast_to(time_points, (output_dim, NUM_DATA))

    observations = np.random.randn(NUM_DATA, output_dim)
    time_points = tf.constant(time_points)
    observations = tf.constant(observations)

    return time_points, observations


def test_iwvi_vs_vi_elbo(with_tf_random_seed, iwvi_svgp_setup):
    """
    Test that the IWVI bound with K=1 importance sample gives the same ELBO as standard VI.

    Tested with different output dimensions and with a mean function.
    """
    iwvi, svgp, input_data = iwvi_svgp_setup

    iwvi_elbo_fn = tf.function(iwvi.elbo)
    iwvi_elbo_samples = np.array([iwvi_elbo_fn(input_data) for _ in range(1000)])
    vi_elbo = svgp.elbo(input_data)

    assert_samples_close_to_mean_in_expectation(iwvi_elbo_samples, vi_elbo)


def test_iwvi_elbo_increase_with_more_samples(with_tf_random_seed, iwvi_svgp_setup):
    """
    Test that the IWVI bound increases as we increase the number of importance samples.

    Tested with different output dimensions and with a mean function.
    """
    iwvi, _, input_data = iwvi_svgp_setup

    iwvi_elbo_fn1 = tf.function(iwvi.elbo)
    iwvi_elbo_1 = np.mean([iwvi_elbo_fn1(input_data) for _ in range(100)])

    iwvi.posterior.num_importance_samples = 10
    iwvi_elbo_fn10 = tf.function(iwvi.elbo)
    iwvi_elbo_10 = np.mean([iwvi_elbo_fn10(input_data) for _ in range(100)])

    iwvi.posterior.num_importance_samples = 50
    iwvi_elbo_fn50 = tf.function(iwvi.elbo)
    iwvi_elbo_50 = np.mean([iwvi_elbo_fn50(input_data) for _ in range(100)])

    assert iwvi_elbo_1 < iwvi_elbo_10
    assert iwvi_elbo_10 < iwvi_elbo_50


def test_iwvi_vs_vi_posterior_samples(with_tf_random_seed, iwvi_svgp_setup):
    """
    Test that the IWVI model with K=1 importance sample gives the same posterior samples
    (in expectation) as the VI model.

    Tested with different output dimensions and with a mean function.
    """
    iwvi, svgp, input_data = iwvi_svgp_setup

    iwvi_fn_1 = iwvi.posterior.sample_f
    iwvi_posterior_samples = np.stack(iwvi_fn_1(input_data[0], 1000, input_data=input_data), axis=0)

    vi_posterior_samples = svgp.posterior.sample_f(input_data[0], 1000)

    assert_samples_close_in_expectation(iwvi_posterior_samples, vi_posterior_samples)


def test_iwvi_posterior_mean_vs_sampling_mean(with_tf_random_seed, iwvi_svgp_setup):
    """
    Test that the IWVI model gives the same posterior mean (in expectation) as averaging
    posterior samples.

    Tested with different output dimensions and with a mean function.
    """
    N_SAMPLES = 5000
    iwvi, _, input_data = iwvi_svgp_setup

    iwvi_fn_1 = tf.function(iwvi.posterior.expected_value)
    iwvi_fn_2 = iwvi.posterior.sample_f

    iwvi_posterior_mean = np.stack(
        [iwvi_fn_1(input_data[0], input_data=input_data) for _ in range(N_SAMPLES)], axis=0
    )
    iwvi_posterior_sampling = np.stack(
        iwvi_fn_2(input_data[0], N_SAMPLES, input_data=input_data), axis=0
    )

    assert_samples_close_in_expectation(iwvi_posterior_mean, iwvi_posterior_sampling)


def test_dregs(with_tf_random_seed, iwvi_svgp_setup):
    """
    Tests that the gradient computed by the DREGS method is unbiased and of lower variance
    than the standard gradient of the iw-elbo.

    Tested with different output dimensions and with a mean function.
    """
    N_SAMPLES = 5000
    iwvi, _, input_data = iwvi_svgp_setup

    @tf.function
    def _grads():
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(iwvi.dist_q.trainable_variables)
            dregs_obj = iwvi.dregs_objective(input_data)
            iwvi_elbo = iwvi.elbo(input_data)
        dregs_grads = tape.gradient(dregs_obj, iwvi.dist_q.trainable_variables)
        dregs_grads = tf.concat([tf.reshape(g, (-1,)) for g in dregs_grads], axis=0)
        iw_grads = tape.gradient(iwvi_elbo, iwvi.dist_q.trainable_variables)
        iw_grads = tf.concat([tf.reshape(g, (-1,)) for g in iw_grads], axis=0)
        return tf.concat([dregs_grads[:, None], iw_grads[:, None]], axis=1)

    grads = tf.stack([_grads() for _ in range(N_SAMPLES)])
    dreg_grad_evals, iwvi_grad_evals = grads[..., 0].numpy(), grads[..., 1].numpy()

    # test that the means are close
    SIGMA = 3
    assert_samples_close_in_expectation(dreg_grad_evals, iwvi_grad_evals, sigma=SIGMA)

    # test that the variance is lower
    dreg_std = dreg_grad_evals.std(0)
    iwvi_std = iwvi_grad_evals.std(0)

    diff = dreg_std - iwvi_std
    # this assumes the difference in the variance of the gradients are perfectly correlated.
    # assuming uncorrelated instead you should multiply the score by `np.sqrt(len(diff))`
    score = np.mean(diff) / np.std(diff)
    # this will skip most of the time unless you set `N_SAMPLES` very high
    if score < SIGMA:
        warnings.warn("dregs std and iwvi std are too close to each other to check which is lower")
    else:
        assert np.mean(dreg_std) < np.mean(iwvi_std)
