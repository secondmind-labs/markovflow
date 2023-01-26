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
"""Module containing the integration tests for the `Kernel` class."""
import numpy as np
import pytest
import tensorflow as tf
import gpflow
from gpflow.likelihoods import Gaussian

from markovflow.likelihoods import MultivariateGaussian
from tests.tools.generate_random_objects import (
    generate_random_pos_def_matrix,
    generate_random_time_observations,
)

FLOAT_TYPE = tf.float64


@pytest.fixture(name="data")
def _setup_data(with_tf_random_seed, batch_shape):
    """Create data with batch_shape."""
    obs_dim = 1
    num_data = 9
    _, observations = generate_random_time_observations(obs_dim, num_data, batch_shape)
    _, f = generate_random_time_observations(obs_dim, num_data, batch_shape)
    shape = tuple(list(batch_shape) + [num_data])
    f_covariances = generate_random_pos_def_matrix(obs_dim, shape)
    return f, f_covariances, observations


def test_multivariate_gaussian_trainable_parameters():
    """Check that MultivariateGaussian tracks its trainable_parameters."""
    chol_covariance = np.eye(2)
    mvngauss = MultivariateGaussian(chol_covariance=chol_covariance)
    assert isinstance(mvngauss, gpflow.Module)
    assert mvngauss.trainable_parameters == (mvngauss.chol_covariance,)
    gpflow.set_trainable(mvngauss.chol_covariance, False)
    assert mvngauss.trainable_parameters == ()


def test_multivariate_gaussian_variational_expectations(data):
    """
    Test multivariate_likelihood variational_expectations
    Univariate Gaussian against multivariate
    """

    # data with output dim  = 1
    f_np, f_covs_np, Y_np = data
    f, f_covs, Y = tf.constant(f_np), tf.constant(f_covs_np), tf.constant(Y_np)
    # data with output dim = 2 -> stacking f and y, block diag the covariances
    f1 = tf.concat([f, f + 1], axis=-1)
    Y1 = tf.concat([Y, Y + 1], axis=-1)
    f_covs1 = tf.linalg.diag(tf.concat([f_covs[..., 0], f_covs[..., 0]], axis=-1))
    f_vars = tf.linalg.diag_part(f_covs)

    # constructing the likelihood objects
    unigauss = Gaussian(1.0)  # univariate normal
    mvngauss = MultivariateGaussian(chol_covariance=np.eye(1))  # 1d mvn
    mvngauss1 = MultivariateGaussian(chol_covariance=np.eye(2))  # 2d mvn

    uni_ve = unigauss.variational_expectations(f, f_vars, Y)
    mvn_ve = mvngauss.variational_expectations(f, f_covs, Y)
    mvn1_ve = mvngauss1.variational_expectations(f1, f_covs1, Y1)

    # testing univariate against multivariate (dim=1)
    np.testing.assert_allclose(uni_ve, mvn_ve, rtol=1e-5)

    # testing 2*univariate against multivariate with identity covariance
    np.testing.assert_allclose(2 * uni_ve, mvn1_ve, rtol=1e-5)


def test_multivariate_gaussian_log_probability_density(data):
    """
    Test multivariate_likelihood log_probability_density
    Univariate Gaussian against multivariate
    """

    # data with output dim  = 1
    f_np, _, Y_np = data
    f, Y = tf.constant(f_np), tf.constant(Y_np)
    # data with output dim = 2 -> stacking f and y, block diag the covariances
    f1 = tf.concat([f, f + 1], axis=-1)
    Y1 = tf.concat([Y, Y + 1], axis=-1)

    # constructing the likelihood objects
    unigauss = Gaussian(1.0)  # univariate normal
    mvngauss = MultivariateGaussian(chol_covariance=np.eye(1))  # 1d mvn
    mvngauss1 = MultivariateGaussian(chol_covariance=np.eye(2))  # 2d mvn

    uni_lp = unigauss.log_prob(f, Y)
    mvn_lp = mvngauss.log_probability_density(f, Y)
    mvn1_lp = mvngauss1.log_probability_density(f1, Y1)

    # testing univariate against multivariate (dim=1)
    np.testing.assert_allclose(uni_lp, mvn_lp, rtol=1e-5)

    # testing 2*univariate against multivariate with identity covariance
    np.testing.assert_allclose(2 * uni_lp, mvn1_lp, rtol=1e-5)


def test_multivariate_gaussian_predict_density(data):
    """
    Test multivariate_likelihood predict_density
    Univariate Gaussian against multivariate
    """

    # data with output dim  = 1
    f_np, f_covs_np, Y_np = data
    f, f_covs, Y = tf.constant(f_np), tf.constant(f_covs_np), tf.constant(Y_np)
    f_vars = tf.linalg.diag_part(f_covs)

    # data with output dim = 2 -> stacking f and y, block diag the covariances
    f1 = tf.concat([f, f + 1], axis=-1)
    Y1 = tf.concat([Y, Y + 1], axis=-1)
    f_covs1 = tf.linalg.diag(tf.concat([f_covs[..., 0], f_covs[..., 0]], axis=-1))

    # constructing the likelihood objects
    unigauss = Gaussian(1.0)  # univariate normal
    mvngauss = MultivariateGaussian(chol_covariance=np.eye(1))  # 1d mvn
    mvngauss1 = MultivariateGaussian(chol_covariance=np.eye(2))  # 2d mvn

    uni_pd = unigauss.predict_density(f, f_vars, Y)
    mvn_pd = mvngauss.predict_density(f, f_covs, Y)
    mvn1_pd = mvngauss1.predict_density(f1, f_covs1, Y1)

    # testing univariate against multivariate (dim=1)
    np.testing.assert_allclose(uni_pd, mvn_pd, rtol=1e-5)

    # testing 2*univariate against multivariate with identity covariance
    np.testing.assert_allclose(2 * uni_pd, mvn1_pd, rtol=1e-5)


def test_predict_y_multivariate_mean(with_tf_random_seed):
    covariance = generate_random_pos_def_matrix(2)
    chol_covariance = np.linalg.cholesky(covariance)
    dist = MultivariateGaussian(chol_covariance=chol_covariance)
    f_mu = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    f_cov = np.array([[[0.0, 1.0], [2, 3]], [[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]])

    y_mus, _ = dist.predict_mean_and_var(tf.constant(f_mu), tf.constant(f_cov))
    np.testing.assert_allclose(y_mus, f_mu, atol=1e-7)


def test_predict_y_multivariate_covar(with_tf_random_seed):
    covariance = generate_random_pos_def_matrix(2)
    chol_covariance = np.linalg.cholesky(covariance)
    dist = MultivariateGaussian(chol_covariance=chol_covariance)
    f_mu = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    f_cov = np.array([[[0.0, 1.0], [2, 3]], [[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]])

    _, y_cov = dist.predict_mean_and_var(tf.constant(f_mu), tf.constant(f_cov))
    np.testing.assert_allclose(y_cov, f_cov + covariance, atol=1e-7)
