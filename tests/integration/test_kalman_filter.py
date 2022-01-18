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
"""Integration tests for the Kalman Filter by comparing it to a hand-crafted version."""
import numpy as np
import pytest
import tensorflow as tf

from markovflow.emission_model import EmissionModel
from markovflow.kalman_filter import KalmanFilter
from markovflow.state_space_model import StateSpaceModel
from tests.tools.generate_random_objects import (
    generate_random_lower_triangular_matrix,
    generate_random_pos_def_matrix,
)
from tests.tools.numpy_kalman_filter import NumpyKalmanFilter


@pytest.fixture(name="kalman_setup")
def _setup(batch_shape):
    """Create the `NumpyKalmanFilter` and `KalmanFilter` and generate a trajectory for tests."""
    num_transitions = 1 #7
    state_dim = 1
    output_dim = 1 #2
    transition_matrix = np.random.uniform(low=-1., high=1., size=(state_dim, state_dim))
    chol_transition_noise = generate_random_lower_triangular_matrix(state_dim)
    observation_matrix = np.random.normal(size=(output_dim, state_dim))
    observation_noise = generate_random_pos_def_matrix(output_dim) #* 0 + 1e20
    chol_observation_noise = np.linalg.cholesky(observation_noise)
    initial_state_prior_mean = np.random.normal(size=state_dim)
    state_offsets = np.random.normal(size=state_dim)
    chol_initial_state_prior_cov = generate_random_lower_triangular_matrix(state_dim)

    # create the numpy Kalman Filter and generate some trajectories
    np_kalman_filter = NumpyKalmanFilter(
        num_timesteps=num_transitions + 1,
        transition_matrix=transition_matrix,
        transition_mean=state_offsets,
        transition_noise=np.einsum(
            "...ij,...kj->...ik", chol_transition_noise, chol_transition_noise
        ),
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        initial_state_prior_mean=initial_state_prior_mean,
        initial_state_prior_cov=np.einsum(
            "...ij,...kj->...ik", chol_initial_state_prior_cov, chol_initial_state_prior_cov
        ),
    )

    y_train = np_kalman_filter.generate_trajectories(batch_shape)
    y_train = y_train * 0
    # reshape the matrices into the form expected by KalmanFilter
    transition_matrix_tf = tf.constant(
        np.broadcast_to(transition_matrix, batch_shape + (num_transitions, state_dim, state_dim))
    )
    chol_transition_noise_tf = tf.constant(
        np.broadcast_to(
            chol_transition_noise, batch_shape + (num_transitions, state_dim, state_dim)
        )
    )
    initial_state_prior_mean_tf = tf.constant(
        np.broadcast_to(initial_state_prior_mean, batch_shape + (state_dim,))
    )
    state_offsets_tf = tf.constant(
        np.broadcast_to(state_offsets, batch_shape + (num_transitions, state_dim))
    )
    chol_initial_state_prior_cov_tf = tf.constant(
        np.broadcast_to(chol_initial_state_prior_cov, batch_shape + (state_dim, state_dim))
    )

    # create the MarkovFlow KalmanFilter
    ssm = StateSpaceModel(
        initial_state_prior_mean_tf,
        chol_initial_state_prior_cov_tf,
        transition_matrix_tf,
        state_offsets_tf,
        chol_transition_noise_tf,
    )

    emission_matrix = np.tile(
        np.reshape(observation_matrix, (1,) * (len(batch_shape) + 1) + observation_matrix.shape),
        batch_shape + (num_transitions + 1, *np.ones_like(observation_matrix.shape)),
    )
    emission_model = EmissionModel(tf.constant(emission_matrix))

    tf_kalman_filter = KalmanFilter(
        ssm, emission_model, tf.constant(y_train), tf.constant(chol_observation_noise)
    )

    return np_kalman_filter, y_train, tf_kalman_filter


def test_posterior_ssm_means(with_tf_random_seed, kalman_setup):
    """Verify that the posterior means match those of the hand-crafted Kalman Filter."""
    np_kalman_filter, y_train, tf_kalman_filter = kalman_setup

    # run the forward and backwards passes to generate the posterior means and covariances
    _, *ff_np = np_kalman_filter.forward_filter(y_train)
    smooth_mus, _ = np_kalman_filter.backward_smoothing_pass(*ff_np)

    posterior_mu = tf_kalman_filter.posterior_state_space_model().marginal_means

    np.testing.assert_allclose(posterior_mu, smooth_mus)


def test_posterior_ssm_covs(with_tf_random_seed, kalman_setup):
    """Verify that the posterior covariances match those of the hand-crafted Kalman Filter."""
    np_kalman_filter, y_train, tf_kalman_filter = kalman_setup

    # run the forward and backwards passes to generate the posterior means and covariances
    _, *ff_np = np_kalman_filter.forward_filter(y_train)
    _, smooth_covs = np_kalman_filter.backward_smoothing_pass(*ff_np)

    posterior_covs = tf_kalman_filter.posterior_state_space_model().marginal_covariances

    np.testing.assert_allclose(*np.broadcast_arrays(smooth_covs, posterior_covs))


def test_log_likelihood(with_tf_random_seed, kalman_setup):
    """Verify that the likelihoods match those of the hand-crafted Kalman Filter."""
    np_kalman_filter, y_train, tf_kalman_filter = kalman_setup

    ll_np, *_ = np_kalman_filter.forward_filter(y_train)

    ll_tf = tf_kalman_filter.log_likelihood()

    np.testing.assert_allclose(np.sum(ll_np), ll_tf)


def test_kalman_forward_filtering(with_tf_random_seed, kalman_setup):
    """Verify that the filtered means and covariances match those of the hand-crafted Kalman Filter."""
    np_kalman_filter, y_train, tf_kalman_filter = kalman_setup

    # run the forward pass
    _, filter_mus_np, filter_ps_np, pred_mus_np, pred_ps_np = np_kalman_filter.forward_filter(y_train)

    # CURRENTLY COMPUTES THE PREDICTIONS
    filter_mus_tf, filter_ps_tf, pred_mus_tf, pred_ps_tf = tf_kalman_filter.forward_filter()

    np.testing.assert_allclose(filter_mus_np[..., :, :], filter_mus_tf[..., :, :])
    # np.testing.assert_allclose(pred_mus_np[..., :, :], pred_mus_tf[..., :, :])
    np.testing.assert_allclose(filter_ps_np[:, :, :], filter_ps_tf[..., :, :, :])  # numpy ps no batch
    # np.testing.assert_allclose(pred_ps_np[:, :, :], pred_ps_tf[..., :, :, :])  # numpy ps no batch


def test_kalman_backward_filtering(with_tf_random_seed, kalman_setup):
    """Verify that the filtered means and covariances match those of the hand-crafted Kalman Filter."""
    _, y_train, tf_kalman_filter = kalman_setup

    posterior_means, posterior_covs = tf_kalman_filter.posterior_state_space_model().marginals

    ffilter_mus_tf, ffilter_ps_tf, fpred_mus_tf, fpred_ps_tf = tf_kalman_filter.forward_filter()
    bfilter_mus_tf, bfilter_ps_tf, bpred_mus_tf, bpred_ps_tf = tf_kalman_filter.backward_filter()

    np.testing.assert_allclose(posterior_means[..., -1, :], ffilter_mus_tf[..., -1, :])
    np.testing.assert_allclose(posterior_means[..., 0, :], bfilter_mus_tf[..., 0, :])
