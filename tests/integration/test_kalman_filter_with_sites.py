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
from markovflow.kalman_filter import KalmanFilterWithSites, UnivariateGaussianSitesNat
from markovflow.state_space_model import StateSpaceModel
from tests.integration.test_kalman_filter import (
    test_log_likelihood,
    test_posterior_ssm_covs,
    test_posterior_ssm_means,
)
from tests.tools.generate_random_objects import (
    generate_random_lower_triangular_matrix,
    generate_random_pos_def_matrix,
)
from tests.tools.numpy_kalman_filter import NumpyKalmanFilterWithSites

np.random.seed(1)

# todo: allow for broadcasting dimension (batch_shape)
batch_shape = ()


@pytest.fixture(name="kalman_setup")
def _setup():
    """Create the `NumpyKalmanFilterWithSites` and `NumpyKalmanFilterWithSites`
    and generate a trajectory for tests."""
    num_transitions = 6
    state_dim = 2
    output_dim = 1

    transition_matrix = np.random.normal(size=(state_dim, state_dim))
    chol_transition_noise = generate_random_lower_triangular_matrix(state_dim)
    means = np.random.normal(size=batch_shape + (num_transitions + 1, output_dim))  # * 0
    observation_matrix = np.random.normal(size=(output_dim, state_dim))
    covariances = generate_random_pos_def_matrix(
        output_dim, batch_shape + (num_transitions + 1,)
    )  # * 0 + tf.eye(output_dim, dtype=tf.float64)
    initial_state_prior_mean = np.random.normal(size=state_dim) * 0.0
    state_offsets = np.random.normal(size=state_dim) * 0.0
    chol_initial_state_prior_cov = generate_random_lower_triangular_matrix(state_dim)

    # create the numpy Kalman Filter and generate some trajectories
    np_kalman_filter = NumpyKalmanFilterWithSites(
        num_timesteps=num_transitions + 1,
        transition_matrix=transition_matrix,
        transition_mean=state_offsets,
        transition_noise=np.einsum(
            "...ij,...kj->...ik", chol_transition_noise, chol_transition_noise
        ),
        observation_covariances=covariances,
        observation_means=means,
        observation_matrix=observation_matrix,
        initial_state_prior_mean=initial_state_prior_mean,
        initial_state_prior_cov=np.einsum(
            "...ij,...kj->...ik", chol_initial_state_prior_cov, chol_initial_state_prior_cov
        ),
    )

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

    # create sites parameterized in natural form
    sites = UnivariateGaussianSitesNat(nat1=means / covariances[..., 0], nat2=-0.5 / covariances)
    tf_kalman_filter = KalmanFilterWithSites(ssm, emission_model, sites)

    return np_kalman_filter, means, tf_kalman_filter


def _test_kalman_with_sites(with_tf_random_seed, kalman_setup):
    # Verify that the posterior means match those of the hand-crafted Kalman Filter.
    test_posterior_ssm_means(with_tf_random_seed, kalman_setup)
    # Verify that the posterior covariances match those of the hand-crafted Kalman Filter.
    test_posterior_ssm_covs(with_tf_random_seed, kalman_setup)
    # Verify that the likelihoods match those of the hand-crafted Kalman Filter.
    test_log_likelihood(with_tf_random_seed, kalman_setup)
