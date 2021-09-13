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
"""Module containing the unit tests for the `Constant` class."""
import numpy as np
import pytest
import tensorflow as tf

from markovflow.kernels.constant import Constant
from markovflow.utils import to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points

LENGTH_SCALE = 2.0
VARIANCE = 2.25
OUTPUT_DIM = 1


@pytest.fixture(name="kernel_setup")
def _setup_kernels(batch_shape):
    """Create random time points with batch_shape and a Constant kernel"""
    # batch_shape = request.param
    num_data = 9

    kernel = Constant(variance=VARIANCE, output_dim=OUTPUT_DIM)
    return (
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        kernel,
        batch_shape,
    )


def test_constant_kernel_zero_variance(with_tf_random_seed):
    """Test that initializing a constant kernel with 0 covariance raises an exception"""

    with pytest.raises(ValueError) as exp:
        Constant(variance=0, output_dim=OUTPUT_DIM)
    assert str(exp.value).find("variance must be positive.") >= 0


def test_jitter_negative(with_tf_random_seed):
    """Test that initializing a constant kernel with a negative jitter raises and exception."""

    with pytest.raises(AssertionError) as exp:
        Constant(variance=0, output_dim=OUTPUT_DIM, jitter=-1.0)
    assert str(exp.value).find("jitter must be a non-negative float number.") >= 0


def test_constant_initial_covariance(with_tf_random_seed, kernel_setup):
    """Test the initial covariance is correct for Constant."""
    _, kernel, batch_shape = kernel_setup

    # Constructed expected initial covariance.
    initial_covariance_np = np.zeros(batch_shape + (kernel.state_dim, kernel.state_dim))
    initial_covariance_np[..., 0, 0] = VARIANCE

    fake_initial_time_points = tf.zeros(batch_shape)[..., None]

    # Compute actual initial covariance and compare with the expected initial covariance.
    initial_covariance = kernel.initial_covariance(fake_initial_time_points)
    np.testing.assert_allclose(initial_covariance, initial_covariance_np)


def test_constant_initial_mean(with_tf_random_seed, kernel_setup):
    """ Test the initial mean is zero for Constant kernel."""
    _, kernel, batch_shape = kernel_setup

    initial_mean = kernel.initial_mean(batch_shape)
    initial_mean_np = np.zeros(batch_shape + (kernel.state_dim,))

    np.testing.assert_allclose(initial_mean, initial_mean_np)


def test_constant_state_transitions(with_tf_random_seed, kernel_setup):
    """Test the state transitions are correct for Constant kernel."""
    time_points, kernel, _ = kernel_setup
    time_deltas = to_delta_time(time_points)
    state_transitions = kernel.state_transitions(time_points, time_deltas)

    expected_transitions = np.zeros(time_deltas.shape + (kernel.state_dim, kernel.state_dim))
    expected_transitions[..., 0, 0] = 1.0

    np.testing.assert_allclose(state_transitions, expected_transitions)


def test_constant_process_covariances(with_tf_random_seed, kernel_setup):
    """Test the process covariances are correct for Constant kernel."""
    time_points, kernel, _ = kernel_setup
    time_deltas = to_delta_time(time_points)
    process_covariances = kernel.process_covariances(time_points, time_deltas)
    covs_np = np.zeros(time_deltas.shape + (kernel.state_dim, kernel.state_dim))
    np.testing.assert_allclose(process_covariances, covs_np)


def test_constant_feedback_matrix(with_tf_random_seed, kernel_setup):
    """Test the feedback_matrix correspond to the state transitions for constant kernel."""
    time_points, kernel, _ = kernel_setup
    time_deltas = to_delta_time(time_points)
    state_transitions = kernel.state_transitions(time_points, time_deltas)
    state_transitions_feedback = tf.linalg.expm(
        kernel.feedback_matrix * time_deltas[..., None, None]
    )

    np.testing.assert_allclose(state_transitions, state_transitions_feedback)


def test_constant_steady_state(with_tf_random_seed, kernel_setup):
    """Test the steady state covariance is correct for the constant"""
    _, kernel, _ = kernel_setup
    np.testing.assert_allclose(
        kernel.steady_state_covariance, tf.identity([[1.0 * kernel._variance]])
    )
