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
"""Module containing the unit tests for the `HarmonicOscillator` class."""
import numpy as np
import pytest
import tensorflow as tf
from gpflow import default_float

from markovflow.kernels.periodic import HarmonicOscillator
from markovflow.utils import to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points

LENGTH_SCALE = 2.0
VARIANCE = 2.25
PERIOD = 2 * np.pi
OUTPUT_DIM = 1


@pytest.fixture(name="kernel_setup")
def _setup_kernels(batch_shape):
    """Create random time points with batch_shape and a HarmonicOscillator kernel"""
    # batch_shape = request.param
    num_data = 9

    kernel = HarmonicOscillator(variance=VARIANCE, period=PERIOD, output_dim=OUTPUT_DIM)
    time_points = tf.convert_to_tensor(
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        tf.float64,
    )
    return time_points, kernel, batch_shape


def test_zero_variance(with_tf_random_seed):
    """Test that initializing a HarmonicOscillator kernel with 0 covariance raises an exception"""
    with pytest.raises(ValueError) as exp:
        HarmonicOscillator(variance=0, period=PERIOD, output_dim=OUTPUT_DIM)
    assert str(exp.value).find("variance must be positive.") >= 0


def test_zero_period(with_tf_random_seed):
    """Test that initializing a HarmonicOscillator kernel with 0 covariance raises an exception"""
    with pytest.raises(ValueError) as exp:
        HarmonicOscillator(variance=1, period=0)
    assert str(exp.value).find("period must be positive.") >= 0


def test_jitter_negative(with_tf_random_seed):
    """
    Test that initializing a HarmonicOscillator kernel
    with a negative jitter raises and exception.
    """
    with pytest.raises(AssertionError) as exp:
        HarmonicOscillator(variance=0, period=PERIOD, output_dim=OUTPUT_DIM, jitter=-1.0)
    assert str(exp.value).find("jitter must be a non-negative float number.") >= 0


def test_state_transitions(with_tf_random_seed, kernel_setup):
    """Test the state transitions are correct for HarmonicOscillator kernel."""
    time_points, kernel, _ = kernel_setup
    time_deltas = to_delta_time(time_points)
    actual = kernel.state_transitions(time_points[..., :-1], time_deltas)
    deltas = time_deltas[..., None, None]
    expected = tf.concat(
        [
            tf.concat([tf.cos(deltas * kernel._lambda), -tf.sin(deltas * kernel._lambda)], axis=-1),
            tf.concat([tf.sin(deltas * kernel._lambda), tf.cos(deltas * kernel._lambda)], axis=-1),
        ],
        axis=-2,
    )
    actual_np, expected_np = actual.numpy(), expected.numpy()
    np.testing.assert_allclose(actual_np, expected_np)


def test_process_covariances(with_tf_random_seed, kernel_setup):
    """Test the process covariances are correct for HarmonicOscillator kernel."""
    time_points, kernel, _ = kernel_setup
    time_deltas = to_delta_time(time_points)
    process_covariances = kernel.process_covariances(time_points[..., :-1], time_deltas)
    covs_np = np.zeros(tuple(time_deltas.shape.as_list()) + (kernel.state_dim, kernel.state_dim))
    np.testing.assert_allclose(process_covariances, covs_np)


def test_feedback_matrix(with_tf_random_seed, kernel_setup):
    """
    Test the feedback_matrix correspond to the state transitions for HarmonicOscillator kernel.
    """
    _, kernel, _ = kernel_setup

    expected = tf.convert_to_tensor(
        value=[[0, -kernel._lambda], [kernel._lambda, 0]], dtype=default_float()
    )

    actual = kernel.feedback_matrix
    np.testing.assert_allclose(actual, expected)


def test_steady_state(with_tf_random_seed, kernel_setup):
    """Test the steady state covariance is correct for the HarmonicOscillator"""
    _, kernel, _ = kernel_setup

    np.testing.assert_allclose(
        kernel.steady_state_covariance, tf.eye(2, dtype=default_float()) * kernel._variance
    )
