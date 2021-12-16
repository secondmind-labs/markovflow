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
"""Module containing the unit tests for the `Matern12` class."""
import numpy as np
import pytest
import tensorflow as tf

from markovflow.base import auto_namescope_enabled
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.kernels.kernel_factories import (
    Matern12TestFactory,
    Matern32TestFactory,
    Matern52TestFactory,
)
from tests.tools.kernels.kernels import DataShape

LENGTH_SCALE = 2.0
VARIANCE = 2.25
OUTPUT_DIM = 1


@pytest.fixture(name="data_fixture")
def _data_fixture(batch_shape):
    """Create random time points with batch_shape"""
    num_data = 9
    data_shape = DataShape(batch_shape, 1)

    return (
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        data_shape,
    )


@pytest.fixture(
    name="kernel_setup", params=[Matern12TestFactory, Matern32TestFactory, Matern52TestFactory]
)
def _setup_fixture(data_fixture, request):
    time_points, data_shape = data_fixture
    sde_kernel_test_factory = request.param
    kernel_factory = sde_kernel_test_factory(LENGTH_SCALE, VARIANCE, data_shape, OUTPUT_DIM)
    time_points = tf.convert_to_tensor(time_points, tf.float64)
    return time_points, kernel_factory, data_shape


def test_kernels_initial_covariance(with_tf_random_seed, kernel_setup):
    """Test the initial covariance is correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup

    initial_covariance = kernel_factory.create_kernel().initial_covariance(time_points[..., 0:1])
    initial_covariance_np = kernel_factory.create_kernel_test().initial_covariance()

    np.testing.assert_allclose(initial_covariance, initial_covariance_np, atol=1e-8)


def test_kernels_initial_mean(with_tf_random_seed, kernel_setup):
    """ Test the initial mean is correct for each kernel."""
    _, kernel_factory, data_shape = kernel_setup

    initial_mean = kernel_factory.create_kernel().initial_mean(data_shape.batch_shape)
    initial_mean_np = kernel_factory.create_kernel_test().initial_mean()

    np.testing.assert_allclose(initial_mean, initial_mean_np, atol=1e-8)


def test_kernels_state_transitions(with_tf_random_seed, kernel_setup):
    """Test the state transitions are correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup
    time_deltas = time_points[1:] - time_points[:-1]
    state_transitions = kernel_factory.create_kernel().state_transitions(time_points, time_deltas)
    state_transitions_np = kernel_factory.create_kernel_test().state_transitions(
        time_points.numpy(), time_deltas.numpy()
    )

    np.testing.assert_allclose(state_transitions, state_transitions_np, atol=1e-8)


def test_kernels_process_covariances(with_tf_random_seed, kernel_setup):
    """Test the process covariances are correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup
    time_deltas = time_points[1:] - time_points[:-1]
    kernel = kernel_factory.create_kernel()
    process_covariances = kernel.process_covariances(time_points, time_deltas)
    process_covariances_np = kernel_factory.create_kernel_test().process_covariances(
        time_points.numpy(), time_deltas.numpy()
    )

    np.testing.assert_allclose(process_covariances, process_covariances_np, atol=1e-8)


def test_kernels_feedback_matrix(with_tf_random_seed, kernel_setup):
    """Test the feedback_matrices correspond to the state transitions for each kernel."""
    time_points, kernel_factory, _ = kernel_setup

    kernel = kernel_factory.create_kernel()
    time_deltas = time_points[..., 1:] - time_points[..., :-1]
    state_transitions = kernel.state_transitions(time_points, time_deltas)
    state_transitions_feedback = tf.linalg.expm(
        kernel.feedback_matrix * time_deltas[..., None, None]
    )

    np.testing.assert_allclose(state_transitions, state_transitions_feedback, atol=1e-8)


def test_kernels_steady_state_covariance(with_tf_random_seed, kernel_setup):
    """Test the steady state covariances are correct for each kernel."""
    _, kernel_factory, _ = kernel_setup

    steady_state_covariance = kernel_factory.create_kernel().steady_state_covariance
    steady_state_covariance_np = kernel_factory.create_kernel_test().steady_state_covariance()

    np.testing.assert_allclose(steady_state_covariance, steady_state_covariance_np, atol=1e-8)


def test_matern_kernels_trainable_variables(with_tf_random_seed, kernel_setup):
    """ Test that the matern trainable parameters exist and are correctly scoped """
    _, kernel_factory, _ = kernel_setup
    kernel = kernel_factory.create_kernel()
    trainables = kernel.trainable_variables
    basename = f"{kernel.__class__.__name__}.__init__/" if auto_namescope_enabled() else ""
    assert any(basename + "lengthscale" in v.name for v in trainables)
    assert any(basename + "variance" in v.name for v in trainables)


def test_can_generate_state_space_model(with_tf_random_seed, kernel_setup):
    """
    Test that we can generate a state space model from the kernel by passing in time points as
    numpy arrays or tensors
    """
    time_points, kernel_factory, _ = kernel_setup

    kernel = kernel_factory.create_kernel()
    kernel.state_space_model(time_points)
    kernel.state_space_model(tf.identity(time_points))


def test_matern_kernels_state_mean(with_tf_random_seed, kernel_setup):
    """Test the Matern kernel state mean"""
    _, kernel_factory, _ = kernel_setup
    kernel = kernel_factory.create_kernel()

    np.testing.assert_allclose(
        kernel.state_mean, tf.zeros([kernel.state_dim], dtype=tf.float64)
    )

    updated_state_mean = tf.ones([kernel.state_dim], dtype=tf.float64)
    kernel.set_state_mean(updated_state_mean)

    np.testing.assert_allclose(
        kernel.state_mean, updated_state_mean
    )
