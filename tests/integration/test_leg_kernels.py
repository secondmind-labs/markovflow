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
"""Module containing the unit tests for the `LatentExponentiallyGenerated` class."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from scipy.linalg import expm

from markovflow.kernels import LatentExponentiallyGenerated
from markovflow.utils import to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points

STATE_DIM = 3


@dataclass
class DataShape:
    batch_shape: Tuple[int, ...]
    time_dim: int


class LatentExponentiallyGeneratedTest:
    def __init__(self, N, R, data_shape: DataShape) -> None:
        self._N = N
        self._R = R
        self._F = -(N @ N.T + R - R.T) * 0.5
        self.state_dim = N.shape[-1]
        self._data_shape = data_shape

    def initial_covariance(self):
        return self.steady_state_covariance() * np.ones(
            self._data_shape.batch_shape + (self.state_dim, self.state_dim)
        )

    def initial_mean(self):
        return np.zeros(self._data_shape.batch_shape + (self.state_dim,))

    def state_transitions(self, time_deltas):
        F_ts = self._F * time_deltas[..., None, None]

        # The scipy `expm` function does not support broadcasting over batches of matrices.
        state_transitions = np.zeros_like(F_ts)
        for index in np.ndindex(time_deltas.shape):
            state_transitions[index] = expm(F_ts[index])

        return state_transitions

    def steady_state_covariance(self):
        return np.eye(self.state_dim)

    def process_covariances(self, time_deltas):
        state_transitions = self.state_transitions(time_deltas)

        A_P0_A_T = np.einsum(
            "...ij,jk,...lk", state_transitions, self.steady_state_covariance(), state_transitions
        )
        return self.steady_state_covariance() - A_P0_A_T


class LegTestFactory:
    def __init__(self, N, R, data_shape: DataShape):
        self._N = N
        self._R = R
        self._data_shape = data_shape

    def create_kernel(self):
        return LatentExponentiallyGenerated(N=self._N, R=self._R)

    def create_kernel_test(self):
        return LatentExponentiallyGeneratedTest(N=self._N, R=self._R, data_shape=self._data_shape)


@pytest.fixture(name="data_fixture")
def _data_fixture(batch_shape):
    """Create random time points with batch_shape"""
    num_data = 9
    data_shape = DataShape(batch_shape, 1)

    return (
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        data_shape,
    )


@pytest.fixture(name="kernel_setup", params=[LegTestFactory])
def _setup_fixture(data_fixture, request):
    time_points, data_shape = data_fixture
    sde_kernel_test_factory = request.param

    R = np.random.rand(STATE_DIM, STATE_DIM)
    N = np.random.rand(STATE_DIM, STATE_DIM)
    kernel_factory = sde_kernel_test_factory(N, R, data_shape)

    return time_points, kernel_factory, data_shape


def test_kernels_initial_covariance(with_tf_random_seed, kernel_setup):
    """Test the initial covariance is correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup

    initial_time_points = time_points[..., 0:1]
    initial_covariance = kernel_factory.create_kernel().initial_covariance(initial_time_points)
    initial_covariance_np = kernel_factory.create_kernel_test().initial_covariance()

    np.testing.assert_allclose(initial_covariance, initial_covariance_np)


def test_kernels_initial_mean(with_tf_random_seed, kernel_setup):
    """ Test the initial mean is correct for each kernel."""
    _, kernel_factory, data_shape = kernel_setup

    initial_mean = kernel_factory.create_kernel().initial_mean(data_shape.batch_shape)
    initial_mean_np = kernel_factory.create_kernel_test().initial_mean()

    np.testing.assert_allclose(initial_mean, initial_mean_np)


def test_kernels_state_transitions(with_tf_random_seed, kernel_setup):
    """Test the state transitions are correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup
    time_deltas = time_points[..., 1:] - time_points[..., :-1]

    tf_time_points = tf.constant(time_points)
    tf_time_deltas = to_delta_time(tf_time_points)

    state_transitions = kernel_factory.create_kernel().state_transitions(
        tf_time_points, tf_time_deltas
    )
    state_transitions_np = kernel_factory.create_kernel_test().state_transitions(time_deltas)

    np.testing.assert_allclose(state_transitions, state_transitions_np)


def test_kernels_process_covariances(with_tf_random_seed, kernel_setup):
    """Test the process covariances are correct for each kernel."""
    time_points, kernel_factory, _ = kernel_setup
    time_deltas = time_points[..., 1:] - time_points[..., :-1]

    tf_time_points = tf.constant(time_points)
    tf_time_deltas = to_delta_time(tf_time_points)
    kernel = kernel_factory.create_kernel()
    state_transitions = kernel.state_transitions(tf_time_points, tf_time_deltas)
    process_covariances = kernel.process_covariances(state_transitions, tf_time_deltas)

    process_covariances_np = kernel_factory.create_kernel_test().process_covariances(time_deltas)

    np.testing.assert_allclose(process_covariances, process_covariances_np)


def test_kernels_feedback_matrix(with_tf_random_seed, kernel_setup):
    """Test the feedback_matrices correspond to the state transitions for each kernel."""
    time_points, kernel_factory, _ = kernel_setup

    tf_time_points = tf.constant(time_points)
    tf_time_deltas = to_delta_time(tf_time_points)

    kernel = kernel_factory.create_kernel()
    state_transitions = kernel.state_transitions(tf_time_points, tf_time_deltas)
    state_transitions_feedback = tf.linalg.expm(
        kernel.feedback_matrix * tf_time_deltas[..., None, None]
    )

    np.testing.assert_allclose(state_transitions, state_transitions_feedback)


def test_kernels_steady_state_covariance(with_tf_random_seed, kernel_setup):
    """Test the steady state covariances are correct for each kernel."""
    _, kernel_factory, _ = kernel_setup

    steady_state_covariance = kernel_factory.create_kernel().steady_state_covariance
    steady_state_covariance_np = kernel_factory.create_kernel_test().steady_state_covariance()

    np.testing.assert_allclose(steady_state_covariance, steady_state_covariance_np)


def test_leg_kernels_trainable_variables(with_tf_random_seed, kernel_setup):
    """ Test that the LEG trainable parameters exist and are correctly scoped """
    _, kernel_factory, _ = kernel_setup
    kernel = kernel_factory.create_kernel()
    trainables = kernel.trainable_variables
    assert any(f"{kernel.__class__.__name__}/R" in v.name for v in trainables)
    assert any(f"{kernel.__class__.__name__}/N" in v.name for v in trainables)
