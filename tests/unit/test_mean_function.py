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
"""Module containing the unit tests for the `MeanFunction` class."""
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import gpflow
from gpflow import default_float

from markovflow.kernels import Matern32
from markovflow.mean_function import (
    ImpulseMeanFunction,
    LinearMeanFunction,
    StepMeanFunction,
    ZeroMeanFunction,
)
from tests.tools.generate_random_objects import generate_random_time_points

EXPECTED_RANGE = 12.0


@pytest.fixture(name="impulse_setup")
def impulse_setup_fixture(batch_shape):
    kernel, action_times, state_perturbations = _setup(batch_shape)
    mean_func = ImpulseMeanFunction(
        action_times=action_times, state_perturbations=state_perturbations, kernel=kernel
    )
    f_perturbations = kernel.generate_emission_model(action_times).project_state_to_f(
        state_perturbations
    )
    return mean_func, action_times, f_perturbations


@pytest.fixture(name="step_setup")
def step_setup_fixture(batch_shape):
    kernel, action_times, state_perturbations = _setup(batch_shape)
    mean_func = StepMeanFunction(
        action_times=action_times, state_perturbations=state_perturbations, kernel=kernel
    )
    f_perturbations = kernel.generate_emission_model(action_times).project_state_to_f(
        state_perturbations
    )
    return mean_func, action_times, f_perturbations


def _setup(batch_shape: Tuple):
    """Setup function that creates a kernel and random perturbations at random times."""
    num_impulses = 5

    kernel = Matern32(lengthscale=1.0, variance=1.0, output_dim=1)

    state_perturbations = tf.random.normal(
        batch_shape + (num_impulses, kernel.state_dim), dtype=default_float()
    )

    action_times = generate_random_time_points(
        expected_range=EXPECTED_RANGE, shape=batch_shape + (num_impulses,)
    )

    return kernel, action_times, state_perturbations


def test_around_impulse(with_tf_random_seed, impulse_setup):
    """Test that the state just before and just after an impulse differ by the impulse."""
    mean_func, action_times, f_perturbations = impulse_setup
    pre_vals = mean_func(action_times - 1e-15) + f_perturbations
    post_vals = mean_func(action_times + 1e-15)

    np.testing.assert_allclose(pre_vals, post_vals, atol=1e-10)


def test_on_impulse(with_tf_random_seed, impulse_setup):
    """Test that the state at the point of the impulse hasn't changed"""
    mean_func, action_times, _ = impulse_setup
    pre_vals = mean_func(action_times - 1e-15)
    post_vals = mean_func(action_times)

    np.testing.assert_allclose(pre_vals, post_vals, atol=1e-10)


def test_far_from_impulse(with_tf_random_seed, impulse_setup):
    """Test that the state far from the impulse is zero."""
    mean_func, action_times, _ = impulse_setup
    far_vals = mean_func(action_times + 10 * EXPECTED_RANGE)

    np.testing.assert_allclose(far_vals, 0.0, atol=1e-10)


def test_before_impulse(with_tf_random_seed, impulse_setup):
    """Test that the state before any impulses is zero."""
    mean_func, action_times, _ = impulse_setup
    before_vals = mean_func(action_times[..., :1] - 1e-6)

    np.testing.assert_allclose(before_vals, 0.0, atol=1e-10)


def test_around_step(with_tf_random_seed, step_setup):
    """Test that the state just before and just after a step don't differ."""
    mean_func, action_times, _ = step_setup
    pre_vals = mean_func(action_times - 1e-15)
    post_vals = mean_func(action_times + 1e-15)

    np.testing.assert_allclose(pre_vals, post_vals, atol=1e-10)


def test_far_from_step(with_tf_random_seed, batch_shape):
    """Test that the state far from each step is the step times the inverse feedback matrix."""
    float_type = default_float()

    num_impulses = 3
    delta = 100.0

    # [-10, 90, 190]
    action_times = tf.broadcast_to(
        tf.range(0.0, num_impulses * delta - 1, delta, dtype=float_type) - 10,
        batch_shape + (num_impulses,),
    )

    kernel = Matern32(lengthscale=3.0, variance=1.0, output_dim=1)

    state_perturbations = tf.random.normal(
        batch_shape + (num_impulses, kernel.state_dim), dtype=float_type
    )

    mean_func = StepMeanFunction(
        action_times=action_times, state_perturbations=state_perturbations, kernel=kernel
    )

    # query points a long way away from each step [89, 189, 289]
    time_points = action_times + delta - 1

    F_inv = tf.linalg.inv(kernel.feedback_matrix)
    exp_far_vals = kernel.generate_emission_model(action_times).project_state_to_f(
        tf.einsum("ij,...j->...i", -F_inv, state_perturbations)
    )

    far_vals = mean_func(time_points)

    np.testing.assert_allclose(far_vals, exp_far_vals, atol=1e-10)


def test_before_step(with_tf_random_seed, step_setup):
    """Test that the state before any step is zero."""
    mean_func, action_times, _ = step_setup
    before_vals = mean_func(action_times[..., :1] - 1e-6)

    np.testing.assert_allclose(before_vals, 0.0, atol=1e-10)


def test_impulse(with_tf_random_seed):
    """Test a single impulse against a known function."""
    float_type = default_float()

    state_perturbations = np.array([[2.3, 1.2]])

    action_times = tf.constant([0.0], dtype=float_type)

    kernel = Matern32(lengthscale=3.0, variance=1.0, output_dim=1)

    mean_func = ImpulseMeanFunction(
        action_times=action_times,
        state_perturbations=tf.constant(state_perturbations),
        kernel=kernel,
    )

    # can't start from zero as the impulse happens fractionally after that
    time_points = tf.linspace(tf.constant(0.01, dtype=float_type), 12.0, num=100)

    # μ(t) = exp(F (t - t₀)) u₀
    expected_state = (
        tf.linalg.expm(kernel.feedback_matrix * time_points[:, None, None])
        @ state_perturbations[..., None]
    )[..., 0]
    expected_f = kernel.generate_emission_model(time_points).project_state_to_f(expected_state)

    tf_values = mean_func(time_points)
    np.testing.assert_allclose(tf_values, expected_f)


def test_step(with_tf_random_seed):
    """Test a single step against a known function."""
    float_type = default_float()

    state_perturbations = tf.constant([[2.3, 1.2]], dtype=float_type)

    action_times = tf.constant([0.0], dtype=float_type)

    kernel = Matern32(lengthscale=3.0, variance=1.0, output_dim=1)

    mean_func = StepMeanFunction(
        action_times=action_times, state_perturbations=state_perturbations, kernel=kernel
    )

    time_points = tf.linspace(tf.constant(0.0, dtype=float_type), 12.0, num=1001)

    # μ(t) = exp(F (t - t₀))F⁻¹u₀ - F⁻¹u₀
    F_inv_u = tf.linalg.inv(kernel.feedback_matrix) @ state_perturbations[..., None]
    expected_state = (
        tf.linalg.expm(kernel.feedback_matrix * time_points[:, None, None]) @ F_inv_u - F_inv_u
    )[..., 0]

    expected_f = kernel.generate_emission_model(time_points).project_state_to_f(expected_state)

    tf_values = mean_func(time_points)
    np.testing.assert_allclose(tf_values, expected_f)


def test_zero_function(with_tf_random_seed, batch_shape):
    """Test the zero function is zero."""
    num_time_points = 7
    obs_dim = 3

    time_points = generate_random_time_points(
        expected_range=EXPECTED_RANGE, shape=batch_shape + (num_time_points,)
    )
    zero_mean_function = ZeroMeanFunction(obs_dim=obs_dim)

    actual = zero_mean_function(tf.constant(time_points))
    np.testing.assert_allclose(actual, 0.0, atol=1e-10)


def test_linear_function(with_tf_random_seed, batch_shape):
    """Test the linear function scales timepoints correctly."""
    num_time_points = 7
    obs_dim = 3
    coefficient = gpflow.Parameter(3.1)

    time_points = generate_random_time_points(
        expected_range=EXPECTED_RANGE, shape=batch_shape + (num_time_points,)
    )
    linear_mean_function = LinearMeanFunction(coefficient=coefficient, obs_dim=obs_dim)
    assert linear_mean_function.trainable_parameters == (coefficient,)
    gpflow.set_trainable(coefficient, False)
    assert linear_mean_function.trainable_parameters == ()

    actual = linear_mean_function(tf.constant(time_points))
    np.testing.assert_allclose(
        actual,
        np.broadcast_to(
            time_points[..., None] * coefficient, batch_shape + (num_time_points, obs_dim)
        ),
    )
