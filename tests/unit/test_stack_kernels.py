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

from typing import List

import numpy as np
import pytest
import tensorflow as tf

from markovflow.kernels import Constant, IndependentMultiOutputStack, Matern12, Matern32, Matern52
from markovflow.kernels.sde_kernel import SDEKernel, StackKernel
from markovflow.utils import augment_matrix, augment_square_matrix, to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.kernels.kernel_creators import JITTER, _create_markovflow_primitive_kernel


@pytest.fixture(name="batch_setup")
def _setup_batch(batch_shape):
    """Create random time points with batch_shape and a Constant kernel"""
    num_data = 9
    np.random.seed(1234)
    shape = batch_shape + (num_data,)
    return (
        tf.convert_to_tensor(
            generate_random_time_points(expected_range=4.0, shape=shape), tf.float64
        ),
        batch_shape,
    )


@pytest.fixture(
    name="child_kernels",
    params=[
        [Constant],
        [Matern12],
        [Matern32, Matern32],
        [Constant, Matern12, Matern32],
        [Constant, Matern12, Matern32, Matern52],
    ],
)
def _child_kernels_fixture(request) -> List[SDEKernel]:
    """
    Create a list of child kernels
    """
    return [_create_markovflow_primitive_kernel(t) for t in request.param]


def _multi_output(kernel_list):
    return IndependentMultiOutputStack(kernel_list)


@pytest.fixture(name="stack_kernel", params=[_multi_output])
def _stack_kernel(request, child_kernels) -> StackKernel:
    """
    Return a `StackKernel` object
    """
    return request.param(child_kernels)


@pytest.fixture(name="stack_kernel_function", params=[IndependentMultiOutputStack])
def _stack_kernel_function(request) -> StackKernel:
    """
    Return a `StackKernel` class
    """
    return request.param


def test_init_with_no_child_kernel(stack_kernel_function):
    """
    Test that creating a `StackKernel` with no child kernel raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        stack_kernel_function([])
    assert "There must be at least one child kernel." in str(exp.value)


def test_init_with_negative_jitter(with_tf_random_seed, stack_kernel_function):
    """
    Test that creating a `StackKernel` with negative jitter raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        stack_kernel_function([Constant(variance=1.0, jitter=JITTER)], jitter=-1.0)
    assert "jitter must be a non-negative float number." in str(exp.value)


def test_initial_covariance(with_tf_random_seed, batch_setup, stack_kernel, child_kernels):
    """Test the initial covariance is correct for the `StackKernel`s"""
    time_points, batch_shape = batch_setup
    *_, num_data = time_points.shape
    num_kernels = len(child_kernels)
    time_points_ext_shape = batch_shape + (num_kernels,) + num_data
    time_points_ext = tf.broadcast_to(time_points[..., None, :], time_points_ext_shape)
    actual_tensor = stack_kernel.initial_covariance(time_points_ext[..., 0:1])

    max_dim = actual_tensor.shape[-1]
    expected_list = [k.initial_covariance(time_points[..., 0:1]) for k in child_kernels]
    expected_tensor = tf.stack(
        [augment_square_matrix(k, max_dim - k.shape[-1]) for k in expected_list], axis=-3
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor, atol=2 * JITTER)


def test_state_transitions(with_tf_random_seed, batch_setup, stack_kernel, child_kernels):
    """Test the state transitions are correct for the `StackKernel`s"""
    time_points, _ = batch_setup
    num_kernels = len(child_kernels)

    time_points_tensor = tf.constant(time_points)
    time_points_tensor_ext = tf.tile(
        time_points[..., None, :], [1] * len(time_points.shape[:-1]) + [num_kernels, 1]
    )

    actual_tensor = stack_kernel.state_transitions(
        time_points_tensor_ext[..., :-1], to_delta_time(time_points_tensor_ext)
    )

    max_dim = actual_tensor.shape[-1]
    expected_list = [
        k.state_transitions(time_points_tensor[..., :-1], to_delta_time(time_points_tensor))
        for k in child_kernels
    ]
    expected_tensor = tf.stack(
        [augment_square_matrix(k, max_dim - k.shape[-1], fill_zeros=True) for k in expected_list],
        axis=-4,
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_process_covariances(with_tf_random_seed, batch_setup, stack_kernel, child_kernels):
    """Test the process covariances are correct for the `StackKernel`s"""
    time_points, _ = batch_setup
    num_kernels = len(child_kernels)

    time_points_tensor = tf.constant(time_points)
    time_points_tensor_ext = tf.tile(
        time_points[..., None, :], [1] * len(time_points.shape[:-1]) + [num_kernels, 1]
    )

    actual_tensor = stack_kernel.process_covariances(
        time_points_tensor_ext[..., :-1], to_delta_time(time_points_tensor_ext)
    )

    max_dim = actual_tensor.shape[-1]
    expected_list = [
        k.process_covariances(time_points_tensor[..., :-1], to_delta_time(time_points_tensor))
        for k in child_kernels
    ]
    expected_tensor = tf.stack(
        [augment_square_matrix(k, max_dim - k.shape[-1], fill_zeros=False) for k in expected_list],
        axis=-4,
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor, atol=2 * JITTER)


def test_feedback_matrix(with_tf_random_seed, stack_kernel, child_kernels):
    """Test the feedback matrix is correct for the `StackKernel`s"""

    actual_tensor = stack_kernel.feedback_matrix

    max_dim = actual_tensor.shape[-1]
    expected_list = [k.feedback_matrix for k in child_kernels]
    expected_tensor = tf.stack(
        [augment_square_matrix(k, max_dim - k.shape[-1], fill_zeros=False) for k in expected_list],
        axis=-3,
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_steady_state_covariance(with_tf_random_seed, stack_kernel, child_kernels):
    """Test the steady state covariance is correct for the `StackKernel`s"""

    actual_tensor = stack_kernel.steady_state_covariance

    max_dim = actual_tensor.shape[-1]
    expected_list = [k.steady_state_covariance for k in child_kernels]
    expected_tensor = tf.stack(
        [augment_square_matrix(k, max_dim - k.shape[-1], fill_zeros=False) for k in expected_list],
        axis=-3,
    )[..., None, :, :]

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_initial_mean(with_tf_random_seed, batch_setup, stack_kernel, child_kernels):
    """Test the initial mean is zero for the `StackKernel`s"""
    _, batch_shape = batch_setup
    num_kernels = len(child_kernels)
    batch_shape_ext = batch_shape + (num_kernels,)

    actual_tensor = stack_kernel.initial_mean(batch_shape_ext)

    max_dim = actual_tensor.shape[-1]
    expected_list = [k.initial_mean(batch_shape) for k in child_kernels]
    expected_tensor = tf.stack(
        [augment_matrix(k, max_dim - k.shape[-1]) for k in expected_list], axis=-2
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_multi_output_emission_matrix(
    with_tf_random_seed, batch_setup, stack_kernel, child_kernels
):
    """Test the emission matrix for the `IndependentMultiOutputStack` kernel"""
    time_points, _ = batch_setup
    num_kernels = len(child_kernels)

    time_points_tensor_ext = tf.tile(
        time_points[..., None, :], [1] * len(time_points.shape[:-1]) + [num_kernels, 1]
    )

    actual_tensor = stack_kernel.generate_emission_model(time_points_tensor_ext).emission_matrix

    max_dim = actual_tensor.shape[-1]
    expected_list = [k.generate_emission_model(time_points).emission_matrix for k in child_kernels]
    expected_tensor = tf.stack(
        [augment_matrix(k, max_dim - k.shape[-1]) for k in expected_list], axis=-4
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)
