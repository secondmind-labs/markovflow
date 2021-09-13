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

from markovflow.kernels import Matern12, Matern32, Matern52, Product
from markovflow.kernels.constant import Constant
from markovflow.utils import kronecker_product, to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.kernels.kernel_creators import JITTER, _create_markovflow_primitive_kernel


@pytest.fixture(name="batch_setup")
def _setup_batch(batch_shape):
    """Create random time points with batch_shape and a Constant kernel"""
    num_data = 3
    np.random.seed(1234)
    time_points = tf.convert_to_tensor(
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        tf.float64,
    )
    return time_points, batch_shape


@pytest.fixture(
    name="product_kernel",
    params=[
        [Constant],
        [Matern12],
        [Constant, Matern12],
        [Constant, Matern12, Matern32],
        [Constant, Matern12, Matern32, Matern52],
    ],
)
def _setup_product_kernel(request) -> Product:
    """
    Create a Product kernel with different children kernels.
    """
    children_kernels = [_create_markovflow_primitive_kernel(t) for t in request.param]
    return Product(children_kernels, jitter=JITTER)


def test_init_with_no_child_kernel():
    """
    Test that create a Product kernel with no child kernel raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        Product([])
    assert str(exp.value).find("There must be at least one child kernel.") >= 0


def test_init_with_negative_jitter(with_tf_random_seed):
    """
    Test that create a Product kernel negative jitter raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        Product([Constant(variance=1.0, jitter=1e-2, output_dim=1)], jitter=-1.0)
    assert str(exp.value).find("jitter must be a non-negative float number.") >= 0


def test_initial_covariance(with_tf_random_seed, batch_setup, product_kernel):
    """Test the initial covariance is correct for the Product kernel."""
    time_points, _ = batch_setup

    kernel = product_kernel

    actual_tensor = kernel.initial_covariance(time_points[..., 0:1])
    # be wary this will sum the jitter on each child matrix.
    expected_tensor = kronecker_product(
        [k.initial_covariance(time_points[..., 0:1]) for k in kernel.kernels]
    )

    actual, expected = actual_tensor, expected_tensor
    np.testing.assert_allclose(actual, expected, atol=2 * JITTER)


def test_state_transitions(with_tf_random_seed, batch_setup, product_kernel):
    """Test the state transitions are correct for the Product kernel."""
    time_points_tensor, _ = batch_setup
    delta_time = to_delta_time(time_points_tensor)
    kernel = product_kernel
    actual_tensor = kernel.state_transitions(time_points_tensor[..., :-1], delta_time)
    expected_tensor = kronecker_product(
        [k.state_transitions(time_points_tensor[..., :-1], delta_time) for k in kernel.kernels]
    )

    actual, expected = actual_tensor, expected_tensor
    np.testing.assert_allclose(actual, expected)


def test_feedback_matrix(with_tf_random_seed, batch_setup, product_kernel):
    """Test the feedback matrix is correct for the Product kernel."""
    kernel = product_kernel
    actual_tensor = kernel.feedback_matrix
    expected_tensor = kronecker_product([k.feedback_matrix for k in kernel.kernels])

    actual, expected = actual_tensor, expected_tensor
    np.testing.assert_allclose(actual, expected)


def test_steady_state_covariance(with_tf_random_seed, batch_setup, product_kernel):
    """Test the steady state covariance is correct for the Product kernel."""
    kernel = product_kernel
    actual_tensor = kernel.steady_state_covariance
    expected_tensor = kronecker_product([k.steady_state_covariance for k in kernel.kernels])

    actual, expected = actual_tensor, expected_tensor
    np.testing.assert_allclose(actual, expected)


def test_emission_model(with_tf_random_seed, product_kernel):
    """ Test the emission model for the Product kernel."""
    time_points = tf.constant(generate_random_time_points(1, (3,)))
    kernel = product_kernel
    actual_tensor = kernel.generate_emission_model(time_points).emission_matrix
    expected_tensor = kronecker_product(
        [k.generate_emission_model(time_points).emission_matrix for k in kernel.kernels]
    )

    actual, expected = actual_tensor, expected_tensor
    np.testing.assert_allclose(actual, expected)
