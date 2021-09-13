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

from markovflow.kernels import (
    Constant,
    FactorAnalysisKernel,
    IndependentMultiOutput,
    Matern12,
    Matern32,
    Matern52,
)
from markovflow.kernels.sde_kernel import ConcatKernel, SDEKernel, Sum
from markovflow.utils import block_diag, to_delta_time
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.kernels.kernel_creators import (
    OUTPUT_DIM,
    DummyNonSDEKernel,
    _create_markovflow_primitive_kernel,
)

JITTER = 1e-6


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


def _sum(kernel_list):
    return Sum(kernel_list, jitter=JITTER)


def _multi_output(kernel_list):
    return IndependentMultiOutput(kernel_list, jitter=JITTER)


def _simple_factor_analysis_kernel(kernel_list):
    weight_function = lambda x: x
    return FactorAnalysisKernel(weight_function, kernel_list, len(kernel_list), jitter=JITTER)


@pytest.fixture(name="concat_kernel", params=[_sum, _multi_output, _simple_factor_analysis_kernel])
def _concat_kernel(request, child_kernels) -> ConcatKernel:
    """
    Return a `ConcatKernel` object
    """
    return request.param(child_kernels)


@pytest.fixture(name="concat_kernel_function", params=[Sum, IndependentMultiOutput])
def _concat_kernel_function(request) -> ConcatKernel:
    """
    Return a `ConcatKernel` class
    """
    return request.param


def test_init_with_no_child_kernel(concat_kernel_function):
    """
    Test that creating a `ConcatKernel` with no child kernel raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        concat_kernel_function([])
    assert str(exp.value).find("There must be at least one child kernel.") >= 0


def test_invalid_child_kernel(concat_kernel_function):
    """
    Test that creating a `ConcatKernel` with no child kernels raises an exception.
    """
    with pytest.raises(TypeError) as exp:
        concat_kernel_function([DummyNonSDEKernel()])
    assert str(exp.value).find("Can only combine SDEKernel instances.") >= 0


def test_init_with_negative_jitter(with_tf_random_seed, concat_kernel_function):
    """
    Test that creating a `ConcatKernel` with negative jitter raises an exception.
    """
    with pytest.raises(AssertionError) as exp:
        concat_kernel_function(
            [Constant(variance=1.0, jitter=JITTER, output_dim=OUTPUT_DIM)], jitter=-1.0
        )
    assert str(exp.value).find("jitter must be a non-negative float number.") >= 0


def test_initial_covariance(with_tf_random_seed, batch_setup, concat_kernel, child_kernels):
    """Test the initial covariance is correct for the `ConcatKernel`s"""
    time_points, _ = batch_setup
    actual_tensor = concat_kernel.initial_covariance(time_points[..., 0:1])
    expected_tensor = block_diag(
        [k.initial_covariance(time_points[..., 0:1]) for k in concat_kernel.kernels]
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor, atol=JITTER)


def test_state_transitions(with_tf_random_seed, batch_setup, concat_kernel, child_kernels):
    """Test the state transitions are correct for the `ConcatKernel`s"""
    time_points, _ = batch_setup
    time_deltas = to_delta_time(time_points)
    actual_tensor = concat_kernel.state_transitions(time_points, time_deltas)
    expected_tensor = block_diag(
        [k.state_transitions(time_points, time_deltas) for k in concat_kernel.kernels]
    )
    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_process_covariances(with_tf_random_seed, batch_setup, concat_kernel, child_kernels):
    """Test the process covariances are correct for the `ConcatKernel`s"""
    time_points, _ = batch_setup
    time_deltas = to_delta_time(time_points)
    actual_tensor = concat_kernel.process_covariances(time_points, time_deltas)

    expected_tensor = block_diag(
        [k.process_covariances(time_points, time_deltas) for k in concat_kernel.kernels]
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor, atol=JITTER)


def test_feedback_matrix(with_tf_random_seed, concat_kernel, child_kernels):
    """Test the feedback matrix is correct for the `ConcatKernel`s"""
    actual_tensor = concat_kernel.feedback_matrix
    expected_tensor = block_diag([k.feedback_matrix for k in concat_kernel.kernels])

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_steady_state_covariance(with_tf_random_seed, concat_kernel, child_kernels):
    """Test the steady state covariance is correct for the `ConcatKernel`s"""
    actual_tensor = concat_kernel.steady_state_covariance
    expected_tensor = block_diag([k.steady_state_covariance for k in concat_kernel.kernels])

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_initial_mean(with_tf_random_seed, batch_setup, concat_kernel, child_kernels):
    """Test the initial mean is zero for the `ConcatKernel`s"""
    _, batch_shape = batch_setup
    actual_tensor = concat_kernel.initial_mean(batch_shape)
    expected_tensor = tf.concat(
        [k.initial_mean(batch_shape) for k in concat_kernel.kernels], axis=-1
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_sum_emission_model(with_tf_random_seed, batch_setup, child_kernels):
    """Test the emission model for the `Sum` kernel"""
    # TODO: this is a pointless test, it checks the same code inside and outside the function
    time_points, _ = batch_setup
    kernel = Sum(child_kernels)
    actual_tensor = kernel.generate_emission_model(time_points).emission_matrix
    expected_tensor = tf.concat(
        [k.generate_emission_model(time_points).emission_matrix for k in kernel.kernels], axis=-1,
    )

    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_multi_output_emission_model(with_tf_random_seed, batch_setup):
    """Test the emission model for the `MultiOutput` kernel"""
    time_points, _ = batch_setup
    child_kernels = [_create_markovflow_primitive_kernel(k) for k in [Matern32, Matern32]]
    kernel = IndependentMultiOutput(child_kernels)
    actual_tensor = kernel.generate_emission_model(time_points).emission_matrix
    child_emission_matrix = child_kernels[0].generate_emission_model(time_points).emission_matrix
    zero_pad = tf.zeros_like(child_emission_matrix)
    expected_tensor = tf.concat(
        [
            tf.concat([child_emission_matrix, zero_pad], axis=-1),
            tf.concat([zero_pad, child_emission_matrix], axis=-1),
        ],
        axis=-2,
    )
    np.testing.assert_allclose(actual_tensor, expected_tensor)


def _setup_weight_functions(batch_shape, output_dim, latent_dim):
    """
    Return equivalent numpy and tensorflow functions which map from shape
        batch_shape + (num_times)
    to
        batch_shape + (num_data, output_dim, latent_dim)
    """
    seed_matrix = np.random.rand(*batch_shape, output_dim, latent_dim)

    def weight_function(times: np.ndarray) -> np.ndarray:
        x = np.einsum("...t,...ik->...tik", times, seed_matrix)
        return x - np.round(x)

    def tf_weight_function(times: tf.Tensor) -> tf.Tensor:
        x = tf.einsum("...t,...ik->...tik", times, tf.constant(seed_matrix))
        return x - tf.round(x)

    return weight_function, tf_weight_function


def test_factor_analysis_kernel_emission_model(with_tf_random_seed, batch_setup, output_dim):
    """Test the emission model for the `FactorAnalysisKernel`"""
    time_points, batch_shape = batch_setup
    child_kernel = _create_markovflow_primitive_kernel(Matern32)
    child_kernels = [child_kernel, child_kernel]
    latent_dim = len(child_kernels)
    wfn, tf_wfn = _setup_weight_functions(batch_shape, output_dim, latent_dim)

    kernel = FactorAnalysisKernel(tf_wfn, child_kernels, output_dim)
    actual_emission_matrix = kernel.generate_emission_model(
        tf.constant(time_points)
    ).emission_matrix

    child_emission_matrix = child_kernels[0].generate_emission_model(time_points).emission_matrix
    zero_pad = tf.zeros_like(child_emission_matrix)
    expected_emission_matrix = tf.concat(
        [
            tf.concat([child_emission_matrix, zero_pad], axis=-1),
            tf.concat([zero_pad, child_emission_matrix], axis=-1),
        ],
        axis=-2,
    )

    expected_emission_matrix = wfn(time_points) @ expected_emission_matrix
    np.testing.assert_allclose(actual_emission_matrix, expected_emission_matrix)


def test_can_generate_state_space_model(with_tf_random_seed, batch_setup, concat_kernel):
    """
    Test that we can generate a state space model from the kernel by passing in time points as
    numpy arrays or tensors
    """
    time_points, _ = batch_setup

    concat_kernel.state_space_model(time_points)
    concat_kernel.state_space_model(tf.identity(time_points))
