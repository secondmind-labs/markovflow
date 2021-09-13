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
"""Module to test f covariances from Markovflow kernels against GPflow kernels."""
import numpy as np
import pytest
import tensorflow as tf

from markovflow.kernels import Constant, Matern12, Matern32, Matern52, Product, SDEKernel, Sum
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.kernels.kernel_creators import (
    _create_gpflow_kernel,
    _create_markovflow_primitive_kernel,
)
from tests.tools.state_space_model import f_covariances

JITTER = 1e-6


@pytest.fixture(name="markov_batch_setup")
def _setup_markov_batch():
    """Create random time points with batch_shape and a Constant kernel"""
    num_data = 9
    np.random.seed(1234)
    batch_shape = tuple()
    return (
        generate_random_time_points(expected_range=4.0, shape=batch_shape + (num_data,)),
        batch_shape,
    )


@pytest.fixture(
    name="markov_product_kernel",
    params=[
        [Constant],
        [Matern12],
        [Constant, Matern12],
        [Constant, Matern12, Matern32],
        [Constant, Matern12, Matern32, Matern52],
    ],
)
def _setup_markov_product_kernel(request) -> Product:
    """
    Create a Product kernel with different children kernels.
    """
    children_kernels = [_create_markovflow_primitive_kernel(t) for t in request.param]
    return Product(children_kernels, jitter=JITTER)


@pytest.fixture(
    name="markov_sum_kernel",
    params=[
        [Constant],
        [Matern12],
        [Constant, Matern12],
        [Constant, Matern12, Matern32],
        [Constant, Matern12, Matern32, Matern52],
    ],
)
def _setup_sum_kernel(request) -> Sum:
    """
    Create a Sum kernel with different children kernels.
    """
    children_kernels = [_create_markovflow_primitive_kernel(t) for t in request.param]
    return Sum(children_kernels, jitter=JITTER)


@pytest.fixture(name="markov_primitive_kernel", params=[Constant, Matern12, Matern32, Matern52])
def _setup_primitive_kernel(request) -> SDEKernel:
    """
    Create a primitive Markovflow kernel.
    """
    return _create_markovflow_primitive_kernel(request.param)


def _test_f_covariance_against_gpflow(time_points: np.ndarray, markovflow_kernel: SDEKernel):
    """
    Test infrastructure to check if the covariance matrix of a markovflow kernel
    is the same as the covariance matrix of the corresponding GPflow kernel.
    :param time_points: time points to evaluate the covariance matrix.
    :param markovflow_kernel: the given markovflow kernel.
    """
    time_points_tensor = tf.constant(time_points)
    markovflow_f_covs = f_covariances(
        markovflow_kernel.state_space_model(time_points_tensor),
        markovflow_kernel.generate_emission_model(time_points_tensor),
    )

    # Create GPflow sum kernel.
    gpflow_kernel = _create_gpflow_kernel(markovflow_kernel)
    gpf_covs = gpflow_kernel.K(time_points[:, None], time_points[:, None])

    np.testing.assert_allclose(markovflow_f_covs, gpf_covs, rtol=1e-2)


def test_product_f_covariance_against_gpflow(markov_batch_setup, markov_product_kernel):
    """
    Test if f covariances from Markovflow Product kernel is the same as GPflow Product kernel.
    """

    time_points, _ = markov_batch_setup
    _test_f_covariance_against_gpflow(time_points, markov_product_kernel)


def test_sum_f_covariance_against_gpflow(markov_batch_setup, markov_sum_kernel):
    """
    Test if f covariances from Markovflow Sum kernel is the same as GPflow Sum kernel.
    """
    time_points, _ = markov_batch_setup
    _test_f_covariance_against_gpflow(time_points, markov_sum_kernel)


def test_primitive_f_covariance_against_gpflow(markov_batch_setup, markov_primitive_kernel):
    """
    Test if f covariances from a Markovflow primitive kernel is the same as the
    corresponding GPflow primitive kernel.
    """
    time_points, _ = markov_batch_setup
    _test_f_covariance_against_gpflow(time_points, markov_primitive_kernel)
