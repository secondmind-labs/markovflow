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
"""Module containing the unit tests for the assert statements in the `Matern` kernels."""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from markovflow.base import default_float
from markovflow.kernels.matern import Matern12, Matern32, Matern52
from markovflow.utils import to_delta_time

MATERN_KERNEL_LIST = [Matern12, Matern32, Matern52]


@pytest.mark.parametrize("matern", MATERN_KERNEL_LIST)
def test_matern_zero_lengthscale(matern):
    """Test that initializing a Matern1/2 kernel with 0 lengthscale raises an exception"""
    with pytest.raises(ValueError) as exp:
        matern(lengthscale=0.0, variance=1.0, output_dim=1)
    assert exp.value.args[0].find("lengthscale must be positive.") >= 0


@pytest.mark.parametrize("matern", MATERN_KERNEL_LIST)
def test_matern12_zero_variance(matern):
    """Test that initializing a Matern1/2 kernel with 0 variance raises an exception"""
    with pytest.raises(ValueError) as exp:
        matern(lengthscale=1.0, variance=0.0, output_dim=1)
    assert exp.value.args[0].find("variance must be positive.") >= 0


@pytest.mark.parametrize("np_time_points", [np.array([[1, 0]])])  # delta time is negative
def test_to_delta_time_positive_difference(with_tf_random_seed, np_time_points):
    """Test that the assertion fires for a negative delta time"""
    time_points = tf.constant(np_time_points, dtype=default_float())

    with pytest.raises(InvalidArgumentError) as exp:
        to_delta_time(time_points)

    assert exp.value.message.find("Condition x >= y") >= 0
