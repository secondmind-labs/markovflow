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
import random

import numpy as np
import pytest
import tensorflow as tf

DEFAULT_SEED = 71892305


@pytest.fixture
def with_tf_random_seed():
    """
    Sets a random seed in Python, NumPy and TensorFlow. This maintains the random seed setting
    behaviour of the `with_tf_session` function, which creates a `tf.test.TestCase`.

    Use this for tests which are being migrated away from using `with_tf_session`, but which
    need the same random seed setting behaviour as that fixture.
    """
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    tf.random.set_seed(DEFAULT_SEED)


@pytest.fixture(
    name="batch_shape", params=[tf.TensorShape([3,]), tf.TensorShape([]), tf.TensorShape([2, 1])],
)
def _batch_shape_fixture(request):
    return request.param


@pytest.fixture(name="output_dim", params=[1, 2])
def _output_dim_fixture(request):
    return request.param


@pytest.fixture(
    name="num_transitions", params=[5, 50, 100, 500],
)
def _num_transitions_fixture(request):
    return request.param


@pytest.fixture(
    name="num_observations", params=[0, 2, 10, 50],
)
def _num_observations_fixture(request):
    return request.param
