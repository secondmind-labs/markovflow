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
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow as tf

from tests.tools.state_space_model import StateSpaceModelBuilder


def _get_samples(batch_shape, state_dim, transitions, sample_shape) -> np.ndarray:
    """
    Return samples from a `StateSpaceModel`.
    """
    ssm, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()

    return ssm.sample(sample_shape).numpy()


def test_empty_batch_shape_tuple(with_tf_random_seed, state_dim, transitions, sample_shape):
    samples = _get_samples(tuple(), state_dim, transitions, sample_shape)

    if isinstance(sample_shape, int):
        sample_shape = (sample_shape,)
    assert samples.shape == sample_shape + (transitions + 1, state_dim)


def test_zero_state_dim(with_tf_random_seed, batch_shape, transitions, sample_shape):
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = _get_samples(batch_shape, 0, transitions, sample_shape)


def test_zero_samples(with_tf_random_seed, batch_shape, state_dim, transitions):
    samples = _get_samples(batch_shape, state_dim, transitions, 0)

    assert samples.size == 0


def _get_almost_deterministic_samples(
    batch_shape, state_dim, transitions, sample_shape
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return samples from a `StateSpaceModel` in which the stochasticity has been reduced to almost
    zero.
    """
    P_0_shape = batch_shape + (state_dim, state_dim)
    Q_s_shape = batch_shape + (transitions, state_dim, state_dim)

    ssm, array_dict = (
        StateSpaceModelBuilder(batch_shape, state_dim, transitions)
        .with_P_0(np.broadcast_to(np.eye(state_dim) * sys.float_info.min, shape=P_0_shape))
        .with_Q_s(np.broadcast_to(np.eye(state_dim) * sys.float_info.min, shape=Q_s_shape))
        .build()
    )

    return ssm.sample(sample_shape).numpy(), array_dict


def _get_expected_samples(A_s, b_s, mu_0, sample_shape) -> np.ndarray:
    """
    Given an initial `mu_0`, calculate the expected samples from an almost-deterministic
    `StateSpaceModel`.
    """
    *batch_shape, transitions, state_dim = b_s.shape
    means_list = [mu_0]
    for i in range(transitions):
        means_list.append(
            np.einsum("...jk,...k->...j", A_s[..., i, :, :], means_list[-1]) + b_s[..., i, :]
        )
    # [... 1, num_transitions, state_dim]
    means = np.stack(means_list, axis=-2)

    # sample_shape +[num_transitions, state_dim]
    sample_shape = [sample_shape,] if isinstance(sample_shape, int) else list(sample_shape)
    expected_samples = np.broadcast_to(
        means, sample_shape + batch_shape + [transitions + 1, state_dim]
    )

    return expected_samples


@pytest.fixture(name="sample_shape", params=[1, (0, 4), (4, 4), 100])
def _sample_shape_fixture(request):
    return request.param


def test_almost_deterministic_model_samples_all_match(
    with_tf_random_seed, batch_shape, state_dim, sample_shape
):
    transitions = 1

    samples, array_dict = _get_almost_deterministic_samples(
        batch_shape, state_dim, transitions, sample_shape
    )

    expected_samples = _get_expected_samples(
        array_dict["A_s"], array_dict["b_s"], array_dict["mu_0"], sample_shape
    )
    np.testing.assert_allclose(samples, expected_samples)


def test_almost_deterministic_transitions(
    with_tf_random_seed, batch_shape, transitions, sample_shape
):
    state_dim = 1

    samples, array_dict = _get_almost_deterministic_samples(
        batch_shape, state_dim, transitions, sample_shape
    )

    expected_samples = _get_expected_samples(
        array_dict["A_s"], array_dict["b_s"], array_dict["mu_0"], sample_shape
    )

    np.testing.assert_allclose(samples, expected_samples)
