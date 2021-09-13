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
"""Unit tests for the ssm-gaussian transformations."""
import numpy as np
import pytest
import tensorflow as tf

from markovflow.kernels import Matern52, Sum
from markovflow.ssm_gaussian_transformations import (
    expectations_to_ssm_params,
    naturals_to_ssm_params,
    naturals_to_ssm_params_no_smoothing,
    ssm_to_expectations,
    ssm_to_naturals,
    ssm_to_naturals_no_smoothing,
)

RELATIVE_TOLERANCE = 1e-7
ABSOLUTE_TOLERANCE = 1e-6


@pytest.fixture(name="ssm_setup")
def _ssm_setup_fixture():
    return _setup()


def _setup(num_transitions=1000):
    """Create a state space model with a given batch shape."""

    kern_list = [Matern52(lengthscale=0.01, variance=0.01) for _ in range(10)]
    kern = Sum(kern_list)
    X = tf.convert_to_tensor(np.linspace(0, 1, num_transitions + 1), tf.float64)
    ssm = kern.state_space_model(X)

    initial_mean = ssm.initial_mean
    chol_initial_covariance = ssm.cholesky_initial_covariance
    state_transitions = ssm.state_transitions
    state_offsets = ssm.state_offsets
    chol_process_covariances = ssm.cholesky_process_covariances

    ssm_params = (
        state_transitions,
        state_offsets,
        chol_initial_covariance,
        chol_process_covariances,
        initial_mean,
    )
    return ssm, ssm_params


def test_expectation_transformations(with_tf_random_seed, ssm_setup):
    """Test transformations to/from expectation parameters."""

    ssm, ssm_params = ssm_setup

    etas = ssm_to_expectations(ssm)
    ssm_params_reconstructed = expectations_to_ssm_params(*etas)

    for param, param_reconstructed in zip(ssm_params, ssm_params_reconstructed):
        np.testing.assert_allclose(
            param, param_reconstructed, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
        )


def test_natural_transformations(with_tf_random_seed, ssm_setup):
    """Test transformations to/from natural parameters."""

    ssm, ssm_params = ssm_setup

    thetas = ssm_to_naturals(ssm)
    ssm_params_reconstructed = naturals_to_ssm_params(*thetas)

    for param, param_reconstructed in zip(ssm_params, ssm_params_reconstructed):
        np.testing.assert_allclose(
            param, param_reconstructed, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
        )


def test_natural_transformations_no_smoothing(with_tf_random_seed, ssm_setup):
    """Test transformations to/from natural parameters without taking into account the smoothing."""

    ssm, ssm_params = ssm_setup

    thetas = ssm_to_naturals_no_smoothing(ssm)
    ssm_params_reconstructed = naturals_to_ssm_params_no_smoothing(*thetas)

    for param, param_reconstructed in zip(ssm_params, ssm_params_reconstructed):
        np.testing.assert_allclose(
            param, param_reconstructed, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
        )
