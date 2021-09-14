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
"""Module containing unit tests for the `EmissionModel` class."""
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from gpflow import default_float

from markovflow.emission_model import ComposedPairEmissionModel, EmissionModel, StackEmissionModel

OUTPUT_DIM = 2
INNER_DIM = 3
STATE_DIM = 4
TIME_POINTS = 20


def _gen_emission_matrix(batch_shape, time_points, output_dim, state_dim):
    return np.random.normal(1.0, 0.5, batch_shape + (time_points, output_dim, state_dim))


def _emission_setup(batch_shape):
    """Return a `EmissionModel` and a lookup dictionary for its emission matrix."""
    emission_matrix = _gen_emission_matrix(batch_shape, TIME_POINTS, OUTPUT_DIM, STATE_DIM)
    emitter = EmissionModel(tf.constant(emission_matrix))
    return emitter, {emitter.emission_matrix.experimental_ref(): emission_matrix}


def _composed_emission_setup(batch_shape):
    """Return a `ComposedEmissionModel` and a lookup dictionary for its emission matrices."""
    inner_emission_matrix = _gen_emission_matrix(batch_shape, TIME_POINTS, INNER_DIM, STATE_DIM)
    outer_emission_matrix = _gen_emission_matrix(batch_shape, TIME_POINTS, OUTPUT_DIM, INNER_DIM)
    outer_emission_model = EmissionModel(tf.constant(outer_emission_matrix))
    inner_emission_model = EmissionModel(tf.constant(inner_emission_matrix))
    emitter = ComposedPairEmissionModel(outer_emission_model, inner_emission_model)
    emission_matrix = outer_emission_matrix @ inner_emission_matrix
    return (
        emitter,
        {
            emitter.emission_matrix.experimental_ref(): emission_matrix,
            emitter.inner_emission_matrix.experimental_ref(): inner_emission_matrix,
        },
    )


def _hacked_emission_setup(batch_shape):
    """
    Return a `ComposedEmissionModel` with all it's methods overwritten such that it behaves as an
    `EmissionModel` while using its intermediate projection functions, and a lookup dictionary for
    its emission
    matrices.
    """
    emission_model, feed_matrix = _composed_emission_setup(batch_shape)
    emission_model.project_state_covariance_to_f = emission_model.project_state_covariance_to_g
    emission_model.project_state_to_f = emission_model.project_state_to_g
    emission_model.project_state_marginals_to_f = emission_model.project_state_marginals_to_g
    feed_matrix[emission_model.emission_matrix.experimental_ref()] = feed_matrix[
        emission_model.inner_emission_matrix.experimental_ref()
    ]
    return emission_model, feed_matrix


@pytest.fixture(
    name="emission_setup",
    params=[_emission_setup, _composed_emission_setup, _hacked_emission_setup],
)
def _setup_fixture_w_hack(request, batch_shape):
    """
    Return an instance of an `EmissionModel` (i.e. `EmissionModel, `ComposedEmissionModel`,
    or a "hacked" `ComposedEmissionModel`) and a lookup dictionary for its emission matrices
    """
    return (*request.param(batch_shape), batch_shape)


def _create_marginals(
    batch_shape: Tuple, state_dim: int, n_time_points: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Create marginal means and covariances to be passed to test methods."""
    return (
        tf.random.normal(batch_shape + (n_time_points, state_dim), dtype=default_float()),
        _create_random_sym_pos_def_matrix(batch_shape + (n_time_points, state_dim, state_dim)),
    )


def _create_random_sym_pos_def_matrix(shape: Tuple) -> tf.Tensor:
    """Create a random symmetric positive definite matrix."""
    random_mat = tf.random.normal(shape, dtype=default_float())
    return tf.matmul(random_mat, random_mat, transpose_b=True)


def _project_state_numpy(emission_matrix: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Linearly transform a state."""
    return np.einsum("...ij,...j->...i", emission_matrix, array)


def _project_state_cov_numpy(
    emission_matrix: np.ndarray, array: np.ndarray, full_output_cov=True
) -> np.ndarray:
    """Linearly transform a covariance."""
    cov = np.einsum("...ij,...jk,...lk->...il", emission_matrix, array, emission_matrix)
    if full_output_cov:
        return cov
    else:
        return np.einsum("...oo->...o", cov)


def test_project_state_to_f(with_tf_random_seed, emission_setup):
    """Test that we correctly project a state to f."""
    emitter, lookup_dict, batch_shape = emission_setup
    state = np.random.normal(1.0, 0.5, batch_shape + (TIME_POINTS, emitter.state_dim,))
    projected = emitter.project_state_to_f(tf.constant(state))

    np.testing.assert_allclose(
        projected,
        _project_state_numpy(lookup_dict[emitter.emission_matrix.experimental_ref()], state),
    )


def test_project_state_covariance_to_f(with_tf_random_seed, emission_setup, batch_shape):
    """ Test that we correctly project the covariance
    from the state space to the observation space """
    emitter, lookup_dict, batch_shape = emission_setup

    state_cov = _create_random_sym_pos_def_matrix(
        batch_shape + (TIME_POINTS, emitter.state_dim, emitter.state_dim)
    )
    state_cov_tf = tf.convert_to_tensor(state_cov)

    # full output covariance
    state, projected_full = (
        state_cov_tf,
        emitter.project_state_covariance_to_f(state_cov_tf, True),
    )
    projected_full_np = _project_state_cov_numpy(
        lookup_dict[emitter.emission_matrix.experimental_ref()], state, True
    )
    # marginal output variance
    state, projected_marg = (
        state_cov_tf,
        emitter.project_state_covariance_to_f(state_cov_tf, False),
    )
    projected_marg_np = _project_state_cov_numpy(
        lookup_dict[emitter.emission_matrix.experimental_ref()], state, False
    )

    # compare numpy against tf
    np.testing.assert_allclose(projected_full, projected_full_np)
    np.testing.assert_allclose(projected_marg, projected_marg_np)

    # consistency check (diagonal of full_output_cov match the marginals)
    np.testing.assert_allclose(np.einsum("...oo->...o", projected_full_np), projected_marg_np)


def test_marginal_means(with_tf_random_seed, emission_setup, batch_shape):
    """Test that the marginal means of f are the state marginals transformed by H."""
    emitter, lookup_dict, batch_shape = emission_setup

    means, covs = _create_marginals(batch_shape, emitter.state_dim, TIME_POINTS)
    state_means, f_means = means, emitter.project_state_marginals_to_f(means, covs)[0]

    np.testing.assert_allclose(
        f_means,
        _project_state_numpy(lookup_dict[emitter.emission_matrix.experimental_ref()], state_means),
    )


def test_marginal_covs(with_tf_random_seed, emission_setup, batch_shape):
    """Test that the marginal covariances of f are the state marginals transformed by H."""
    emitter, lookup_dict, batch_shape = emission_setup

    means, covs = _create_marginals(batch_shape, emitter.state_dim, TIME_POINTS)
    state_covs, f_covs = (
        covs,
        emitter.project_state_marginals_to_f(means, covs, full_output_cov=True)[1],
    )

    np.testing.assert_allclose(
        f_covs,
        _project_state_cov_numpy(
            lookup_dict[emitter.emission_matrix.experimental_ref()], state_covs
        ),
    )


def test_samples(with_tf_random_seed, emission_setup, batch_shape):
    """Test that the samples of f are the state samples transformed by H."""
    emitter, lookup_dict, batch_shape = emission_setup

    sample_shape = (2, 7)

    state_samples = tf.random.normal(
        sample_shape + batch_shape + (TIME_POINTS, emitter.state_dim), dtype=default_float()
    )

    samples = emitter.project_state_to_f(state_samples)

    np.testing.assert_allclose(
        _project_state_numpy(
            lookup_dict[emitter.emission_matrix.experimental_ref()], state_samples
        ),
        samples,
    )


def _emission_latent_setup(batch_shape):
    """Return a `StackEmissionModel` and a lookup dictionary for its emission matrix."""
    emission_matrix = np.random.normal(
        1.0, 0.5, batch_shape + (OUTPUT_DIM, TIME_POINTS, 1, STATE_DIM)
    )
    emitter = StackEmissionModel(tf.constant(emission_matrix))
    return emitter, {emitter.emission_matrix.experimental_ref(): emission_matrix}


@pytest.fixture(name="latent_emission_setup", params=[_emission_latent_setup])
def _setup_latent_emission(request, batch_shape):
    """
    Return an instance of an `StackEmissionModel` and a lookup dictionary for its emission matrices
    """
    return (*request.param(batch_shape), batch_shape)


def test_project_state_to_f_multi_output(with_tf_random_seed, latent_emission_setup):
    """Test that we correctly project a state to f with a `StackEmissionModel`."""
    emitter, lookup_dict, batch_shape = latent_emission_setup
    num_latent = emitter.output_dim

    state = np.random.normal(1.0, 0.5, batch_shape + (TIME_POINTS, emitter.state_dim,))
    state_ext = np.repeat(state[..., None, :, :], num_latent, axis=-3)
    projected = emitter.project_state_to_f(tf.constant(state_ext))

    projected_np = _project_state_numpy(
        lookup_dict[emitter.emission_matrix.experimental_ref()], state[..., None, :, :]
    )
    projected_np = np.moveaxis(projected_np[..., 0], -2, -1)
    np.testing.assert_allclose(projected, projected_np)


def test_project_state_covariance_to_f_multi_output(
    with_tf_random_seed, latent_emission_setup, batch_shape
):
    """ Test that we correctly project the covariance
    from the state space to the observation space with latent emission model """
    emitter, lookup_dict, batch_shape = latent_emission_setup
    num_latent = emitter.output_dim

    state_cov = _create_random_sym_pos_def_matrix(
        batch_shape + (TIME_POINTS, emitter.state_dim, emitter.state_dim)
    )
    state_cov_ext = np.repeat(state_cov[..., None, :, :, :], num_latent, axis=-4)
    state_cov_tf_ext = tf.convert_to_tensor(state_cov_ext)

    # marginal output variance
    projected_marg = emitter.project_state_covariance_to_f(state_cov_tf_ext, False)
    projected_marg_np = _project_state_cov_numpy(
        lookup_dict[emitter.emission_matrix.experimental_ref()],
        state_cov[..., None, :, :, :],
        False,
    )
    projected_marg_np = np.moveaxis(projected_marg_np[..., 0], -2, -1)

    # compare numpy against tf
    np.testing.assert_allclose(projected_marg, projected_marg_np)


def test_samples_multi_output(with_tf_random_seed, latent_emission_setup, batch_shape):
    """ Test that the samples of f are the state samples transformed by H
    with a `StackEmissionModel`. """
    emitter, lookup_dict, batch_shape = latent_emission_setup
    num_latent = emitter.output_dim

    sample_shape = (2, 7)

    state_samples = np.random.normal(
        0.0, 1.0, sample_shape + batch_shape + (TIME_POINTS, emitter.state_dim)
    )
    state_samples_ext = np.repeat(state_samples[..., None, :, :], num_latent, axis=-3)

    samples = emitter.project_state_to_f(state_samples_ext)

    samples_np = _project_state_numpy(
        lookup_dict[emitter.emission_matrix.experimental_ref()], state_samples[..., None, :, :]
    )
    samples_np = np.moveaxis(samples_np[..., 0], -2, -1)

    np.testing.assert_allclose(samples, samples_np)
