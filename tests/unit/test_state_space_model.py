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
"""Integration tests for the `StateSpaceModel`."""
import inspect

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from markovflow.base import auto_namescope_enabled, default_float
from markovflow.state_space_model import StateSpaceModel
from tests.tools.state_space_model import StateSpaceModelBuilder, precision_spingp


@pytest.fixture(name="ssm_setup")
def _ssm_setup_fixture(batch_shape):
    return _setup(batch_shape)


def _setup(batch_shape, state_dim=3, transitions=5):
    """Create a state space model with a given batch shape."""
    return StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()


def test_log_det(with_tf_random_seed, ssm_setup):
    """Verify that the log determinant is correctly calculated."""
    ssm, _ = ssm_setup

    log_det_p = ssm.log_det_precision()
    log_det_dense = tf.linalg.logdet(ssm.precision.to_dense())

    np.testing.assert_allclose(log_det_p, log_det_dense)


def test_precision(with_tf_random_seed, ssm_setup):
    """Verify that the precision is correct by calculating it two ways."""
    ssm, _ = ssm_setup

    prec, prec_simple = ssm.precision.as_band, precision_spingp(ssm)
    np.testing.assert_allclose(prec, prec_simple)


def test_precision_zero_transitions(with_tf_random_seed, batch_shape):
    with pytest.raises(tf.errors.InvalidArgumentError):
        _setup(batch_shape, transitions=0)


def test_marginal_means(with_tf_random_seed, ssm_setup):
    """ Test that we generate the correct marginal means and covariances. """
    ssm, array_dict = ssm_setup
    transitions = ssm.num_transitions

    marginal_ms = ssm.marginal_means
    means = [array_dict["mu_0"]]
    for i in range(transitions):
        means.append(
            np.einsum("...jk,...k->...j", array_dict["A_s"][..., i, :, :], means[-1])
            + array_dict["b_s"][..., i, :]
        )
    np.testing.assert_allclose(marginal_ms, np.stack(means, axis=-2))


def test_marginal_covs(with_tf_random_seed, ssm_setup):
    """ Test that we generate the correct marginal means and covariances. """
    ssm, array_dict = ssm_setup
    transitions = ssm.num_transitions

    marginal_covs = ssm.marginal_covariances
    covs = [array_dict["P_0"]]
    for i in range(transitions):
        A_t = array_dict["A_s"][..., i, :, :]
        Q_t = array_dict["Q_s"][..., i, :, :]
        covs.append(np.einsum("...ij,...jk,...lk->...il", A_t, covs[-1], A_t) + Q_t)
    np.testing.assert_allclose(marginal_covs, np.stack(covs, axis=-3))


def test_subsequent_marginal_covs(with_tf_random_seed, ssm_setup):
    """Test that we generate the correct Cov(xₖ₊₁, xₖ) = AₖPₖ for each state."""
    ssm, array_dict = ssm_setup

    marginal_covs_tf = ssm.marginal_covariances
    marginal_covs, subsequent_covs = (
        marginal_covs_tf,
        ssm.subsequent_covariances(marginal_covs_tf),
    )
    np.testing.assert_allclose(subsequent_covs, array_dict["A_s"] @ marginal_covs[..., :-1, :, :])


def test_sample(with_tf_random_seed, ssm_setup):
    """Test that we can sample from the state space model."""
    ssm, _ = ssm_setup
    transitions = ssm.num_transitions
    state_dim = ssm.state_dim

    num_samples = 7
    samples = ssm.sample(num_samples)
    assert samples.shape == (num_samples,) + tuple(ssm.batch_shape) + (transitions + 1, state_dim,)

    sample_shape = (3, 5)
    samples = ssm.sample(sample_shape)
    assert samples.shape == sample_shape + tuple(ssm.batch_shape) + (transitions + 1, state_dim,)


def test_create_variable_ssm_same(with_tf_random_seed, ssm_setup):
    """Test that the created variable SSM is initialised to be the same."""
    dist_p, _ = ssm_setup
    dist_q = dist_p.create_trainable_copy()

    k_l = dist_q.kl_divergence(dist_p)

    initial_kl = k_l
    np.testing.assert_allclose(initial_kl, 0.0, atol=1e-6)


def test_variable_ssm(with_tf_random_seed, ssm_setup):
    """Test that the created SSM can be modified."""
    dist_p, _ = ssm_setup
    dist_q = dist_p.create_trainable_copy()

    dist_q.trainable_variables[1].assign_add(
        tf.ones_like(dist_q.trainable_variables[1], dtype=default_float())
    )

    final_kl = dist_q.kl_divergence(dist_p)
    # check that the k_l is significantly non-zero
    np.testing.assert_array_less(1e-6, final_kl)


@pytest.mark.parametrize("batch_shape", [(3,), tuple(), (2, 1)])
def test_kl_divergence(with_tf_random_seed, batch_shape):
    """
    Test that we can take the KL divergence of two state space models.

    Since the joint distributions are Gaussian, we can use tensorflow to calculate them for
    comparison.
    """

    state_dim = 2
    transitions = 1  # keep trajectories short as KL's get quite large
    ssm_1, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()
    ssm_2, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()

    mu_1 = tf.reshape(ssm_1.marginal_means, tuple(ssm_1.batch_shape) + (-1,))
    mu_2 = tf.reshape(ssm_2.marginal_means, tuple(ssm_2.batch_shape) + (-1,))

    prec_1 = ssm_1.precision.to_dense()

    prec_2 = ssm_2.precision.to_dense()

    # the equivalent multivariate gaussians of the state space models
    normal = tfp.distributions.MultivariateNormalFullCovariance
    dist_1 = normal(loc=mu_1, covariance_matrix=tf.linalg.inv(prec_1))
    dist_2 = normal(loc=mu_2, covariance_matrix=tf.linalg.inv(prec_2))

    # KL divergence isn't symmetric so check both ways.
    tf_kl, mkf_kl = dist_1.kl_divergence(dist_2), ssm_1.kl_divergence(ssm_2)
    np.testing.assert_allclose(tf_kl, mkf_kl, rtol=1e-3)

    tf_kl, mkf_kl = dist_2.kl_divergence(dist_1), ssm_2.kl_divergence(ssm_1)

    np.testing.assert_allclose(tf_kl, mkf_kl, rtol=1e-3)


def test_self_kl_divergence(with_tf_random_seed, ssm_setup):
    """Test that the KL divergence of an SSM with itself is zero."""
    ssm, _ = ssm_setup

    mkf_kl = ssm.kl_divergence(ssm)
    np.testing.assert_allclose(0.0, mkf_kl, atol=1e-6)


def test_ssm_create_trainable_copy_creates_trainables(ssm_setup):
    """ Test parameters are trainable when a trainable copy of a state space model is created. """
    dist_p, _ = ssm_setup
    assert not dist_p.trainable_variables
    dist_q = dist_p.create_trainable_copy()
    # Check there are correct trainable variables & correctly scoped
    # All the arguments to StateSpace model should be trainable and named as such
    args = inspect.signature(StateSpaceModel)
    basename = f"{dist_q.name}.create_trainable_copy/" if auto_namescope_enabled() else ""
    expected_variables = [basename + arg for arg in args.parameters.keys()]
    for expected in expected_variables:
        assert any(expected in trainable.name for trainable in dist_q.trainable_variables)


@pytest.mark.parametrize("batch_shape", [(3,), (2, 1)])
def test_ssm_log_pdf_evaluation(batch_shape):
    """
    Test the log pdf evaluation for ssm.
    This test samples states (evaluating the pdf along the way)
    The log pdf is compared to 2 alternatives:
    * a method computing each factor of the factorising pdf separately
    * a method calling the log pdf of a multivariate normal with banded precision"""
    state_dim = 2
    # keep trajectories short, computing the covariance from the precision may be inaccurate.
    transitions = 5
    sample_shape = (7, 2)

    # create state space model
    ssm, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()

    # sample and evaluate log pdf along the way
    states = ssm.sample(sample_shape)

    # stack the state vectors
    states_flat = np.reshape(states, sample_shape + batch_shape + ((transitions + 1) * state_dim,))

    # log pdf using the state space representation
    tf_log_pdf_ssm = ssm.log_pdf(states)

    # log pdf using the dense covariance (from the banded precision)
    mu = tf.reshape(ssm.marginal_means, tuple(ssm.batch_shape) + (-1,))
    chol_cov = tf.linalg.cholesky(tf.linalg.inv(ssm.precision.to_dense()))
    normal = tfp.distributions.MultivariateNormalTriL
    tf_log_pdf_dense = normal(loc=mu, scale_tril=chol_cov).log_prob(states_flat)

    # compare log densities
    np.testing.assert_allclose(tf_log_pdf_dense, tf_log_pdf_ssm, rtol=1e-4)
