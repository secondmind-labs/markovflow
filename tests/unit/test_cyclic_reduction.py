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

"""Integration tests for the `CyclicReduction`."""
import inspect

import numpy as np
import pytest
import tensorflow as tf
from gpflow.base import default_float

from markovflow import cyclic_reduction_utils
from markovflow.cyclic_reduction import covariance_blocks_to_CR, precision_to_CR
from markovflow.kernels import Matern32
from tests.tools.generate_random_objects import generate_random_time_points
from tests.tools.state_space_model import StateSpaceModelBuilder


@pytest.fixture(name="cr_setup")
def _cr_setup_fixture(batch_shape):
    return _setup(batch_shape)


def _setup(batch_shape, state_dim=3, transitions=5):
    """Create a CR with a given batch shape."""
    ssm, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()
    return precision_to_CR(ssm.precision, ssm.marginal_means), ssm


def test_log_det(with_tf_random_seed, cr_setup):
    """Verify that the log determinant is correctly calculated."""
    cr, ssm = cr_setup

    log_det_cr = cr.log_det_precision
    log_det_ssm = ssm.log_det_precision

    np.testing.assert_allclose(log_det_cr, log_det_ssm)


def test_precision(with_tf_random_seed, cr_setup):
    """Verify that the precision is correct by calculating it two ways."""
    cr, ssm = cr_setup
    np.testing.assert_allclose(cr.precision.to_dense(), ssm.precision.to_dense())


def test_marginal_means(with_tf_random_seed, cr_setup):
    """ Test that we generate the correct marginal means and covariances. """
    cr, ssm = cr_setup
    np.testing.assert_allclose(ssm.marginal_means, cr.marginal_means)


def test_marginal_covs(with_tf_random_seed, cr_setup):
    """ Test that we generate the correct marginal means and covariances. """
    cr, ssm = cr_setup

    np.testing.assert_allclose(ssm.marginal_covariances, cr.marginal_covariances)


def test_sample(with_tf_random_seed, cr_setup):
    """Test that we can sample from the state space model."""
    cr, ssm = cr_setup

    num_samples = 7
    samples = cr.sample(num_samples)
    assert samples.shape == (num_samples,) + tuple(cr.batch_shape) + (
        cr.num_transitions + 1,
        cr.state_dim,
    )

    sample_shape = (3, 5)
    samples = ssm.sample(sample_shape)
    assert samples.shape == sample_shape + tuple(ssm.batch_shape) + (
        cr.num_transitions + 1,
        cr.state_dim,
    )


def test_create_variable_ssm_same(with_tf_random_seed, cr_setup):
    """Test that the created variable SSM is initialised to be the same."""
    cr, _ = cr_setup
    cr_q = cr.create_trainable_copy()

    k_l = cr.kl_divergence(cr_q)

    initial_kl = k_l
    np.testing.assert_allclose(initial_kl, 0.0, atol=1e-6)


def test_variable_ssm(with_tf_random_seed, cr_setup):
    """Test that the created SSM can be modified."""
    cr, _ = cr_setup
    cr_q = cr.create_trainable_copy()

    cr_q.trainable_variables[1].assign_add(
        tf.ones_like(cr_q.trainable_variables[1], dtype=default_float())
    )

    final_kl = cr_q.kl_divergence(cr)
    # check that the k_l is significantly non-zero
    np.testing.assert_array_less(1e-6, final_kl)


@pytest.mark.parametrize("batch_shape", [(3,), tuple(), (2, 1)])
def test_kl_divergence(with_tf_random_seed, batch_shape):
    """
    Test that we can take the KL divergence of two CyclicReduction instances, and that it's the same
    as for the SSM equivalents
    """
    state_dim = 2
    transitions = 5
    ssm_1, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()
    ssm_2, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()
    cr_1 = precision_to_CR(ssm_1.precision, ssm_1.marginal_means)
    cr_2 = precision_to_CR(ssm_2.precision, ssm_2.marginal_means)

    # KL divergence isn't symmetric so check both ways.
    ssm_kl = ssm_1.kl_divergence(ssm_2)
    cr_kl = cr_1.kl_divergence(cr_2)
    np.testing.assert_allclose(ssm_kl, cr_kl, rtol=1e-3)

    ssm_kl = ssm_2.kl_divergence(ssm_1)
    cr_kl = cr_2.kl_divergence(cr_1)
    np.testing.assert_allclose(ssm_kl, cr_kl, rtol=1e-3)


def test_self_kl_divergence(with_tf_random_seed, cr_setup):
    """Test that the KL divergence of a CR with itself is zero."""
    cr, _ = cr_setup

    mkf_kl = cr.kl_divergence(cr)
    np.testing.assert_allclose(0.0, mkf_kl, atol=1e-6)


@pytest.mark.parametrize("batch_shape", [(3,), (2, 1)])
def test_cr_ssm_log_pdf_evaluation(batch_shape):
    """
    Test the log pdf evaluation for cr vs ssm
    This test samples states (evaluating the pdf along the way)
    """
    state_dim = 2
    # keep trajectories short, computing the covariance from the precision may be inaccurate.
    transitions = 5
    sample_shape = (7, 2)

    # create state space model
    ssm, _ = StateSpaceModelBuilder(batch_shape, state_dim, transitions).build()
    cr = precision_to_CR(ssm.precision, ssm.marginal_means)

    # sample and evaluate log pdf along the way
    states = ssm.sample(sample_shape)

    # log pdf using the state space representation
    log_pdf_ssm = ssm.log_pdf(states)
    log_pdf_cr = cr.log_pdf(states)

    np.testing.assert_allclose(log_pdf_cr, log_pdf_ssm, rtol=1e-4)


def create_upper_blockband_matrix(batch_shape, num_blocks, block_size, is_square):
    Ds = np.random.standard_normal(batch_shape + (num_blocks, block_size, block_size))
    if is_square:
        Os = np.random.standard_normal(batch_shape + (num_blocks - 1, block_size, block_size))
    else:
        Os = np.random.standard_normal(batch_shape + (num_blocks, block_size, block_size))
    return Ds, Os


def upper_blockband_to_dense(Ds, Os):
    num_blocks = Ds.shape[-3]
    block_size = Ds.shape[-2]
    is_square = Ds.shape[-3] == (Os.shape[-3] + 1)

    if is_square:
        shape = Ds.shape[:-3] + (num_blocks * block_size, num_blocks * block_size)
    else:
        shape = Ds.shape[:-3] + (num_blocks * block_size, (num_blocks + 1) * block_size)
    U = np.zeros(shape)
    for i in range(num_blocks):
        U[..., i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size] = Ds[
            ..., i, :, :
        ]
    for i in range(Os.shape[-3]):
        U[
            ..., i * block_size : (i + 1) * block_size, (i + 1) * block_size : (i + 2) * block_size
        ] = Os[..., i, :, :]
    return U


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
@pytest.mark.parametrize("num_blocks", [3, 20, 1])
@pytest.mark.parametrize("block_size", [1, 10])
@pytest.mark.parametrize("is_square", [True, False])
def test_Ux(batch_shape, num_blocks, block_size, is_square):
    Ds, Os = create_upper_blockband_matrix(batch_shape, num_blocks, block_size, is_square)
    U_dense = upper_blockband_to_dense(Ds, Os)
    if is_square:
        rhs_shape = batch_shape + (num_blocks, block_size)
    else:
        rhs_shape = batch_shape + (num_blocks + 1, block_size)
    rhs = np.random.standard_normal(rhs_shape)
    Ux_block = cyclic_reduction_utils.Ux(Ds, Os, rhs)
    Ux_dense = tf.reshape(
        tf.matmul(U_dense, tf.reshape(rhs, batch_shape + (-1, 1))), batch_shape + (-1, block_size)
    )
    np.testing.assert_allclose(Ux_dense.numpy(), Ux_block.numpy())


def get_diag_blocks(A, block_size):
    """
    Given a matrix A, with compatible block size, extract the diagonal blocks.
    """
    assert A.shape[-1] % block_size == 0
    assert A.shape[-2] % block_size == 0
    num_blocks = min(A.shape[-1], A.shape[-2]) // block_size
    blocks = [
        A[..., None, i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size]
        for i in range(num_blocks)
    ]
    return tf.concat(blocks, axis=-3)


def get_upper_offdiag_blocks(A, block_size):
    """
    Given a matrix A, with compatible block size, extract the uppoer-off-diagonal blocks.
    """
    assert A.shape[-1] % block_size == 0
    assert A.shape[-2] % block_size == 0
    return get_diag_blocks(A[..., :, block_size:], block_size)


def get_lower_offdiag_blocks(A, block_size):
    return tf.linalg.matrix_transpose(
        get_upper_offdiag_blocks(tf.linalg.matrix_transpose(A), block_size)
    )


def create_symmetric_blockdiag(batch_shape, num_blocks, block_size):
    tmp = np.random.randn(*(batch_shape + (block_size * num_blocks, (block_size + 1) * num_blocks)))
    S = tf.matmul(tmp, tmp, transpose_b=True)
    Sdiag = get_diag_blocks(S, block_size)
    Soff = get_lower_offdiag_blocks(S, block_size)
    return Sdiag, Soff, S


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
@pytest.mark.parametrize("num_blocks", [3, 20, 1])
@pytest.mark.parametrize("block_size", [1, 10])
@pytest.mark.parametrize("is_square", [True, False])
def test_Utx(batch_shape, num_blocks, block_size, is_square):
    Ds, Os = create_upper_blockband_matrix(batch_shape, num_blocks, block_size, is_square)
    U_dense = upper_blockband_to_dense(Ds, Os)
    rhs = np.random.standard_normal(batch_shape + (num_blocks, block_size))
    Utx_block = cyclic_reduction_utils.Utx(Ds, Os, rhs)
    Utx_dense = tf.reshape(
        tf.matmul(U_dense, tf.reshape(rhs, batch_shape + (-1, 1)), transpose_a=True),
        batch_shape + (-1, block_size),
    )
    np.testing.assert_allclose(Utx_dense.numpy(), Utx_block.numpy())


@pytest.mark.parametrize("is_square", [True, False])
@pytest.mark.parametrize("block_size", [1, 3, 4])
@pytest.mark.parametrize("num_blocks", [2, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
def test_SigU(batch_shape, num_blocks, block_size, is_square):
    U_diag, U_offdiag = create_upper_blockband_matrix(
        batch_shape, num_blocks, block_size, is_square
    )
    U_dense = upper_blockband_to_dense(U_diag, U_offdiag)
    S_diag, S_offdiag, S_dense = create_symmetric_blockdiag(batch_shape, num_blocks, block_size)

    SigU_diag1, SigU_offdiag1 = cyclic_reduction_utils.SigU(S_diag, S_offdiag, U_diag, U_offdiag)
    SigU_dense = tf.matmul(S_dense, U_dense)
    SigU_diag2 = get_diag_blocks(SigU_dense, block_size=block_size)
    SigU_offdiag2 = get_upper_offdiag_blocks(SigU_dense, block_size=block_size)
    np.testing.assert_allclose(SigU_diag1.numpy(), SigU_diag2.numpy())
    np.testing.assert_allclose(SigU_offdiag1.numpy(), SigU_offdiag2.numpy())


@pytest.mark.parametrize("is_square", [True, False])
@pytest.mark.parametrize("block_size", [1, 3, 4])
@pytest.mark.parametrize("num_blocks", [2, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
def test_reverse_SigU(batch_shape, num_blocks, block_size, is_square):
    U_diag, U_offdiag = create_upper_blockband_matrix(
        batch_shape, num_blocks, block_size, is_square
    )

    S_diag, S_offdiag, _ = create_symmetric_blockdiag(batch_shape, num_blocks, block_size)
    SigU_diag, SigU_offdiag = cyclic_reduction_utils.SigU(S_diag, S_offdiag, U_diag, U_offdiag)
    U_diag_reconstructed, U_offdiag_reconstructed = cyclic_reduction_utils.reverse_SigU(
        S_diag, S_offdiag, SigU_diag, SigU_offdiag
    )

    np.testing.assert_allclose(U_diag, U_diag_reconstructed)
    np.testing.assert_allclose(U_offdiag, U_offdiag_reconstructed)


@pytest.mark.parametrize("is_square", [True, False])
@pytest.mark.parametrize("block_size", [1, 3, 4])
@pytest.mark.parametrize("num_blocks", [2, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
def test_UUT(batch_shape, num_blocks, block_size, is_square):
    U_diag, U_offdiag = create_upper_blockband_matrix(
        batch_shape, num_blocks, block_size, is_square
    )
    UUT_diag, UUT_lower = cyclic_reduction_utils.UUt(U_diag, U_offdiag)

    U_dense = upper_blockband_to_dense(U_diag, U_offdiag)
    UUT_dense = tf.matmul(U_dense, U_dense, transpose_b=True)
    UUT_dense_diag = get_diag_blocks(UUT_dense, block_size)
    UUT_dense_lower = get_lower_offdiag_blocks(UUT_dense, block_size)

    np.testing.assert_allclose(UUT_dense_diag, UUT_diag)
    np.testing.assert_allclose(UUT_dense_lower, UUT_lower)


@pytest.mark.parametrize("is_square", [True, False])
@pytest.mark.parametrize("block_size", [1, 3, 4])
@pytest.mark.parametrize("num_blocks", [2, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1), (0,)])
def test_UtV_diags(batch_shape, num_blocks, block_size, is_square):
    U_diag, U_offdiag = create_upper_blockband_matrix(
        batch_shape, num_blocks, block_size, is_square
    )
    V_diag, V_offdiag = create_upper_blockband_matrix(
        batch_shape, num_blocks, block_size, is_square
    )
    UtV_diags = cyclic_reduction_utils.UtV_diags(U_diag, U_offdiag, V_diag, V_offdiag)

    U_dense = upper_blockband_to_dense(U_diag, U_offdiag)
    V_dense = upper_blockband_to_dense(V_diag, V_offdiag)
    UtV_dense = tf.matmul(U_dense, V_dense, transpose_a=True)
    UtV_dense_diags = get_diag_blocks(UtV_dense, block_size)

    np.testing.assert_allclose(UtV_diags, UtV_dense_diags)


@pytest.mark.parametrize("num_time_points", [5, 6, 15, 20])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1)])
def test_kernel_cyclic_reduction(num_time_points, batch_shape):

    shape = batch_shape + (num_time_points,)
    time_points = np.cumsum(np.ones(shape) / num_time_points, axis=-1)
    # time_points = np.linspace(0., 10., num_time_points)
    # leading_dims = len(batch_shape) * [None]
    # time_points = np.tile(time_points[leading_dims], batch_shape + (1,))
    # time_points = generate_random_time_points(expected_range=10., shape=list(shape))

    # declare kernel
    kernel = Matern32(variance=1.0, lengthscale=0.1)

    # construct state space model
    ssm = kernel.state_space_model(time_points=time_points)

    # construct precision from state space model
    cr_ = precision_to_CR(ssm.precision, ssm.marginal_means)

    # direct construction of cyclic reduction decomposition from kernel
    cr = kernel.cyclic_reduction_decomposition(time_points=time_points)

    # check length of the cyclic reduction statistics
    assert len(cr.chols) == len(cr_.chols)
    assert len(cr.Fs) == len(cr_.Fs)
    assert len(cr.Gs) == len(cr_.Gs)

    # check shapes of individual of cyclic reduction statistics per layer
    for a, b in zip(cr.chols, cr_.chols):
        np.testing.assert_array_almost_equal(a, b, decimal=3)

    for a, b in zip(cr.Fs, cr_.Fs):
        np.testing.assert_array_almost_equal(a, b, decimal=3)

    for a, b in zip(cr.Gs, cr_.Gs):
        np.testing.assert_array_almost_equal(a, b, decimal=3)


@pytest.mark.parametrize("num_time_points", [2, 3, 10, 50])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1)])
def test_covariance_to_CR_starting_from_kernel(with_tf_random_seed, num_time_points, batch_shape):
    shape = batch_shape + (num_time_points,)
    time_points = generate_random_time_points(expected_range=10.0, shape=list(shape))

    # declare kernel
    kernel = Matern32(variance=1.0, lengthscale=0.01)

    cr = kernel.cyclic_reduction_decomposition(time_points)
    Sig_diag, Sig_off = cr.covariance_blocks()
    cr2 = covariance_blocks_to_CR(Sig_diag, Sig_off)

    np.testing.assert_equal(len(cr2.Fs), len(cr.Fs))
    np.testing.assert_equal(len(cr2.Gs), len(cr.Gs))
    np.testing.assert_equal(len(cr2.chols), len(cr.chols))
    _ = [np.testing.assert_allclose(F1, F2, atol=1e-7) for F1, F2 in zip(cr2.Fs, cr.Fs)]
    _ = [np.testing.assert_allclose(G1, G2, atol=1e-7) for G1, G2 in zip(cr2.Gs, cr.Gs)]
    _ = [np.testing.assert_allclose(D1, D2, atol=1e-7) for D1, D2 in zip(cr2.chols, cr.chols)]
    np.testing.assert_allclose(cr2.marginal_means, cr.marginal_means)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 1)])
@pytest.mark.parametrize("state_dim", [1, 2, 5])
@pytest.mark.parametrize("num_transitions", [1, 5])
def test_covariance_to_CR_starting_from_ssm(
    with_tf_random_seed, batch_shape, state_dim, num_transitions
):
    ssm, _ = StateSpaceModelBuilder(batch_shape, state_dim, num_transitions).build()
    Sig_diag = ssm.marginal_covariances
    Sig_off = ssm.subsequent_covariances(Sig_diag)
    cr = covariance_blocks_to_CR(Sig_diag, Sig_off)
    Sig_diag_new, Sig_off_new = cr.covariance_blocks()

    np.testing.assert_allclose(Sig_diag_new, Sig_diag)
    np.testing.assert_allclose(Sig_off_new, Sig_off)
