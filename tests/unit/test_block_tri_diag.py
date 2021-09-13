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
"""Module containing the unit tests for the `BlockTriDiagonal` class."""
from typing import Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf

from markovflow.block_tri_diag import LowerTriangularBlockTriDiagonal, SymmetricBlockTriDiagonal

INNER_DIMS = [1, 3]
OUTER_DIMS = [1, 4]


@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", [2, 4])
def test_block_sub_diag(with_tf_random_seed, batch_shape, inner_dim, outer_dim):
    """Test that the `block_sub_diagonal` method works."""
    diag = np.random.normal(size=batch_shape + (outer_dim, inner_dim, inner_dim))

    sub_diag_np = np.random.normal(size=batch_shape + (outer_dim - 1, inner_dim, inner_dim))

    block_tri_diag = LowerTriangularBlockTriDiagonal(tf.constant(diag), tf.constant(sub_diag_np))

    sub_diag_tf = block_tri_diag.block_sub_diagonal
    np.testing.assert_allclose(sub_diag_np, sub_diag_tf)


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_dense(with_tf_random_seed, batch_shape, has_sub_diag, inner_dim, outer_dim):
    """Test that the `to_dense` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    dense_np, block_tri_diag = _generate_random_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag
    )

    dense = block_tri_diag.to_dense()
    np.testing.assert_allclose(dense, dense_np)


@pytest.mark.parametrize("has_sub_diag_1", [True, False])
@pytest.mark.parametrize("has_sub_diag_2", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_add(
    with_tf_random_seed, batch_shape, has_sub_diag_1, has_sub_diag_2, inner_dim, outer_dim
):
    """Test that the addition of two `SymmetricBlockTriDiagonal` matrices works."""
    if (has_sub_diag_1 or has_sub_diag_2) and outer_dim == 1:
        return

    dense_np_1, block_tri_diag_1 = _generate_random_pos_def_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag_1
    )
    dense_np_2, block_tri_diag_2 = _generate_random_pos_def_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag_2
    )
    added = (block_tri_diag_1 + block_tri_diag_2).to_dense()
    np.testing.assert_allclose(added, dense_np_1 + dense_np_2)


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_abs_log_det(with_tf_random_seed, batch_shape, has_sub_diag, inner_dim, outer_dim):
    """Test that the `abs_log_det` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    dense_np, block_tri_diag = _generate_random_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag
    )
    _, log_det_np = np.linalg.slogdet(dense_np)
    log_det = block_tri_diag.abs_log_det()
    np.testing.assert_allclose(log_det, log_det_np)


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_cholesky(with_tf_random_seed, batch_shape, has_sub_diag, inner_dim, outer_dim):
    """Test that the `cholesky` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    dense_np, block_tri_diag = _generate_random_pos_def_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag
    )
    chol_np = np.linalg.cholesky(dense_np)

    chol = block_tri_diag.cholesky.to_dense()
    np.testing.assert_allclose(chol, chol_np, rtol=1e-3)


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("transpose_left", [True, False])
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_solve(
    with_tf_random_seed, batch_shape, has_sub_diag, inner_dim, transpose_left, outer_dim
):
    """Test that the `solve` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    dense_np, block_tri_diag = _generate_random_tri_diag(
        batch_shape, outer_dim, inner_dim, has_sub_diag
    )

    right = np.random.normal(size=batch_shape + (outer_dim, inner_dim))
    solve = block_tri_diag.solve(tf.constant(right), transpose_left=transpose_left)

    if transpose_left:
        einsum_string = "...ji,...j->...i"
    else:
        einsum_string = "...ij,...j->...i"
    solve_np = np.einsum(
        einsum_string,
        np.linalg.inv(dense_np),
        right.reshape(batch_shape + (outer_dim * inner_dim,)),
    )
    np.testing.assert_allclose(solve, solve_np.reshape(batch_shape + (outer_dim, inner_dim)))


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
@pytest.mark.parametrize("symmetrise", [True, False])
@pytest.mark.parametrize("transpose_left", [True, False])
def test_dense_mult(
    with_tf_random_seed,
    batch_shape,
    has_sub_diag,
    inner_dim,
    transpose_left,
    symmetrise,
    outer_dim,
):
    """Test that the `dense_mult` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    if transpose_left and symmetrise:
        # can't symmetrise and transpose at the same time
        return

    if symmetrise:
        dense_np, block_tri_diag = _generate_random_pos_def_tri_diag(
            batch_shape, outer_dim, inner_dim, has_sub_diag
        )
    else:
        dense_np, block_tri_diag = _generate_random_tri_diag(
            batch_shape, outer_dim, inner_dim, has_sub_diag
        )

    right = np.random.normal(size=batch_shape + (outer_dim, inner_dim))
    mult = block_tri_diag.dense_mult(tf.constant(right), transpose_left=transpose_left)

    if transpose_left:
        einsum_string = "...ji,...j->...i"
    else:
        einsum_string = "...ij,...j->...i"

    mult_np = np.einsum(
        einsum_string, dense_np, right.reshape(batch_shape + (outer_dim * inner_dim,))
    )
    np.testing.assert_allclose(mult, mult_np.reshape(batch_shape + (outer_dim, inner_dim)))


@pytest.mark.parametrize("has_sub_diag", [True, False])
@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", OUTER_DIMS)
def test_diagonal_of_inverse(with_tf_random_seed, batch_shape, has_sub_diag, inner_dim, outer_dim):
    """Test that the `block_diagonal_of_inverse` method works."""
    if has_sub_diag and outer_dim == 1:
        return
    dense_np, _ = _generate_random_pos_def_tri_diag(batch_shape, outer_dim, inner_dim, has_sub_diag)
    chol_np = np.linalg.cholesky(dense_np)

    diag, sub_diag = _blocktridiag_from_dense(chol_np, inner_dim, has_sub_diag)
    if has_sub_diag:
        sub_diag = tf.constant(sub_diag)

    block_tri_diag = LowerTriangularBlockTriDiagonal(tf.constant(diag), sub_diag)

    diag_of_inv = block_tri_diag.block_diagonal_of_inverse()

    diag_of_inv_np, _ = _blocktridiag_from_dense(np.linalg.inv(dense_np), inner_dim, False)
    np.testing.assert_allclose(diag_of_inv_np, diag_of_inv, rtol=1e-3)


@pytest.mark.parametrize("inner_dim", INNER_DIMS)
@pytest.mark.parametrize("outer_dim", [3, 5])
def test_upper_diagonal_lower(with_tf_random_seed, batch_shape, inner_dim, outer_dim):
    """Test that the `upper_diagonal_lower` method works."""
    dense_np, block_tri_diag = _generate_random_pos_def_tri_diag(
        batch_shape, outer_dim, inner_dim, True
    )

    lower_tf, diag_tf = block_tri_diag.upper_diagonal_lower()

    lower, diag = lower_tf.to_dense(), diag_tf.to_dense()

    chol_d_u = np.swapaxes(diag, -1, -2) @ lower

    # make sure the lower triangular matrix is actually lower
    np.testing.assert_allclose(lower, np.tril(lower))
    # ensure the block diagonal matrix is actually block diagonal
    assert diag_tf.block_sub_diagonal is None

    # verify that recombining results in the original matrix
    np.testing.assert_allclose(dense_np, np.swapaxes(chol_d_u, -1, -2) @ chol_d_u, rtol=1e-6)


def _to_dense(diag: np.ndarray, sub_diag: Optional[np.ndarray]) -> np.ndarray:
    """ Convert a diagonal and sub-diagonal to a dense matrix. """
    *batch_shape, outer_dim, inner_dim, _ = diag.shape
    dense = np.zeros(batch_shape + [outer_dim * inner_dim, outer_dim * inner_dim])
    for i in range(outer_dim):
        block_start = i * inner_dim
        for j in range(inner_dim):
            for k in range(j + 1):  # only want the lower half of the diagonal matrices
                dense[..., block_start + j, block_start + k] = diag[..., i, j, k]

    if sub_diag is not None:
        for i in range(outer_dim - 1):
            block_start_k = i * inner_dim
            block_start_j = block_start_k + inner_dim
            for j in range(inner_dim):
                for k in range(inner_dim):
                    dense[..., block_start_j + j, block_start_k + k] = sub_diag[..., i, j, k]
    return dense


def _blocktridiag_from_dense(
    array: np.ndarray, inner_dim: int, has_sub_diag: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract the diagonal and sub-diagonal from a dense matrix."""
    *batch_shape, outer_inner, _ = array.shape
    outer_dim = outer_inner // inner_dim

    diag = np.zeros(batch_shape + [outer_dim, inner_dim, inner_dim])
    for i in range(outer_dim):
        block_start = i * inner_dim
        for j in range(inner_dim):
            for k in range(inner_dim):
                diag[..., i, j, k] = array[..., block_start + j, block_start + k]

    sub_diag = None
    if has_sub_diag:
        sub_diag = np.zeros(batch_shape + [outer_dim - 1, inner_dim, inner_dim])
        for i in range(outer_dim - 1):
            block_start_k = i * inner_dim
            block_start_j = block_start_k + inner_dim
            for j in range(inner_dim):
                for k in range(inner_dim):
                    sub_diag[..., i, j, k] = array[..., block_start_j + j, block_start_k + k]
    return diag, sub_diag


def _generate_random_pos_def_tri_diag(
    batch_shape: Tuple, outer_dim: int, inner_dim: int, has_sub_diag: bool
) -> Tuple[np.ndarray, SymmetricBlockTriDiagonal]:
    """
    Create a random tri-diagonal symmetric positive definite matrix.

    This works by creating a lower triangular tri-diagonal matrix and multiplying it by
    its transpose to make a symmetric positive definite tri diagonal.
    """
    diag = np.tril(np.random.normal(loc=1.0, size=batch_shape + (outer_dim, inner_dim, inner_dim)))

    sub_diag_np = None
    if has_sub_diag:
        sub_diag_np = np.random.normal(size=batch_shape + (outer_dim - 1, inner_dim, inner_dim))

    dense_np = _to_dense(diag, sub_diag_np)
    dense_np = np.einsum("...ij,...kj->...ik", dense_np, dense_np)
    diag, sub_diag = _blocktridiag_from_dense(dense_np, inner_dim, has_sub_diag)
    if has_sub_diag:
        sub_diag = tf.constant(sub_diag)

    return dense_np, SymmetricBlockTriDiagonal(tf.constant(diag), sub_diag)


def _generate_random_tri_diag(
    batch_shape: Tuple, outer_dim: int, inner_dim: int, has_sub_diag: bool
) -> Tuple[np.ndarray, LowerTriangularBlockTriDiagonal]:
    diag = np.tril(np.random.normal(size=batch_shape + (outer_dim, inner_dim, inner_dim)))

    sub_diag = None
    sub_diag_np = None
    if has_sub_diag:
        sub_diag_np = np.random.normal(size=batch_shape + (outer_dim - 1, inner_dim, inner_dim))
        sub_diag = tf.constant(sub_diag_np)

    return (
        _to_dense(diag, sub_diag_np),
        LowerTriangularBlockTriDiagonal(tf.constant(diag), sub_diag),
    )
