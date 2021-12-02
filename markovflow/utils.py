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
"""Module containing utility functions."""
from functools import wraps
from typing import List, Optional

import tensorflow as tf
from gpflow import default_float
from tensorflow.compat.v1.linalg import (
    LinearOperatorBlockDiag,
    LinearOperatorFullMatrix,
    LinearOperatorKronecker,
)

from markovflow.base import auto_namescope_enabled


def tf_scope_fn_decorator(fn):
    """
    Decorator to wrap the function call in a name_scope
    of the form ".{name of function}".

    The prefix `.` is required because names_scopes cannot be prefixed with `_`.

    Without this some function names (such as private functions) would raise an error.
    """
    if not auto_namescope_enabled():
        return fn

    @wraps(fn)
    def decorated_fn(*args, **kwargs):
        with tf.name_scope(f".{fn.__name__}"):
            return fn(*args, **kwargs)

    return decorated_fn


def tf_scope_class_decorator(cls):
    """
    Decorator to wrap all the methods in a class in a name_scope
    of the form "{name of class}.{name of method}".

    Do not decorate the top level; TensorBoard renders badly if there is only one block.
    """
    if not auto_namescope_enabled():
        return cls

    def decorator(fn, scope_name):
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            with tf.name_scope(scope_name):
                return fn(*args, **kwargs)

        return decorated_fn

    for maybe_fn_name in cls.__dict__:
        if callable(getattr(cls, maybe_fn_name)):
            fn = getattr(cls, maybe_fn_name)
            setattr(cls, maybe_fn_name, decorator(fn, f"{cls.__name__}.{maybe_fn_name}"))
    return cls


def block_diag(matrices: List[tf.Tensor]) -> tf.Tensor:
    """
    Construct block diagonal matrices from a list of batched 2D tensors.

    :param matrices: A list of tensors with shape ``[..., Nᵢ, Mᵢ]``. That is, a list of
        matrices with the same batch dimension.
    :return: A matrix with the input matrices stacked along its main diagonal, with
        shape ``[..., ∑ᵢ,Nᵢ, ∑ᵢ,Mᵢ]``.
    """
    return LinearOperatorBlockDiag(LinearOperatorFullMatrix(m) for m in matrices).to_dense()


def to_delta_time(time_points: tf.Tensor) -> tf.Tensor:
    """
    Convert a tensor of time points to differences between the times. This function returns:

    .. math:: Δtₖ = tₖ₊₁ -  tₖ

    Time points must be a strictly increasing vector, for example:

    .. math:: tₖ₊₁ >  tₖ, so Δtₖ > 0

    :param time_points: A tensor :math:`tₖ` with shape ``[..., a]``.
    :return: A tensor :math:`Δtₖ` with shape ``[..., a - 1]``.
    :raises InvalidArgumentError: Raises if :math:`Δtₖ ≤ 0`.
    """
    delta_t = time_points[..., 1:] - time_points[..., :-1]
    tf.debugging.assert_greater_equal(delta_t, tf.constant(0.0, dtype=delta_t.dtype))
    return tf.identity(delta_t)


def kronecker_product(matrices: List[tf.Tensor]) -> tf.Tensor:
    """
    Return the tensor representing the Kronecker product of the argument matrices.

    :param matrices: The list of matrices to compute the Kronecker product from.
    :return: The Kronecker product tensor.
    """
    return LinearOperatorKronecker([LinearOperatorFullMatrix(m) for m in matrices]).to_dense()


def augment_square_matrix(matrix: tf.Tensor, extra_dim: int, fill_zeros: bool = False) -> tf.Tensor:
    """
    Augment a square matrix to match `state_dim + extra_dim`, where
    `state_dim` is the dimensionality of the inner square matrix of matrix.

    Effectively it creates a block diagonal by padding (if necessary) with an identity or zeros:

    .. math:: matrix -> [[matrix, 0s], [0s, (I or 0s)]]

    :param matrix: A tensor with shape ``[..., state_dim, state_dim]``.
    :param extra_dim: The extra dimension we want to augment it with. If `extra_dim` is :math:`0`,
        the matrix remains unaltered.
    :param fill_zeros: Whether to fill with zeros or identity.
    :return: A tensor with shape ``[..., max_dim, max_dim]``,
        where `max_dim = state_dim + extra_dim`.
    """
    scalar = 0 if fill_zeros else 1
    batch_shape = tf.shape(matrix)[:-2]
    identity = tf.eye(extra_dim, batch_shape=batch_shape, dtype=default_float())
    return block_diag([matrix, scalar * identity])


def augment_matrix(matrix: tf.Tensor, extra_dim: int) -> tf.Tensor:
    """
    Augment a non-square matrix so that the last dimension becomes
    `state_dim + extra_dim`, where `state_dim` is the size of the last dimension of the matrix.

    Effectively it expands the matrix (if necessary) with zeros in the last dimension
    to match `max_dim = state_dim + extra_dim`. In other words:

    .. math:: matrix -> [matrix, 0s]

    :param matrix: A tensor with shape ``[..., state_dim]``.
    :param extra_dim: The extra dimension we want to augment it with. If `extra_dim` is :math:`0`
        the matrix remains unaltered.
    :return: A tensor with shape ``[..., max_dim]``, where `max_dim = state_dim + extra_dim`.
    """
    shape = tf.shape(matrix)[:-1]
    zeros = tf.zeros(tf.concat([shape, [extra_dim]], axis=-1), dtype=default_float())
    return tf.concat([matrix, zeros], axis=-1)


def batch_base_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    r"""
    Given a g1_n and g2_n, and distributions ps and qs such that
      p_n(g2_n) = N(g2_n; 0, Kmm)  (independent of n)
      p_n(g1_n) = N(g1; 0, knn)
      p_n(g1_n | g2_n) = N(g1_n; knm (Kmm⁻¹) g2_n, knn - knm (Kmm⁻¹) kmn)

    And
      q_n(g2_n) = N(g2_n; f_n, q_sqrt_n q_sqrt_nᵀ)

    This method computes the means and (co)variances of
      q_n(g1_n) = ∫ q_n(g2_n) p_n(g1_n| g2_n)

    :param Kmn: [M, ..., N]
    :param Kmm: [M, M]
    :param Knn: [..., N, N]  or  N
    :param f: [M, N]
    :param q_sqrt: If this is a Tensor, it must have shape [N, M, M] (lower
        triangular) or [M, N] (diagonal)
    :param white: bool
    :return: [N,], [N,]
    """
    # compute kernel stuff
    # get the leadings dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Kmm, ["M", "M"]),
        (Knn, [..., "N"]),
        (f, ["M", "N"]),
    ]
    if q_sqrt is not None:
        shape_constraints.append(
            (q_sqrt, (["M", "N"] if q_sqrt.shape.ndims == 2 else ["N", "M", "M"]))
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]
    Lm = tf.linalg.cholesky(Kmm)  # [M, M]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    # compute the covariance due to the conditioning
    fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.reduce_sum(A * f, axis=0)  # [N,]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = q_sqrt * A  # [N, M]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            LTA = tf.einsum("nmo,on->mn", L, A)  # [M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [N,]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for M, N dimensions
        (fmean, [..., "N"]),
        (fvar, [..., "N"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="base_conditional() return values")

    return fmean, fvar
