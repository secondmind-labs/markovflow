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
"""Module representing block tridiagonal matrices."""
import abc
from abc import abstractmethod
from typing import Optional, Tuple

import tensorflow as tf
from banded_matrices.banded import (
    BandedMatrixTensor,
    band_to_block,
    block_to_band,
    cholesky_band,
    inverse_from_cholesky_band,
    product_band_mat,
    solve_triang_mat,
    unpack_banded_matrix_to_dense,
)

from markovflow.utils import tf_scope_class_decorator, tf_scope_fn_decorator


@tf_scope_class_decorator
class BlockTriDiagonal(abc.ABC):
    """
    Abstract class representing a block tridiagonal matrix.

    All precisions in Markovflow are of this form, so this class provides an adapter between the
    TensorFlow banded_matrices and the MarkovFlow code.
    """

    def __init__(
        self, diagonal: tf.Tensor, symmetric: bool, sub_diagonal: Optional[tf.Tensor] = None
    ) -> None:
        """
        :param diagonal: A tensor with shape ``[... outer_dim, inner_dim, inner_dim]``.
        :param symmetric: Whether the block tridiagonal matrix will be symmetric.
        :param sub_diagonal: A tensor with shape ``[... outer_dim - 1, inner_dim, inner_dim]``.
        """
        self._diag = diagonal
        # check the shape of the diagonal
        tf.debugging.assert_equal(
            self.inner_dim,
            tf.shape(diagonal)[-1],
            message="Last two dimensions of the block diagonal must match.",
        )

        tf.debugging.assert_greater_equal(
            self.outer_dim,
            0,
            message="""Outer dimensions of the block matrix must
                                                     be greater than 0.""",
        )

        if sub_diagonal is not None:
            tf.debugging.assert_greater(
                self.outer_dim,
                1,
                message="""There is no sub-diagonal with outer
                                                   dimension of one.""",
            )

            # check the sub diagonal has the required shape if it exists
            shape = tf.concat(
                [self.batch_shape, [self.outer_dim - 1, self.inner_dim, self.inner_dim]], axis=0,
            )
            tf.debugging.assert_equal(
                tf.shape(sub_diagonal),
                shape,
                message=f"""Sub_diagonal has shape {sub_diagonal.shape}
                                                  but must have shape: {shape}""",
            )

        self._sub_diag = sub_diagonal
        self._symmetric = symmetric

    @property
    def as_band(self) -> BandedMatrixTensor:
        """
        Return a TensorFlow tensor (or NumPy array) representing a banded matrix.

        The (dense) tensor should be of dimension :math:`K×N`, where :math:`K` is the bandwidth
        of the represented :math:`N×N` matrix.
        """
        return self._convert_to_band()

    @property
    def bandwidth(self) -> int:
        """
        Return the (lower) bandwidth of the tensor.
        """
        bandwidth = self.inner_dim - 1
        if self._sub_diag is not None:
            bandwidth += self.inner_dim
        return bandwidth

    @property
    def batch_shape(self) -> tf.TensorShape:
        """
        Return the batch shape of this object.
        """
        return self._diag.shape[:-3]

    @property
    def inner_dim(self) -> int:
        """
        Return the inner dimension of the block tridiagonal matrix. That is,
        the dimensions of the block.
        """
        return self._diag.shape[-2]

    @property
    def outer_dim(self) -> tf.Tensor:
        """
        Return the outer dimension of the block tridiagonal matrix. That is, the number of blocks.
        """
        return tf.shape(self._diag)[-3]

    @property
    def block_diagonal(self) -> tf.Tensor:
        """
        Return the block diagonal.

        :return: Diagonal with shape ``[... outer_dim, inner_dim, inner_dim]``.
        """
        return self._diag

    @property
    def block_sub_diagonal(self) -> Optional[tf.Tensor]:
        """
        Return the block sub-diagonal, if it exists.

        :return: Sub-diagonal with shape ``[... outer_dim, inner_dim, inner_dim]`` or `None`.
        """
        return self._sub_diag

    def to_dense(self) -> tf.Tensor:
        """
        Convert this object to a dense tensor.

        This is useful mainly for debugging and testing purposes.

        :return: A tensor with shape ``[... outer_dim * inner_dim, outer_dim * inner_dim]``
        """
        lower = unpack_banded_matrix_to_dense(
            self.as_band, lower_bandwidth=self.bandwidth, upper_bandwidth=0
        )

        dim = self.outer_dim * self.inner_dim

        shape = tf.concat([self.batch_shape, [dim, dim]], axis=0)
        tf.debugging.assert_equal(tf.shape(lower), shape)

        if self._symmetric:
            return (
                lower
                + tf.linalg.matrix_transpose(lower)
                - tf.linalg.diag(tf.linalg.diag_part(lower))
            )
        return lower

    def dense_mult(self, right: tf.Tensor, transpose_left: bool = False) -> tf.Tensor:
        """
        Multiply a dense vector by this object.

        If this object is :math:`L` and right is :math:`x`, calculate :math:`Lx` as a dense tensor.

        :param right: A tensor with shape ``[... outer_dim, inner_dim]``.
        :param transpose_left: Whether to transpose :math:`L` before multiplying.
        :return: A tensor with shape ``[... outer_dim, inner_dim]``.
        """
        # [..., outer_dim * inner_dim, 1]
        right_flat = self._flatten_right(right)

        # [..., outer_dim * inner_dim, 1]
        prod = product_band_mat(
            self.as_band,
            right_flat,
            left_lower_bandwidth=self.bandwidth,
            left_upper_bandwidth=0,
            transpose_left=transpose_left,
            symmetrise_left=self._symmetric,
        )

        # [..., outer_dim, inner_dim]
        return self._unflatten_right(prod)

    @abstractmethod
    def __add__(self, other):
        """Add two :class:`BlockTriDiagonal` tensors together."""
        raise NotImplementedError

    def _convert_to_band(self) -> BandedMatrixTensor:
        """
        :return: A `BandedMatrixTensor` representation of this matrix
        """
        if self._sub_diag is None:
            # [... outer_dim * inner_dim, inner_dim]
            concatted = tf.reshape(
                tf.linalg.matrix_transpose(self._diag),
                tf.concat(
                    [self.batch_shape, [self.outer_dim * self.inner_dim, self.inner_dim]], axis=0,
                ),
            )
        else:
            shape = tf.concat([self.batch_shape, [1, self.inner_dim, self.inner_dim]], axis=0)
            padding_zeros = tf.zeros(shape, dtype=self._sub_diag.dtype)
            # [... outer_dim, inner_dim, state_dim]
            padded_sub_diag = tf.concat([self._sub_diag, padding_zeros], axis=-3)
            transposed = list(map(tf.linalg.matrix_transpose, [(self._diag), padded_sub_diag]))
            # [... outer_dim * inner_dim, 2 * inner_dim]
            concatted = tf.reshape(
                tf.concat(transposed, axis=-1),
                tf.concat(
                    [self.batch_shape, [self.outer_dim * self.inner_dim, 2 * self.inner_dim]],
                    axis=0,
                ),
            )

        return block_to_band(
            tf.linalg.matrix_transpose(concatted),
            block_size=self.inner_dim,
            symmetric=self._symmetric,
        )

    def _flatten_right(self, right: tf.Tensor) -> tf.Tensor:
        """
        Reshape the rhs for banded_ops compatibility. See also self._assert_compatible_right_shape.

        :param right: the tensor whose shape we want to alter.
        """
        broadcast_shape = tf.shape(right)[:-2]
        right_shape_flat = tf.concat(
            [broadcast_shape, (self.outer_dim * self.inner_dim, 1)], axis=0
        )
        return tf.reshape(right, right_shape_flat)

    def _unflatten_right(self, flat_right: tf.Tensor) -> tf.Tensor:
        """
        Reshape the rhs for banded_ops compatibility. See also self._assert_compatible_right_shape.

        :param flat_right: the tensor whose shape we want to alter.
        """
        broadcast_shape = tf.shape(flat_right)[:-2]
        right_shape = tf.concat([broadcast_shape, (self.outer_dim, self.inner_dim)], axis=0)
        return tf.reshape(flat_right, right_shape)

    def _assert_compatible_right_shape(self, right: tf.Tensor) -> None:
        """
        Make sure the Tensor 'right' is a suitable right-hand-side tensor for
        multiplication and solving. The inner and outer dims should match, and
        the broadcast_dim should be compatible.

        :param right: the tensor whose shape we want to check.
        """
        shape = tf.shape(right)

        # make sure right's outer_dim and inner_dim match
        outer_dim, inner_dim = shape[-2], shape[-1]
        tf.assert_equal([outer_dim, inner_dim], [self.outer_dim, self.inner_dim])

        if right.shape.ndims > 2:
            # right is trying to broadcast over self.batch-shape. Assert that
            # right's implied batch_shape is compatible
            if right.shape.ndims >= (2 + len(self.batch_shape)):
                right_batch_shape = shape[-(2 + len(self.batch_shape)) : -2]
            else:
                right_batch_shape = shape[:-2]

            shapes_match = tf.logical_or(
                tf.equal(right_batch_shape, self.batch_shape),
                tf.logical_or(tf.equal(right_batch_shape, 1), tf.equal(self.batch_shape, 1)),
            )
            tf.assert_equal(tf.reduce_all(shapes_match), True)


@tf_scope_class_decorator
class LowerTriangularBlockTriDiagonal(BlockTriDiagonal):
    """
    Represents a lower triangular block tridiagonal matrix::

               [D₁              ]
               [A₁ D₂           ]
               [    A₂ D₃       ]
               [        ᨞  ᨞    ]
               [         Aₙ₋₁ Dₙ]

    This is typically the Cholesky of a :class:`SymmetricBlockTriDiagonal`.

    Each matrix :math:`Dᵢ` is lower triangular and square with dimension `inner_dim`.
    :math:`Aᵢ` is square with dimension `inner_dim`.

    The `outer_dim` is :math:`n`; that is, the number of block matrices on the main diagonal.

    :math:`Dᵢ` are the diagonal matrices and :math:`Aᵢ` are the sub-diagonal matrices.
    """

    def __init__(self, diagonal: tf.Tensor, sub_diagonal: Optional[tf.Tensor] = None) -> None:
        """
        :param diagonal: A tensor with shape ``[... outer_dim, inner_dim, inner_dim]``.
        :param sub_diagonal: A tensor with shape ``[... outer_dim - 1, inner_dim, inner_dim]``.
        """
        super().__init__(diagonal, symmetric=False, sub_diagonal=sub_diagonal)

    def block_diagonal_of_inverse(self) -> tf.Tensor:
        """
        If this object is :math:`L` and :math:`M = LLᵀ`, return the block diagonal
        elements of :math:`M⁻¹`.

        :math:`M⁻¹` will not, in general, be a banded matrix, and we are normally interested
        only in the diagonal elements.

        :return: A tensor with shape ``[... outer_dim, inner_dim, inner_dim]``.
        """
        # the slice is to only grab the diagonal elements
        # [..., inner_dim, outer_dim * inner_dim]
        band = band_to_block(
            inverse_from_cholesky_band(self.as_band)[..., : self.inner_dim, :], self.inner_dim
        )

        shape = tf.concat(
            [self.batch_shape, [self.outer_dim, self.inner_dim, self.inner_dim]], axis=0
        )
        return tf.reshape(tf.linalg.matrix_transpose(band), shape)

    def solve(self, right: tf.Tensor, transpose_left: bool = False) -> tf.Tensor:
        """
        If this object is :math:`L` and right is :math:`x`, calculate :math:`L⁻¹ x`
        as a dense tensor.

        :param right: A tensor with shape ``[... outer_dim, inner_dim]``.
        :param transpose_left: Whether to transpose :math:`L` before solving.
        :return: A tensor with shape ``[... outer_dim, inner_dim]``.
        """
        self._assert_compatible_right_shape(right)
        right_flat = self._flatten_right(right)
        solved = solve_triang_mat(self.as_band, right_flat, transpose_left)
        return self._unflatten_right(solved)

    def abs_log_det(self) -> tf.Tensor:
        """
        Return the absolute log determinant of this matrix.
        This is just the log of the product of diagonal elements (since it is lower triangular).

        For numerical stability and nicer gradients, we do:

        .. math:: log |L| = Σₙ log |Lₙₙ| =  Σₙ  ½ log |Lₙₙ|²

        :return: A tensor with shape ``batch_shape``, representing the log determinant.
        """
        return 0.5 * tf.reduce_sum(
            input_tensor=tf.math.log(tf.square(self.as_band[..., 0, :])), axis=-1
        )

    def __add__(
        self, other: "LowerTriangularBlockTriDiagonal"
    ) -> "LowerTriangularBlockTriDiagonal":
        """Add two :class:`LowerTriangularBlockTriDiagonal` tensors together."""
        sub_diag = None
        if self._sub_diag is not None:
            sub_diag = self.block_sub_diagonal
            if other.block_sub_diagonal is not None:
                sub_diag += other.block_sub_diagonal
        else:
            sub_diag = other.block_sub_diagonal

        return LowerTriangularBlockTriDiagonal(self.block_diagonal + other.block_diagonal, sub_diag)


@tf_scope_class_decorator
class SymmetricBlockTriDiagonal(BlockTriDiagonal):
    """
    Represents a symmetric block tridiagonal matrix::

               [D₁ A₁ᵀ              ]
               [A₁ D₂  A₂ᵀ          ]
               [    A₂ D₃ A₂ᵀ       ]
               [        ᨞  ᨞   Aₙ₋₁ᵀ]
               [         Aₙ₋₁  Dₙ   ]

    This is the form of the precision matrix for a
    :class:`~markovflow.state_space_model.StateSpaceModel`.

    Each matrix :math:`Dᵢ` is symmetric square with dimension `inner_dim`.
    :math:`Aᵢ` is square with dimension `inner_dim`.

    The `outer_dim` is :math:`n`; that is, the number of block matrices on the main diagonal.

    :math:`Dᵢ` are the diagonal matrices and :math:`Aᵢ` are the sub-diagonal matrices.
    """

    def __init__(self, diagonal: tf.Tensor, sub_diagonal: Optional[tf.Tensor] = None) -> None:
        """
        :param diagonal: A tensor with shape ``[... outer_dim, inner_dim, inner_dim]``.
        :param sub_diagonal: A tensor with shape ``[... outer_dim - 1, inner_dim, inner_dim]``.
        """
        super().__init__(diagonal, symmetric=True, sub_diagonal=sub_diagonal)

    def __add__(self, other: "SymmetricBlockTriDiagonal") -> "SymmetricBlockTriDiagonal":
        """Add two :class:`SymmetricBlockTriDiagonal` tensors together."""
        if self._sub_diag is not None:
            sub_diag = self.block_sub_diagonal
            if other.block_sub_diagonal is not None:
                sub_diag += other.block_sub_diagonal
        else:
            sub_diag = other.block_sub_diagonal

        return SymmetricBlockTriDiagonal(self.block_diagonal + other.block_diagonal, sub_diag)

    @property
    def cholesky(self) -> LowerTriangularBlockTriDiagonal:
        """
        Calculates the Cholesky factorisation of this matrix.

        Cholesky factorisations require a symmetric matrix.
        The `cholesky_band` matrix only operates on the lower triangle, so no copy is needed.

        Cholesky factorisations preserve band structure, so the result is a
        :class:`LowerTriangularBlockTriDiagonal` with the same shape.

        :return: A matrix of the same shape, representing the Cholesky.
        """
        return _banded_to_block_tri(cholesky_band(self.as_band), self.inner_dim)

    def upper_diagonal_lower(
        self,
    ) -> Tuple[LowerTriangularBlockTriDiagonal, LowerTriangularBlockTriDiagonal]:
        r"""
        For this matrix, calculate the :math:`UDUᵀ` factorisation. This is where::

            Uᵀ =  [ I             ]       D = [ D₀          ]
                  [U₁ᵀ, I         ]           [    D₁       ]
                  [    U₂ᵀ, I     ]           [       ᨞     ]
                  [         ᨞  ᨞ ]            [         ᨞  ]
                  [         Uₙᵀ, I]           [           Dₙ]

        :math:`D₀, D₁... Dₙ` are symmetric.

        This is related to the Cholesky :math:`LLᵀ` and :math:`LDLᵀ` decompositions, but is more
        natural when dealing with the inverses of matrices. That is, if
        :math:`K⁻¹ = UDUᵀ` then :math:`K = LD⁻¹Lᵀ` where :math:`L=U⁻ᵀ`.

        This can be used to find the :class:`~markovflow.state_space_model.StateSpaceModel`
        that represents this :class:`SymmetricBlockTriDiagonal`, since:

        .. math:: K⁻¹ = A⁻ᵀ Q⁻¹ A⁻¹

        Hence we can identify the state transition matrices :math:`Aᵢ = -Uᵢᵀ`
        and the initial and process noise covariance matrices :math:`D₀= P₀⁻¹, Dᵢ= Qᵢ⁻¹`
        with the above form:

        .. math::
            K = &| D₀ + U₁ D₁ U₁ᵀ &| U₁ D₁          &| 0...\\
                &| D₁ U₁ᵀ         &| D₁ + U₂ D₂ U₂ᵀ &|  U₂ D₂         &| 0...\\
                &|   0            &| D₂ U₂ᵀ         &| D₂ + D₃ U₃ D₃ᵀ &| U₃ D₃ &| 0...\\
            ...

        We can write the following recurrence relation:

        .. math::
           &Dₙ = Kₙₙ\\
           &Dₖ = Kₖₖ - Kₖₖ₊₁ᵀ Dₖ₊₁⁻¹Kₖₖ₊₁\\
           &Uₖᵀ = Dₖ⁻¹ Kₖₖ₋₁

        ...where :math:`Kⱼₖ` is the block matrix at location j, k of this matrix.
        This method allows us to return the posterior from a
        :class:`~markovflow.kalman_filter.KalmanFilter` as a
        :class:`~markovflow.state_space_model.StateSpaceModel`.

        :return: A tuple of :math:`Uᵀ` and `chol_D`.
        """
        # must have a sub diagonal, otherwise there are no Us
        assert self._sub_diag is not None

        def step(chol_D_s, counter) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Calculate Dₖ iteratively, given by::

                Dₖ = Kₖₖ - Kₖₖ₊₁ᵀ Dₖ₊₁⁻¹Kₖₖ₊₁

            ...where Kⱼₖ is the block matrix at location j, k of this matrix.
            This loop proceeds backwards, from the last state to the first.
            """
            diag_k = self._diag[..., counter, :, :]  # [... inner_dim, inner_dim]
            sub_diag_k = self._sub_diag[..., counter, :, :]  # [... inner_dim, inner_dim]

            # Dₖ₊₁⁻¹Kₖₖ₊₁
            d_inv_k = tf.linalg.cholesky_solve(chol_D_s[..., 0, :, :], sub_diag_k)

            # Dₖ = Kₖₖ - Kₖₖ₊₁ᵀ Dₖ₊₁⁻¹Kₖₖ₊₁ [... inner_dim, inner_dim]
            D_k = diag_k - tf.matmul(sub_diag_k, d_inv_k, transpose_a=True)

            # add chol_Dₖ to the list of chol_Ds
            return (
                tf.concat([tf.linalg.cholesky(D_k[..., None, :, :]), chol_D_s], axis=-3),
                counter - 1,
            )

        # set up the loop variables and shape invariants
        # chol_Dₙ = chol_Kₙₙ [... 1, inner_dim, inner_dim]
        loop_vars = (
            tf.linalg.cholesky(self._diag[..., -1:, :, :]),
            tf.constant(self.outer_dim - 2, tf.int32),
        )

        shape_invars = (
            tf.TensorShape(self.batch_shape + (self.outer_dim, self.inner_dim, self.inner_dim)),
            tf.TensorShape([]),
        )

        # [chol_D₀, chol_D₁, chol_D₂, ....] [... outer_dim, inner_dim, inner_dim]
        chol_d_s, _ = tf.while_loop(
            cond=lambda _, counter: counter >= 0,
            body=step,
            loop_vars=loop_vars,
            shape_invariants=shape_invars,
        )

        identities = tf.eye(
            self.inner_dim,
            dtype=self._diag.dtype,
            batch_shape=self.batch_shape + (self.outer_dim,),
        )

        chol_d_s.set_shape(tf.shape(identities))
        # Uₖᵀ = Dₖ⁻¹ Kₖₖ₋₁ [... outer_dim - 1, inner_dim, inner_dim]
        u_s = tf.linalg.cholesky_solve(chol_d_s[..., 1:, :, :], self._sub_diag)

        return (
            LowerTriangularBlockTriDiagonal(identities, u_s),
            LowerTriangularBlockTriDiagonal(chol_d_s),
        )


@tf_scope_fn_decorator
def _banded_to_block_tri(
    banded: BandedMatrixTensor, block_size: int
) -> LowerTriangularBlockTriDiagonal:
    """
    Convert a banded matrix to a LowerTriangularBlockTriDiagonal.

    NOTE This is designed for internal use with this module as it doesn't check the shapes, so
    may not work for `BandedMatrixTensor`s that aren't LowerTriangularBlockTriDiagonal.
    """
    block = band_to_block(banded, block_size=block_size, symmetric=False)

    batch_shape = block.shape[:-2]
    num_diags_inner = block.shape[-2]
    outer_inner = block.shape[-1]
    outer = outer_inner // block_size
    num_diags = num_diags_inner // block_size
    assert num_diags in (
        1,
        2,
    ), """Banded matrix tensor has " + num_diags + " block diagonals
                                   so isn't block tri diagonal"""

    shape1 = tf.concat([batch_shape, [num_diags, block_size, outer_inner]], axis=0)
    shape2 = tf.concat([batch_shape, [num_diags, outer, block_size, block_size]], axis=0)
    # [... num_diags, outer, block_size, block_size]
    diags = tf.linalg.matrix_transpose(
        # [... num_diags, outer, block_size, block_size]
        tf.reshape(
            # [... num_diags, outer * block_size, block_size]
            tf.linalg.matrix_transpose(
                # [... num_diags, block_size, outer * block_size]
                tf.reshape(block, shape1)
            ),
            shape2,
        )
    )

    sub_diag = None
    if num_diags == 2:
        # [... outer - 1, block_size, block_size]
        sub_diag = diags[..., 1, :-1, :, :]
    diag = diags[..., 0, :, :, :]

    return LowerTriangularBlockTriDiagonal(diag, sub_diag)
