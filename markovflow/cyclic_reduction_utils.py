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

"""Module containing various utility functions for the `Cyclic Reduction` class."""

from typing import Tuple

import tensorflow as tf
from gpflow.base import TensorType


def Ux(diags: TensorType, offdiags: TensorType, x: TensorType) -> tf.Tensor:
    """
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute U @ x
    """
    n = tf.shape(diags)[-3]
    m = tf.shape(offdiags)[-3]
    tf.debugging.assert_equal(m + 1, tf.shape(x)[-2])
    tf.debugging.assert_greater_equal(
        n, m, message="diag elements must be have same length or 1 longer than off-diag elements"
    )
    tf.debugging.assert_less_equal(
        n - m,
        1,
        message="diag elements must be have same length or 1 longer than off-diag elements",
    )
    # tf.debugging.assert_equal(True, tf.logical_or(n == m, n == (m + 1)))

    Dx = tf.matmul(diags, x[..., :n, :, None])[..., 0]
    Ox = tf.matmul(offdiags, x[..., 1:, :, None])[..., 0]
    return tf.concat([Dx[..., :m, :] + Ox, Dx[..., m:, :]], axis=-2)


def Utx(diags: TensorType, offdiags: TensorType, x: TensorType) -> tf.Tensor:
    """
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute U.T @ x
    """
    n = tf.shape(diags)[-3]
    m = tf.shape(offdiags)[-3]
    tf.debugging.assert_equal(n, tf.shape(x)[-2])
    tf.debugging.assert_equal(True, tf.logical_or(n == m, n == (m + 1)))

    Dx = tf.matmul(diags, x[..., None], transpose_a=True)[..., 0]
    Ox = tf.matmul(offdiags, x[..., :m, :, None], transpose_a=True)[..., 0]
    return tf.concat(
        [Dx[..., :1, :], Dx[..., 1:, :] + Ox[..., : n - 1, :], Ox[..., n - 1 :, :]], axis=-2
    )


def SigU(
    S_diag: TensorType, S_offdiag: TensorType, U_diag: TensorType, U_offdiag: TensorType
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Let Sig be a symmetric block-tridiagonal matrix whose
    - diagonal blocks are S_diag
    - lower off-diagonals are S_offdiag

    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by U_diag
    - upper off-diagonals are given by U_offdiag

    We would like to compute block-tridiagonal blocks of Sig @ U
    """

    n = tf.shape(U_diag)[-3]
    mainline = tf.concat(
        [
            tf.matmul(S_diag[..., :1, :, :], U_diag[..., :1, :, :]),
            (
                tf.matmul(S_diag[..., 1:, :, :], U_diag[..., 1:, :, :])
                + tf.matmul(S_offdiag, U_offdiag[..., : n - 1, :, :])
            ),
        ],
        axis=-3,
    )

    upline = tf.concat(
        [
            (
                tf.matmul(S_diag[..., :-1, :, :], U_offdiag[..., : n - 1, :, :])
                + tf.matmul(S_offdiag, U_diag[..., 1:, :, :], transpose_a=True)
            ),
            tf.matmul(S_diag[..., n - 1 :, :, :], U_offdiag[..., n - 1 :, :, :]),
        ],
        axis=-3,
    )

    return mainline, upline


def reverse_SigU(
    S_diag: TensorType, S_offdiag: TensorType, SUDi_diag: TensorType, SUDi_off: TensorType
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    This function reverses the operation of `SigU`.

    Let Sig be a symmetric block-tridiagonal matrix whose:
    - diagonal blocks are S_diag
    - lower off-diagonals are S_offdiag

    Let SUDi = Sig @ U @ D⁻¹ be an upper block-bidiagonal matrix whose:
    - diagonals are given by SUDi_diag
    - upper off-diagonals are given by SUDi_off

    We would like to compute the upper block-bidiagonal U @ D⁻¹ whose:
    - diagonals are given by UDi_diag
    - upper off-diagonals are given by UDi_offdiag


    In maths we have:

        diag(Sig @ UD⁻¹)     =  [diag(Sig)₁ @ diag(UD⁻¹)₁                                       ]
                                [diag(Sig)₂ @ diag(UD⁻¹)₂ + offdiag(Sig)₁ @ offdiag(UD⁻¹)₁      ]
                                [            ⋮                                                  ]
                                [diag(Sig)ₙ @ diag(UD⁻¹)ₙ + offdiag(Sig)ₙ₋₁ @ offdiag(UD⁻¹)ₙ₋₁  ]

        offdiag(Sig @ UD⁻¹) =   [diag(Sig)₁ @ offdiag(UD⁻¹)₁ + offdiag(Sig)₁ᵀ @ diag(UD⁻¹)₂     ]
                                [diag(Sig)₂ @ offdiag(UD⁻¹)₂ + offdiag(Sig)₂ᵀ @ diag(UD⁻¹)₃     ]
                                [            ⋮                                                  ]
                                [diag(Sig)ₙ @ offdiag(UD⁻¹)ₙ                                    ].

    Let's rewrite the above formulae for simplicity:

        E = [A₁X₁           ]
            [A₂X₂ + B₁Y₁    ]
            [         ⋮     ]
            [AₙXₙ + Bₙ₋₁Yₙ₋₁]

        F = [A₁Y₁ + Bᵀ₁X₂   ]
            [A₂Y₂ + Bᵀ₂X₃   ]
            [         ⋮     ]
            [AₙYₙ           ].

    With simple algebra we can recover the solution:

        X = [A₁⁻¹E₁                                         ]
            [(A₂ - B₁A₁⁻¹B₁ᵀ)⁻¹(E₂ - B₁A₁⁻¹F₁)              ]
            [            ⋮                                  ]
            [(Aₙ - Bₙ₋₁Aₙ₋₁⁻¹Bₙ₋₁ᵀ)⁻¹(Eₙ - Bₙ₋₁Aₙ₋₁⁻¹Fₙ₋₁)  ],

        Y = [A₁⁻¹F₁ - A₁⁻¹B₁ᵀC₂ ]
            [A₂⁻¹F₂ - A₂⁻¹B₂ᵀC₃ ]
            [         ⋮         ]
            [Aₙ⁻¹Fₙ             ].

    """

    n = SUDi_diag.shape[-3]
    m = SUDi_off.shape[-3]

    chols = tf.linalg.cholesky(S_diag)

    half_AiBT = tf.linalg.triangular_solve(
        chols[..., :-1, :, :], tf.linalg.matrix_transpose(S_offdiag)
    )
    BAiBT = tf.matmul(half_AiBT, half_AiBT, transpose_a=True)
    AiBT = tf.linalg.triangular_solve(chols[..., :-1, :, :], half_AiBT, adjoint=True)

    half_AiF = tf.linalg.triangular_solve(chols[..., :-1, :, :], SUDi_off[..., : n - 1, :, :])
    AiF = tf.linalg.cholesky_solve(chols[..., :-1, :, :], SUDi_off[..., : n - 1, :, :])
    BAiF = tf.matmul(half_AiBT, half_AiF, transpose_a=True)

    conditional_chol = tf.linalg.cholesky(
        tf.concat([S_diag[..., :1, :, :], S_diag[..., 1:, :, :] - BAiBT], axis=-3)
    )

    EminusBAiF = tf.concat([SUDi_diag[..., :1, :, :], SUDi_diag[..., 1:, :, :] - BAiF], axis=-3)

    UDi_diag = tf.linalg.cholesky_solve(conditional_chol, EminusBAiF)

    UDi_offdiag = AiF - tf.matmul(AiBT, UDi_diag[..., 1:, :, :])

    if n == m:
        UDi_offdiag = tf.concat(
            [
                UDi_offdiag,
                tf.linalg.cholesky_solve(chols[..., -1:, :, :], SUDi_off[..., n - 1 :, :, :]),
            ],
            axis=-3,
        )

    return UDi_diag, UDi_offdiag


def UtV_diags(
    U_diag: TensorType, U_offdiag: TensorType, V_diag: TensorType, V_offdiag: TensorType
) -> tf.Tensor:
    """
    U is upper diagonal with (U_diag, U_offdiag)

    V is upper didiagonal with (V_diag,V_offdiag)

    We want the diagonal blocks of U.T @ V
    """

    n = tf.shape(U_diag)[-3]
    return tf.concat(
        [
            tf.matmul(U_diag[..., :1, :, :], V_diag[..., :1, :, :], transpose_a=True),
            (
                tf.matmul(U_diag[..., 1:, :, :], V_diag[..., 1:, :, :], transpose_a=True)
                + tf.matmul(
                    U_offdiag[..., : n - 1, :, :], V_offdiag[..., : n - 1, :, :], transpose_a=True
                )
            ),
            tf.matmul(
                U_offdiag[..., n - 1 :, :, :], V_offdiag[..., n - 1 :, :, :], transpose_a=True
            ),
        ],
        axis=-3,
    )


def UUt(diags: TensorType, offdiags: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute the diagonal and lower-off-diagonal blocks of U@U.T
    """
    n = tf.shape(diags)[-3]
    m = tf.shape(offdiags)[-3]
    diag_sq = tf.matmul(diags, diags, transpose_b=True)
    offdiag_sq = tf.matmul(offdiags, offdiags, transpose_b=True)
    UUt_diag = tf.concat([diag_sq[..., :m, :, :] + offdiag_sq, diag_sq[..., m:, :, :]], axis=-3)
    UUt_offdiag = tf.matmul(diags[..., 1:, :, :], offdiags[..., : n - 1, :, :], transpose_b=True)
    return UUt_diag, UUt_offdiag
