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
"""
Module transforming identities to and from expectation and natural parameters.
"""

from typing import Tuple

import tensorflow as tf
from banded_matrices.banded import band_to_block, inverse_from_cholesky_band, solve_triang_band
from gpflow.base import TensorType

from markovflow.block_tri_diag import LowerTriangularBlockTriDiagonal, SymmetricBlockTriDiagonal
from markovflow.state_space_model import StateSpaceModel
from markovflow.utils import tf_scope_fn_decorator


@tf_scope_fn_decorator
def ssm_to_expectations(ssm: StateSpaceModel) -> Tuple[TensorType, TensorType, TensorType]:
    r"""
    Transform a :class:`~markovflow.state_space_model.StateSpaceModel` to the expectation
    parameters of the equivalent Gaussian distribution.

    The expectation parameters are defined as the expected value of the sufficient statistics
    :math:`𝔼[φ(x)]`, where :math:`φ(x)` are the sufficient statistics. For the case of a Gaussian
    distribution that is described via a state space model they are given by:

    .. math:: φ(x) = [x, \verb|block_tri_diag|(xxᵀ)]

    The expectation parameters :math:`η` and :math:`Η` are therefore given by::

            [μ₀  ]
            [μ₁  ]
        η = [⋮   ]
            [μₙ₋₁]
            [μₙ  ],

            [Σ₀ + μ₀μ₀ᵀ      Σ₀A₁ᵀ + μ₀μ₁ᵀ                                          ]
            [A₁Σ₀ + μ₁μ₀ᵀ    Σ₁ + μ₁μ₁ᵀ      Σ₁A₂ᵀ + μ₁μ₂ᵀ                          ]
        H = [                    ᨞               ᨞              Σₙ₋₁Aₙᵀ + μₙ₋₁μₙᵀ   ]
            [                                AₙΣₙ₋₁ + μₙμₙ₋₁ᵀ   Σₙ + μₙμₙᵀ          ],

    ...where:

        * :math:`μᵢ` and :math:`Σᵢ` are the marginal means and covariances at each
          time step :math:`i`
        * :math:`Aᵢ` are the transition matrices of the state space model

    :param ssm: The object to transform to expectation parameters.
    :return: A tuple containing the 3 expectation parameters:

        * `eta_linear` corresponds to :math:`η` with shape ``[..., N+1, D]``
        * `eta_diag` corresponds to the block diagonal part of :math:`Η`
          with shape ``[..., N+1, D, D]``
        * `eta_subdiag` corresponds to the lower block sub-diagonal of :math:`Η`
          with shape ``[..., N, D, D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # [..., N+1, D, 1]
    marginal_means = ssm.marginal_means[..., None]
    # [..., N+1, D, D]
    marginal_covs = ssm.marginal_covariances
    # [..., N, D, D]
    As = ssm.state_transitions

    # [..., N+1, D, 1]
    eta_linear = marginal_means[..., 0]
    # [..., N+1, D, D]
    eta_diag = marginal_covs + tf.matmul(marginal_means, marginal_means, transpose_b=True)
    # [..., N, D, D]
    eta_subdiag = As @ marginal_covs[..., :-1, :, :] + tf.matmul(
        marginal_means[..., 1:, :, :], marginal_means[..., :-1, :, :], transpose_b=True
    )

    return eta_linear, eta_diag, eta_subdiag


@tf_scope_fn_decorator
def expectations_to_ssm_params(
    eta_linear: TensorType, eta_diag: TensorType, eta_subdiag: TensorType
) -> Tuple[TensorType, TensorType, TensorType, TensorType, TensorType]:
    r"""
    Transform the expectation parameters to parameters of a
    :class:`~markovflow.state_space_model.StateSpaceModel`.

    The covariance of the joint distribution is given by:

    .. math:: Σ = Η - ηηᵀ

    ...which results in::

            [Σ₀         Σ₀A₁ᵀ       Σ₀A₁ᵀA₂ᵀ    …                               ]
            [A₁Σ₀       Σ₁          Σ₁A₂ᵀ       Σ₁A₂ᵀA₃ᵀ    …                   ]
        Σ = [A₂A₁Σ₀     A₂Σ₁        Σ₂          Σ₂A₃ᵀ       …                   ]
            [⋮          ⋮           ᨞           ᨞           ᨞           Σₙ₋₁Aₙᵀ ]
            [                                   …           AₙΣₙ₋₁      Σₙ      ],

    ...where:

        * :math:`Σᵢ` are the marginal covariances at each time step :math:`i`
        * :math:`Aᵢ` are the transition matrices of the state space model

    If we denote by :math:`Σᵢᵢ₋₁` the lower block sub-diagonal of the joint covariance, and by
    :math:`Σᵢᵢ` the block diagonal of it, then we can get the state space model parameters using
    the following identities:

    .. math::
        &Aᵢ = Σᵢᵢ₋₁ (Σᵢᵢ)⁻¹\\
        &Qᵢ = Σᵢ - AᵢΣᵢ₋₁Aᵢᵀ\\
        &bᵢ = ηᵢ - Aᵢηᵢ₋₁\\
        &P₀ = Σ₀\\
        &μ₀ = η₀

    :param eta_linear: Corresponds to :math:`η` with shape ``[..., N+1, D]``.
    :param eta_diag: Corresponds to the block diagonal part of :math:`Η`
        with shape ``[..., N+1, D, D]``.
    :param eta_subdiag: Corresponds to the lower block sub-diagonal of :math:`Η` with
        shape ``[..., N, D, D]``.
    :return: A tuple containing the 5 parameters of the state space model in the following order:

        * `As` corresponds to the transition matrices :math:`Aᵢ` with shape ``[..., N, D, D]``
        * `offsets` corresponds to the state offset vectors :math:`bᵢ` with shape ``[..., N, D]``
        * `chol_initial_covariance` corresponds to the Cholesky of :math:`P₀`
          with shape ``[..., D, D]``
        * `chol_process_covariances` corresponds to the Cholesky of :math:`Qᵢ`
          with shape ``[..., N, D, D]``
        * `initial_mean` corresponds to the mean of the initial distribution :math:`μ₀`
          with shape ``[..., D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # [..., N+1, D, 1]
    eta_linear = eta_linear[..., None]
    marginal_means = eta_linear
    # [..., N+1, D, D]
    marginal_covs = eta_diag - tf.matmul(eta_linear, eta_linear, transpose_b=True)
    # [..., N, D, D]
    covs_sub_diag = tf.linalg.matrix_transpose(eta_subdiag) - tf.matmul(
        eta_linear[..., :-1, :, :], eta_linear[..., 1:, :, :], transpose_b=True
    )

    marginal_chols = tf.linalg.cholesky(marginal_covs)
    # [..., N, D, D]
    As = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(marginal_chols[..., :-1, :, :], covs_sub_diag)
    )

    # [..., D]
    initial_mean = marginal_means[..., 0, :, 0]
    # [..., D, D]
    chol_initial_covariance = marginal_chols[..., 0, :, :]
    offsets = marginal_means[..., 1:, :, :] - As @ marginal_means[..., :-1, :, :]
    # [..., N, D]
    offsets = offsets[..., 0]

    # [..., N, D, D]
    conditional_covs = marginal_covs[..., 1:, :, :] - As @ tf.matmul(
        marginal_covs[..., :-1, :, :], As, transpose_b=True
    )

    # [..., N, D, D]
    chol_process_covariances = tf.linalg.cholesky(conditional_covs)

    return As, offsets, chol_initial_covariance, chol_process_covariances, initial_mean


@tf_scope_fn_decorator
def ssm_to_naturals(ssm: StateSpaceModel) -> Tuple[TensorType, TensorType, TensorType]:
    """
    Transform a :class:`~markovflow.state_space_model.StateSpaceModel` to the
    natural parameters of the equivalent Gaussian distribution.

    The natural parameters :math:`θ` and :math:`Θ` are given by::

            [P₀⁻¹μ₀ - A₁ᵀQ₁⁻¹b₁     ]
            [Q₁⁻¹b₁ - A₂ᵀQ₂⁻¹b₂     ]
        θ = [⋮                      ]
            [Qₙ₋₁⁻¹bₙ₋₁ - AₙᵀQₙ⁻¹bₙ ]
            [Qₙ⁻¹bₙ                 ],

            [-½(P₀⁻¹ + A₁ᵀ Q₁⁻¹ A₁)     A₁ᵀ Q₁⁻¹                                            ]
            [Q₁⁻¹ A₁                    -½(Q₁⁻¹ + A₂ᵀ Q₂⁻¹ A₂)      A₂ᵀ Q₂⁻¹                ]
        Θ = [                           ᨞                           ᨞               AₙᵀQₙ⁻¹ ]
            [                                                       Qₙ⁻¹Aₙ          -½Qₙ⁻¹  ]

    ...where:

        * :math:`bᵢ`, :math:`Aᵢ` and :math:`Qᵢ` are the state offsets, transition
          matrices and covariances of the state space model
        * :math:`μ₀` and :math:`P₀` are the mean and covariance of the initial state

    :param ssm: The object to transform to natural parameters.
    :return: A tuple containing the 3 natural parameters:

        * `theta_linear` corresponds to :math:`θ` with shape ``[..., N+1, D]``.
        * `theta_diag` corresponds to the block diagonal part of :math:`Θ`
          with shape ``[..., N+1, D, D]``.
        * `theta_subdiag` corresponds to the lower block sub-diagonal of :math:`Θ`
          with shape ``[..., N, D, D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # [..., N, D, D]
    As = ssm.state_transitions
    # [..., N+1, D, 1]
    offsets = ssm.concatenated_state_offsets[..., None]
    # [..., N+1, D, D]
    chols = ssm.concatenated_cholesky_process_covariance

    # [..., N, D, D]
    Linv_As = tf.linalg.triangular_solve(chols[..., 1:, :, :], As)
    theta_subdiag = tf.linalg.triangular_solve(chols[..., 1:, :, :], Linv_As, adjoint=True)

    # [..., N+1, D, 1]
    tmp = tf.linalg.cholesky_solve(chols, offsets)
    # [..., N+1, D, 1]
    theta_linear = tf.concat(
        [
            tmp[..., :-1, :, :] - tf.matmul(As, tmp[..., 1:, :, :], transpose_a=True),
            tmp[..., -1:, :, :],
        ],
        axis=-3,
    )

    # [..., N+1, D]
    theta_linear = theta_linear[..., 0]

    # [..., N, D, D]
    tmp = tf.matmul(Linv_As, Linv_As, transpose_a=True)
    # [..., N+1, D, 1]
    tmp = tf.concat([tmp, tf.zeros_like(tmp[..., :1, :, :])], axis=-3)

    eye = tf.eye(ssm.state_dim, batch_shape=tf.shape(chols)[:-2], dtype=chols.dtype)
    # [..., N+1, D, D]
    precisions = tf.linalg.cholesky_solve(chols, eye)
    # [..., N+1, D, D]
    theta_diag = -0.5 * (precisions + tmp)

    return theta_linear, theta_diag, theta_subdiag


@tf_scope_fn_decorator
def ssm_to_naturals_no_smoothing(
    ssm: StateSpaceModel,
) -> Tuple[TensorType, TensorType, TensorType]:
    """
    Transform a :class:`~markovflow.state_space_model.StateSpaceModel` to the natural
    parameters of the equivalent Gaussian distribution.

    It is similar to :func:`ssm_to_naturals` but in this case the natural
    parameters do not contain information from the future (smoothing). The updates regarding
    the smoothing have been pushed into the partition function, as described in::

        @inproceedings{pmlr-v97-lin19b,
          title = 	 {Fast and Simple Natural-Gradient Variational Inference with Mixture of
                          Exponential-family Approximations},
          author = 	 {Lin, Wu and Khan, Mohammad Emtiyaz and Schmidt, Mark},
          booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
          pages = 	 {3992--4002},
          year = 	 {2019},
          url = 	 {http://proceedings.mlr.press/v97/lin19b.html},
        }

    The natural parameters :math:`θ` and :math:`Θ` are given by::

            [P₀⁻¹μ₀     ]
            [Q₁⁻¹b₁     ]
        θ = [⋮          ]
            [Qₙ₋₁⁻¹bₙ₋₁ ]
            [Qₙ⁻¹bₙ     ],

            [-½P₀⁻¹     A₁ᵀ Q₁⁻¹                            ]
            [Q₁⁻¹ A₁    -½Q₁⁻¹      A₂ᵀ Q₂⁻¹                ]
        Θ = [           ᨞           ᨞               AₙᵀQₙ⁻¹ ]
            [                       Qₙ⁻¹Aₙ          -½Qₙ⁻¹  ]

    ...where:

        * :math:`bᵢ`, :math:`Aᵢ` and :math:`Qᵢ` are the state offsets, transition matrices
          and covariances of the state space model
        * :math:`μ₀` and :math:`P₀` are the mean and covariance of the initial state

    :param ssm: The object to transform to natural parameters.
    :return: A tuple containing the 3 natural parameters:

        * `theta_linear` corresponds to :math:`θ` with shape ``[..., N+1, D]``
        * `theta_diag` corresponds to the block diagonal part of :math:`Θ`
          with shape ``[..., N+1, D, D]``.
        * `theta_subdiag` corresponds to the lower block sub-diagonal of :math:`Θ`
          with shape ``[..., N, D, D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # [..., N, D, D]
    As = ssm.state_transitions
    # [..., N+1, D, 1]
    offsets = ssm.concatenated_state_offsets[..., None]
    # [..., N+1, D, D]
    chols = ssm.concatenated_cholesky_process_covariance

    # [..., N, D, D]
    theta_subdiag = tf.linalg.cholesky_solve(chols[..., 1:, :, :], As)

    # [..., N+1, D, 1]
    theta_linear = tf.linalg.cholesky_solve(chols, offsets)
    # [..., N+1, D]
    theta_linear = theta_linear[..., 0]

    eye = tf.eye(ssm.state_dim, batch_shape=tf.shape(chols)[:-2], dtype=chols.dtype)
    # [..., N+1, D, D]
    precisions = tf.linalg.cholesky_solve(chols, eye)
    # [..., N+1, D, D]
    theta_diag = -0.5 * precisions

    return theta_linear, theta_diag, theta_subdiag


@tf_scope_fn_decorator
def naturals_to_ssm_params(
    theta_linear: TensorType, theta_diag: TensorType, theta_subdiag: TensorType
) -> Tuple[TensorType, TensorType, TensorType, TensorType, TensorType]:
    """
    Transform the natural parameters to parameters of a
    :class:`~markovflow.state_space_model.StateSpaceModel`.

    The precision of the joint distribution is given by::

            [-2Θ₀₀      -Θ₁₀ᵀ                           ]
            [-Θ₁₀       -2Θ₁₁       -Θ₂₁ᵀ               ]
        P = [           ᨞           ᨞           -Θₙₙ₋₁ᵀ ]
            [                       -Θₙₙ₋₁      -2Θₙₙ   ],

    ...where :math:`Θᵢᵢ` and :math:`Θᵢᵢ₋₁` are the block diagonal and block sub-diagonal
    of the natural parameter :math:`Θ`::

            [-½(P₀⁻¹ + A₁ᵀ Q₁⁻¹ A₁)     A₁ᵀ Q₁⁻¹                                            ]
            [Q₁⁻¹ A₁                    -½(Q₁⁻¹ + A₂ᵀ Q₂⁻¹ A₂)      A₂ᵀ Q₂⁻¹                ]
        Θ = [                           ᨞                           ᨞               AₙᵀQₙ⁻¹ ]
            [                                                       Qₙ⁻¹Aₙ          -½Qₙ⁻¹  ],

    ...and where:

        * :math:`Aᵢ` and :math:`Qᵢ` are the state transition matrices and covariances
          of the state space model
        * :math:`P₀` is the covariance of the initial state

    Inverting the precision gives as the joint covariance matrix::

            [Σ₀         Σ₀A₁ᵀ       Σ₀A₁ᵀA₂ᵀ    …                               ]
            [A₁Σ₀       Σ₁          Σ₁A₂ᵀ       Σ₁A₂ᵀA₃ᵀ    …                   ]
        Σ = [A₂A₁Σ₀     A₂Σ₁        Σ₂          Σ₂A₃ᵀ       …                   ]
            [⋮          ⋮           ᨞           ᨞           ᨞           Σₙ₋₁Aₙᵀ ]
            [                                   …           AₙΣₙ₋₁      Σₙ      ],

    ...where:

        * :math:`Σᵢ` are the marginal covariances at each time step :math:`i`
        * :math:`Aᵢ` are the transition matrices of the state space model

    If we define as :math:`Σᵢᵢ₋₁` the lower block sub-diagonal of the joint covariance,
    and as :math:`Σᵢᵢ` the block diagonal of it, we can get the state transition matrices from:

    .. math:: Aᵢ = Σᵢᵢ₋₁ (Σᵢᵢ)⁻¹

    We then follow the SpInGP paper and create the matrices::

               [ I               ]          [P₀             ]
               [-A₁     I        ]          [   Q₁          ]
        A⁻¹ =  [    ᨞       ᨞    ]      Q = [       ᨞       ]
               [        -Aₙ     I]          [           Qₙ  ]

    ...so that:

    .. math:: P = A⁻ᵀQ⁻¹A⁻¹

    If we solve :math:`(A⁻¹)⁻¹ P` we get::

                                         [P₀⁻¹                  ]
                                         [-Q₁⁻¹A₁   Q₁⁻¹        ]
        (A⁻¹)⁻¹ P = Q⁻¹A⁻¹,     Q⁻¹A⁻¹ = [      ᨞       ᨞       ]
                                         [      -Qₙ⁻¹Aₙ     Qₙ⁻¹],

    ...where the block diagonal of :math:`Q⁻¹A⁻¹` holds the process noise precisions :math:`Qᵢ⁻¹`
    and the precision of the initial state :math:`P₀⁻¹`.

    To get the offsets we follow a similar strategy but solve against :math:`θ`. First we write::

            [P₀⁻¹μ₀ - A₁ᵀQ₁⁻¹b₁ ]   [I   -A₁ᵀ     ][P₀⁻¹             ][μ₀]
            [Q₁⁻¹b₁ - A₂ᵀQ₂⁻¹b₂ ]   [    I   -A₂ᵀ ][     Q₁⁻¹        ][b₁]
        θ = [⋮                  ] = [        ᨞   ᨞][         ᨞       ][⋮ ]
            [Qₙ⁻¹bₙ             ]   [            I][             Qₙ⁻¹][bₙ].

    Then we solve :math:`(A⁻ᵀ)⁻¹θ` to get::

                   [P₀⁻¹             ][μ₀]
                   [     Q₁⁻¹        ][b₁]
        (A⁻ᵀ)⁻¹θ = [         ᨞       ][⋮ ]
                   [             Qₙ⁻¹][bₙ].

    Finally, :math:`Q(A⁻ᵀ)⁻¹θ`::

        [μ₀]
        [b₁]
        [⋮ ] = Q(A⁻ᵀ)⁻¹θ.
        [bₙ]

    :param theta_linear: Corresponds to :math:`θ` with shape ``[..., N+1, D]``.
    :param theta_diag: Corresponds to the block diagonal part of :math:`Θ`
        with shape ``[..., N+1, D, D]``.
    :param theta_subdiag: Corresponds to the lower block sub-diagonal
        of :math:`Θ` with shape ``[..., N, D, D]``.
    :return: A tuple containing the 5 parameters of the state space model in the following order:

        * `As` corresponds to the transition matrices :math:`Aᵢ` with shape ``[..., N, D, D]``
        * `offsets` corresponds to the state offset vectors :math:`bᵢ` with shape ``[..., N, D]``
        * `chol_initial_covariance` corresponds to the Cholesky of :math:`P₀`
          with shape ``[..., D, D]``
        * `chol_process_covariances` corresponds to the Cholesky of :math:`Qᵢ`
          with shape ``[..., N, D, D]``
        * `initial_mean` corresponds to the mean of the initial distribution :math:`μ₀`
          with shape ``[..., D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # create the precision from the natural parameters
    precision = SymmetricBlockTriDiagonal(-2 * theta_diag, -theta_subdiag)

    # Get the diag and sub_diag blocks of the covariance
    block = band_to_block(
        inverse_from_cholesky_band(precision.cholesky.as_band), precision.inner_dim
    )
    shape = tf.concat(
        [
            precision.batch_shape,
            [precision.outer_dim, precision.inner_dim, 2 * precision.inner_dim],
        ],
        axis=0,
    )
    cov_blocks = tf.reshape(tf.linalg.matrix_transpose(block), shape)

    # [... N+1, D, D]
    marginal_covs = cov_blocks[..., : precision.inner_dim]
    # [... N+1, D, D]
    sub_diag = cov_blocks[..., -precision.inner_dim :]

    # The tranistions are given by solving the diag against the sub_diag
    # [... N, D, D]
    As = tf.linalg.matrix_transpose(tf.linalg.solve(marginal_covs, sub_diag))[..., :-1, :, :]

    # Create the big A⁻¹ from SpInGP
    eye = tf.eye(
        precision.inner_dim,
        dtype=As.dtype,
        batch_shape=tf.concat([precision.batch_shape, [precision.outer_dim]], axis=0),
    )
    a_inv_block = LowerTriangularBlockTriDiagonal(eye, -As)

    # tmp will have the conditional precisions times the A⁻¹, with the precisions in the diagonal
    tmp = solve_triang_band(
        a_inv_block.as_band,
        precision.as_band,
        right_lower_bandwidth=precision.bandwidth,
        right_upper_bandwidth=0,
        result_lower_bandwidth=precision.bandwidth,
        result_upper_bandwidth=0,
        transpose_left=True,
    )

    # get only the block diag which are the noise process precisions
    tmp_block = band_to_block(tmp, precision.inner_dim)[..., : precision.inner_dim, :]
    # [..., N+1, D, D]
    shape = tf.concat(
        [precision.batch_shape, [precision.outer_dim, precision.inner_dim, precision.inner_dim],],
        axis=0,
    )
    conditional_precisions = tf.reshape(tf.linalg.matrix_transpose(tmp_block), shape)

    # [..., N+1, D, D]
    chol_conditional_precisions = tf.linalg.cholesky(conditional_precisions)
    covariances = tf.linalg.cholesky_solve(chol_conditional_precisions, eye)
    chols = tf.linalg.cholesky(covariances)
    # [..., D, D]
    chol_initial_covariance = chols[..., 0, :, :]
    # [..., N, D, D]
    chol_process_covariances = chols[..., 1:, :, :]

    # [..., N+1, D]
    precision_times_offsets = a_inv_block.solve(theta_linear, transpose_left=True)
    # [..., N+1, D, 1]
    offsets = covariances @ precision_times_offsets[..., None]

    # [..., D]
    initial_mean = offsets[..., 0, :, 0]
    # [..., N, D]
    offsets = offsets[..., 1:, :, 0]

    return As, offsets, chol_initial_covariance, chol_process_covariances, initial_mean


@tf_scope_fn_decorator
def naturals_to_ssm_params_no_smoothing(
    theta_linear: TensorType, theta_diag: TensorType, theta_subdiag: TensorType
) -> Tuple[TensorType, TensorType, TensorType, TensorType, TensorType]:
    """
    Transform the natural parameters to parameters of a
    :class:`~markovflow.state_space_model.StateSpaceModel`.

    This is similar to :func:`naturals_to_ssm_params` but in this case the natural parameters
    do not contain information from the future (smoothing). The updates regarding the
    smoothing have been pushed into the partition function.

    We know that the natural parameters have the following form::

            [-½P₀⁻¹     A₁ᵀ Q₁⁻¹                        ]
            [Q₁⁻¹ A₁    -½Q₁⁻¹      A₂ᵀ Q₂⁻¹            ]
        Θ = [           ᨞           ᨞           AₙᵀQₙ⁻¹ ]
            [                       Qₙ⁻¹Aₙ      -½Qₙ⁻¹  ],

            [P₀⁻¹μ₀]   [P₀⁻¹            ][μ₀]
            [Q₁⁻¹b₁]   [     Q₁⁻¹       ][b₁]
        θ = [⋮     ] = [         ᨞      ][⋮ ]
            [Qₙ⁻¹bₙ]   [            Qₙ⁻¹][bₙ],

    ...where:

        * :math:`bᵢ`, :math:`Aᵢ` and :math:`Qᵢ` are the state offsets, transition matrices
          and covariances of the state space model
        * :math:`μ₀` and :math:`P₀` are the mean and covariance of the initial state

    So by inverting the block diagonal of :math:`Θ` we get the process noise covariance matrices.
    Solving the block diagonal against the sub diagonal yields the state transition matrices.
    Solving the block diagonal of :math:`Θ` against :math:`θ` yields the state offsets and
    the initial mean.

    :param theta_linear: Corresponds to :math:`θ` with shape ``[..., N+1, D]``.
    :param theta_diag: Corresponds to the block diagonal part of :math:`Θ`
        with shape ``[..., N+1, D, D]``.
    :param theta_subdiag: Corresponds to the lower block sub-diagonal of :math:`Θ`
        with shape ``[..., N, D, D]``.
    :return: A tuple containing the 5 parameters of the state space model in the following order:

        * `As` corresponds to the transition matrices :math:`Aᵢ` with shape ``[..., N, D, D]``
        * `offsets` corresponds to the state offset vectors :math:`bᵢ` with shape ``[..., N, D]``
        * `chol_initial_covariance` corresponds to the Cholesky of :math:`P₀`
          with shape ``[..., D, D]``
        * `chol_process_covariances` corresponds to the Cholesky of :math:`Qᵢ`
          with shape ``[..., N, D, D]``
        * `initial_mean` corresponds to the mean of the initial distribution :math:`μ₀`
          with shape ``[..., D]``

        Note each returned object in the tuple is a :data:`~markovflow.base.TensorType`.
    """
    # [..., N+1, D, D]
    chol_conditional_prec = tf.linalg.cholesky(-2 * theta_diag)
    # [..., N, D, D]
    As = tf.linalg.cholesky_solve(chol_conditional_prec[..., 1:, :, :], theta_subdiag)

    # [..., N+1, D, 1]
    offsets = tf.linalg.cholesky_solve(chol_conditional_prec, theta_linear[..., None])
    # [..., D]
    initial_mean = offsets[..., 0, :, 0]
    # [..., N, D]
    offsets = offsets[..., 1:, :, 0]

    eye = tf.eye(
        tf.shape(chol_conditional_prec)[-1],
        batch_shape=tf.shape(chol_conditional_prec[..., 0, 0]),
        dtype=chol_conditional_prec.dtype,
    )
    # [..., N+1, D, D]
    conditional_covs = tf.linalg.cholesky_solve(chol_conditional_prec, eye)
    chols = tf.linalg.cholesky(conditional_covs)

    # [..., D, D]
    chol_initial_covariance = chols[..., 0, :, :]
    # [..., N, D, D]
    chol_process_covariances = chols[..., 1:, :, :]

    return As, offsets, chol_initial_covariance, chol_process_covariances, initial_mean
