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
"""Functions used in tests involving `StateSpaceModels`"""
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from banded_matrices.banded import (
    product_band_band,
    solve_triang_band,
    unpack_banded_matrix_to_dense,
)
from banded_matrices.types import BandedMatrixTensor
from gpflow import default_float

from markovflow.block_tri_diag import LowerTriangularBlockTriDiagonal
from markovflow.emission_model import EmissionModel
from markovflow.state_space_model import StateSpaceModel, state_space_model_from_covariances
from tests.tools.generate_random_objects import generate_random_pos_def_matrix


class StateSpaceModelBuilder:
    """
    A builder object for `StateSpaceModel` objects.
    """

    def __init__(self, batch_shape: Tuple[int], state_dim: int, transitions: int) -> None:
        """
        :param batch_shape: Shape of the batches in the `StateSpaceModel`
        :param state_dim: Length of the state dimension in the `StateSpaceModel`
        :param transitions: Number of transitions in trajectories sampled from the `StateSpaceModel`
        """
        self._batch_shape = batch_shape
        self._state_dim = state_dim
        self._transitions = transitions

        self._P_0 = generate_random_pos_def_matrix(state_dim, batch_shape)
        self._Q_s = generate_random_pos_def_matrix(state_dim, batch_shape + (transitions,))

    def with_P_0(self, P_0: np.ndarray) -> "StateSpaceModelBuilder":
        """ Set the initial covariance matrix. """
        self._P_0 = P_0
        return self

    def with_Q_s(self, Q_s: np.ndarray) -> "StateSpaceModelBuilder":
        """ Set the process covariance matrix. """
        self._Q_s = Q_s
        return self

    def build(self) -> Tuple[StateSpaceModel, Dict[str, Any]]:
        """
        :return: A state space model and the associated feed dict.
        """
        transitions_matrix_shape = (self._transitions, self._state_dim, self._state_dim)

        mu_0 = np.random.normal(size=self._batch_shape + (self._state_dim,))
        b_s = np.random.normal(size=self._batch_shape + (self._transitions, self._state_dim))
        A_s = np.random.normal(size=self._batch_shape + transitions_matrix_shape)

        ssm = state_space_model_from_covariances(
            initial_mean=tf.constant(mu_0),
            initial_covariance=tf.constant(self._P_0),
            state_transitions=tf.constant(A_s),
            state_offsets=tf.constant(b_s),
            process_covariances=tf.constant(self._Q_s),
        )

        return ssm, {"mu_0": mu_0, "P_0": self._P_0, "A_s": A_s, "b_s": b_s, "Q_s": self._Q_s}


def precision_spingp(ssm: StateSpaceModel) -> BandedMatrixTensor:
    """
    This method is used for verifying the implementation of the construction of the precision.

    It follows the SpInGP paper more directly

    K⁻¹ = A⁻ᵀ Q⁻¹ A⁻¹

    where

    A⁻¹ =  [ I             ]      Q⁻¹ =  [ P₀⁻¹          ]
            [-A₁, I         ]            [    Q₁⁻¹       ]
            [    -A₂, I     ]            [       ᨞      ]
            [         ᨞  ᨞  ]            [         ᨞    ]
            [         -Aₙ, I]            [           Qₙ⁻¹]

    We construct the two matrices as `BandedMatrixTensor` and multiply them together.

    :return: The precision as a BandedMatrixTensor with upper bandwidth: 0,
                lower bandwidth: 2 * state_dim - 1
    """

    # create A⁻¹
    #    A⁻¹ =  [ I             ]
    #           [-A₁, I         ]
    #           [    -A₂, I     ]
    #           [         ᨞  ᨞  ]
    #           [         -Aₙ, I]

    # [... num_transitions + 1, state_dim, state_dim]
    identities = tf.eye(
        ssm.state_dim,
        dtype=default_float(),
        batch_shape=tf.concat([ssm.batch_shape, tf.TensorShape([ssm.num_transitions + 1])], axis=0),
    )

    q_inv_block = tf.linalg.cholesky_solve(ssm.concatenated_cholesky_process_covariance, identities)

    a_inv_band = LowerTriangularBlockTriDiagonal(identities, -ssm._A_s)
    q_inv_band = LowerTriangularBlockTriDiagonal(q_inv_block)

    # calculate Q⁻¹A⁻¹
    inv_q_inv_a = product_band_band(
        q_inv_band.as_band,
        a_inv_band.as_band,
        left_lower_bandwidth=q_inv_band.bandwidth,
        left_upper_bandwidth=0,
        right_lower_bandwidth=a_inv_band.bandwidth,
        right_upper_bandwidth=0,
        result_lower_bandwidth=a_inv_band.bandwidth,
        result_upper_bandwidth=q_inv_band.bandwidth,
        symmetrise_left=True,
        transpose_left=False,
        transpose_right=False,
    )

    # calculate A⁻ᵀ(Q⁻¹A⁻¹)
    return product_band_band(
        a_inv_band.as_band,
        inv_q_inv_a,
        left_lower_bandwidth=a_inv_band.bandwidth,
        left_upper_bandwidth=0,
        right_lower_bandwidth=a_inv_band.bandwidth,
        right_upper_bandwidth=q_inv_band.bandwidth,
        result_lower_bandwidth=a_inv_band.bandwidth,
        result_upper_bandwidth=0,
        symmetrise_left=False,
        symmetrise_right=False,
        transpose_left=True,
        transpose_right=False,
    )


def chol_state_covariance(ssm: StateSpaceModel) -> tf.Tensor:
    """
    Create the cholesky of the dense covariance between all pairs of states in a `StateSpaceModel`.

    This method uses the fact that the covariance matrix K is given by:

        Kᵢⱼ = cov(xᵢ, xⱼ) = A Q Aᵀ = L Lᵀ

    where L = A cholQ = (A⁻¹)⁻¹ cholQ and

    A⁻¹ =  [ I             ]      cholQ =  [ cholP₀          ]
            [-A₁, I         ]              [    cholQ₁       ]
            [    -A₂, I     ]              [         ᨞       ]
            [         ᨞  ᨞  ]              [           ᨞     ]
            [         -Aₙ, I]              [           cholQₙ]

    This method constructs A⁻¹ and uses a banded operation to solve against cholQ to return L.

    :param ssm: The state space model
    :return: A tensor A cholQ, of shape batch_shape + [state_dim * num_data, state_dim * num_data]
    """
    # [... num_transitions + 1, state_dim, state_dim]
    identities = tf.eye(
        ssm.state_dim,
        dtype=default_float(),
        batch_shape=tf.concat([ssm.batch_shape, tf.TensorShape([ssm.num_transitions + 1])], axis=0),
    )

    # construct cholQ as a LowerTriangularBlockTriDiagonal
    chol_q = LowerTriangularBlockTriDiagonal(ssm.concatenated_cholesky_process_covariance)

    # construct A⁻¹ as a LowerTriangularBlockTriDiagonal
    a_inv = LowerTriangularBlockTriDiagonal(identities, -ssm._A_s)

    lower_bandwidth = a_inv.outer_dim * a_inv.inner_dim - 1

    # (A⁻¹)⁻¹ cholQ, as a banded matrix batch_shape + [state_dim * num_data, state_dim * num_data]
    a_chol_q_band = solve_triang_band(
        left=a_inv.as_band,
        right=chol_q.as_band,
        right_lower_bandwidth=chol_q.bandwidth,
        right_upper_bandwidth=0,
        result_lower_bandwidth=lower_bandwidth,
        result_upper_bandwidth=0,
    )

    # convert to a dense matrix,  batch_shape + [state_dim * num_data, state_dim * num_data]
    return unpack_banded_matrix_to_dense(
        a_chol_q_band, lower_bandwidth=lower_bandwidth, upper_bandwidth=0
    )


def f_covariances(ssm: StateSpaceModel, emission_model: EmissionModel) -> tf.Tensor:
    """
    Create the dense covariance matrix between all pairs of f's for a `StateSpaceModel`.

    This is calculated as cov(fᵢ, fⱼ) = GKGᵀ, where G = block_diag(H) = I ⨂ H and Kᵢⱼ = cov(xᵢ, xⱼ)

    :param emission_model: the emission_model
    :param ssm: The state space model
    :return: A tensor of shape batch_shape + [output_dim * num_data, output_dim * num_data]
    """
    # [H, 0, ...]
    # [0, H, ...]
    # [     ᨞ ᨞ ]
    # [..., 0, H]
    *_, num_data, __, state_dim = emission_model.emission_matrix.shape.as_list()
    emission_matricies = tf.unstack(emission_model.emission_matrix, axis=-3)

    # TODO: this doesn't work due to a bug in tensorflow.
    # block_h = tf.linalg.LinearOperatorBlockDiag([
    #    tf.linalg.LinearOperatorFullMatrix(emission_matrix)
    #    for emission_matrix in emission_matricies], is_square=False)
    # HL where S = LLᵀ
    # chol_f_covs = block_h.matmul(chol_state_covariance(ssm))
    # GLLᵀGᵀ = S
    # return tf.matmul(chol_f_covs, chol_f_covs, transpose_b=True)

    pre_pad = 0
    post_pad = num_data * state_dim
    padded_emission_matricies = []
    for emission_matrix in emission_matricies:
        *other_dims, d = emission_matrix.shape
        post_pad -= d
        paddings = tf.constant([[0, 0]] * len(other_dims) + [[pre_pad, post_pad]], dtype=tf.int64)
        pre_pad += d
        padded_emission_matricies.append(tf.pad(emission_matrix, paddings, "CONSTANT"))

    block_h = tf.concat(padded_emission_matricies, axis=-2)
    chol_f_covs = tf.matmul(block_h, chol_state_covariance(ssm))
    return tf.matmul(chol_f_covs, chol_f_covs, transpose_b=True)


def stich_state_space_models(ssms: List[StateSpaceModel]):
    """
    Stitches state space models together:
    The first ssm in the list is copied as is,
    transitions from the remaining ssms are appended,
    with their first state distribution discarded
    input:
       [ssm₁(s₁,..., sₙ) = p₁(s₁)Πₖp₁(sₖ₊₁|sₖ),
       ssm₂(sₙ₊₁, ...) = p₂(sₙ₊₁)Πₖp₂(sₖ₊₁|sₖ),
       ...,
       ssmᵢ(...)]
    output:
       ssm(s₁,...) = ssm₁(s₁,..., sₙ)ssm₂(sₙ₊₁,..., sₙ₊ₘ) / p₂(sₙ₊₁)
                  = p₁(s₁) Πₖp₁(sₖ₊₁|sₖ) Πₖp₂(sₖ₊₁|sₖ) ...

    State space models need to have the same batch and state dimensions

    :param ssms: List of i StateSpaceModel with [N₁, ..., Nₖ,..., Nᵢ] nodes
    :return: a StateSpaceModel with N₁ + ... + Nₖ-1 + ... + (Nᵢ-1) nodes
    """

    # check compatibility
    state_dim = ssms[0].state_dim
    batch_shape = ssms[0].batch_shape
    for ssm in ssms[1:]:
        tf.debugging.assert_equal(state_dim, ssm.state_dim)
        tf.debugging.assert_equal(batch_shape, ssm.batch_shape)

    return StateSpaceModel(
        ssms[0].initial_mean,
        ssms[0].cholesky_initial_covariance,
        tf.concat([ssm.state_transitions for ssm in ssms], axis=-3),
        tf.concat([ssm.state_offsets for ssm in ssms], axis=-2),
        tf.concat([ssm.cholesky_process_covariances for ssm in ssms], axis=-3),
    )
