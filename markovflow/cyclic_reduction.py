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

"""Module containing the `Cyclic Reduction` class."""
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from gpflow.base import Parameter, TensorType, default_float
from gpflow.utilities import set_trainable, triangular

from markovflow.block_tri_diag import SymmetricBlockTriDiagonal
from markovflow.cyclic_reduction_utils import SigU, UtV_diags, Utx, UUt, Ux, reverse_SigU
from markovflow.gauss_markov import GaussMarkovDistribution, check_compatible


class CyclicReduction(GaussMarkovDistribution):
    """
    A representation of a Markov-structured Gaussian using cyclic reduction.

    Details on shape convention:

    Cyclic reduction of a linear Gaussian state space model corresponds to an iterative
    binary tree structured factorization of the joint state density p(x₁, ..., xₙ)

    Starting from the full joint p(s), the state vector is split into even and odd indices.
    and a first factorization is made: p(s)= p(xᵉ|xᶜ)p(xᶜ).
    The same factorization procedure is then applied onto p(xᶜ) until it is empty.
    At each iteration (or layer in the tree), statistics of p(xᵉ|xᶜ) are stored.

    The stored statistics are F, G and L where
    * L is the Cholesky factor of the conditional precision xᵉ|xᶜ
    * F and G parameterize the conditional mean through: 𝔼 xᵉ|xᶜ = - L⁻ᵀ Uᵀ xᶜ
    with
    Uᵀ = | F₁ᵀ                [      |]  and  L⁻ᵀ = |L₁⁻ᵀ             |
        |  G₁ᵀ, F₂ᵀ           [      |]             |   L₂⁻ᵀ          |
        |     , G₂ᵀ,⋱         [      |]             |      L₃⁻ᵀ       |
        |           ⋱ ⋱       [      |]             |        ⋱       |
        |             ⋱ Fₙ₋₁ᵀ [      |]             |          ⋱     |
        |               Gₙ₋₁ᵀ [ Fₙᵀ  |]             |            Lₙ⁻ᵀ |
    Last column is either there or not depending of whether nᵉ=nᶜ or nᵉ=nᶜ+1
    """

    def __init__(
        self,
        mean: TensorType,
        chols: tf.RaggedTensor,
        Fs: tf.RaggedTensor,
        Gs: tf.RaggedTensor,
        as_param=False,
    ) -> None:
        """
        """

        super().__init__(self.__class__.__name__)

        # checking the inner dimensions of the cyclic reduction statistics
        # i.e, the shape of the conditioning vectors.

        # tf.debugging.assert_shapes(
        #     [(mean, [..., 'num_data', 'state_dim'])] +
        #     [(chols, [..., 'state_dim', 'state_dim'])] +
        #     [(Fs, [..., 'state_dim', 'state_dim'])] +
        #     [(Gs, [..., 'state_dim', 'state_dim'])]
        # )

        # # checking the number of conditional precision factors L matches the number of data points
        # N = tf.constant(0, dtype=tf.int32)
        # for i in tf.range(chols.nrows()):
        #     N = N + chols[i].to_tensor().shape[-3]
        # tf.assert_equal(N, tf.shape(mean)[-2])

        # # TODO: check shapes of Gs and Ls

        # # store the tensors in self
        # self.mean, self.chols, self.Fs, self.Gs = mean, chols, Fs, Gs

        if as_param:
            with self.name_scope:
                self._Fs = Parameter(Fs.flat_values, name="F")
                self._Gs = Parameter(Gs.flat_values, name="G")
                self._chols = Parameter(chols.flat_values, transform=triangular(), name="chol")
                self.mean = Parameter(mean, name="mean")
        else:
            self._Fs = Fs.flat_values
            self._Gs = Gs.flat_values
            self._chols = chols.flat_values
            self.mean = mean

        self._Fs_nested_row_splits = Fs.nested_row_splits
        self._Gs_nested_row_splits = Gs.nested_row_splits
        self._chols_nested_row_splits = chols.nested_row_splits
        self._tree_depth = chols.nrows()

    @property
    def Fs(self) -> tf.RaggedTensor:
        return tf.RaggedTensor.from_nested_row_splits(self._Fs, self._Fs_nested_row_splits)

    @property
    def Gs(self) -> tf.RaggedTensor:
        return tf.RaggedTensor.from_nested_row_splits(self._Gs, self._Gs_nested_row_splits)

    @property
    def chols(self) -> tf.RaggedTensor:
        return tf.RaggedTensor.from_nested_row_splits(self._chols, self._chols_nested_row_splits)

    @property
    def event_shape(self) -> tf.Tensor:
        """
        The shape of the event in `CyclicReduction` which is [num_transitions + 1, state_dim].
        """
        return tf.shape(self.mean)[-2:]

    @property
    def batch_shape(self) -> tf.TensorShape:
        """
        The shape of any leading dimensions that come before the `event_shape`
        """
        return self.mean.shape[:-2]

    @property
    def state_dim(self) -> int:
        """
        The state dimension (aka blocksize) of the `CyclicReduction`.
        """
        return self.mean.shape[-1]

    @property
    def num_transitions(self) -> tf.Tensor:
        """
        The number of transitions in the Markov chain.
        """
        return tf.shape(self.mean)[-2] - 1

    @property
    def tree_depth(self) -> tf.Tensor:
        """
        The shape of the event in `CyclicReduction` which is [num_transitions + 1, state_dim].
        """
        return self._tree_depth

    @property
    def marginal_means(self) -> tf.Tensor:
        """
        :return: The marginal means of the joint Gaussian with shape:
            batch_shape + [num_transitions + 1, state_dim]
        """
        return self.mean

    # TODO: FIX ME
    @property
    def marginal_covariances(self) -> tf.Tensor:
        """
        :return: The marginal covariances of the joint Gaussian with shape:
            batch_shape + [num_transitions + 1, state_dim, state_dim]
        """
        return self.covariance_blocks()[0]

    @property
    def marginals(self) -> tf.Tensor:
        """
        Return the means μₖ and the covariances Σₖₖ of the marginal distributions
        over consecutive states xₖ.

        :return: The means and covariances with shapes:
                 batch_shape + [num_transitions + 1, state_dim],
                 batch_shape + [num_transitions + 1, state_dim, state_dim]
        """
        return self.marginal_means, self.marginal_covariances

    def _fix_shape(self, tensor: TensorType) -> tf.Tensor:
        return tf.einsum("i...jk->...ijk", tensor)

    def sample(self, sample_shape: Union[Tuple, List, int]) -> tf.Tensor:
        """
        Sample trajectories from the CR distribution
        """
        tf_sample_shape = tf.TensorShape(sample_shape)

        ragged_rank = 1  # len(tf_sample_shape) + len(self.batch_shape) + 1
        ragged_shape = (
            tf.TensorShape([None, None])
            + tf_sample_shape
            + self.batch_shape
            + tf.TensorShape([self.state_dim])
        )
        empty_tensor = tf.zeros(
            shape=(1, 0) + tf_sample_shape + self.batch_shape + (self.state_dim,),
            dtype=default_float(),
        )
        epsilons = tf.RaggedTensor.from_tensor(empty_tensor, ragged_rank=ragged_rank)

        for i in tf.range(self.tree_depth):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(epsilons, ragged_shape)])

            L = self.chols[i]
            shape = tf.concat(
                [(tf.shape(L)[0],), tf_sample_shape, self.batch_shape, (self.state_dim,)], axis=0
            )
            epsilon = tf.random.normal(shape, dtype=default_float())
            epsilons = tf.concat([epsilons, epsilon[None]], axis=0)

        # remove initial dummy entry
        epsilons = epsilons[1:]
        return self._backhalfsolve(epsilons)

    # TODO: FIX ME
    def _halfsolve(self, y):
        """
        Computes x = L⁻¹ y recursively

        L is factorised recursively such that at each level k:
            Lₖ = Pₖ [ Dₖ  0    ]  (Pₖ the odd / even permutation)
                    [ Uₖ  Lₖ₊₁ ]

        This returns the sequence of xₖ built recursively
        [xₖ₊₁] =  Lₖ⁻¹ yₖ =  [ Dₖ⁻¹              0      ] Pₖᵀ yₖ =  [ Dₖ⁻¹ yᵉₖ  ]
        [yₖ₊₁]               [ -Lₖ₊₁⁻¹ Uₖ Dₖ⁻¹    Lₖ₊₁⁻¹ ]           [ Lₖ₊₁⁻¹ (- Uₖ Dₖ⁻¹yᵉₖ + yᶜₖ)]
        starting from y₁ = y

        :param y: shape [...] + self.batch_shape + self.event_shape
        :return: a list of tensors splitting  L⁻¹ y by even indices of the cyclic reduction levels
        """

        ytilde = y
        xs = []

        # Iterate over levels starting from the lowest level (with all nodes)
        for i in range(len(self.chols)):

            # split the odd and even indices
            y_e, y_o = ytilde[..., ::2, :], ytilde[..., 1::2, :]

            # TODO: remove dummy broadcasting once tf2.2 is ready
            dummy_zero = tf.zeros_like(y_e[..., None])
            broadcasted_chol = self.chols[i] + dummy_zero

            # at each level, multiply even states: D⁻¹ yᵉ
            xs.append(tf.linalg.triangular_solve(broadcasted_chol, y_e[..., None])[..., 0])
            if ytilde.shape[0] > 1:
                # at each level, update odd states: yᶜ - U D⁻¹ yᵉ
                ytilde = y_o - Ux(self.Fs[i], self.Gs[i], xs[-1])
            else:
                break

        return xs

    def _backhalfsolve(self, right: tf.RaggedTensor):
        """
        Computes x = L⁻ᵀ y recursively (starting from the end level of the recursion)

        L is factorised recursively such that at each level k:
            Lₖ = Pₖ [ Dₖ  0    ]  (Pₖ the odd / even permutation)
                    [ Uₖ  Lₖ₊₁ ]

        This is used to sample from a mvn with precision P=LLᵀ, using y ~ 𝓝(0, I)

        Input y represents the white noise seed used to sample from the conditionals at each level
        At the top level m : xₘ = Dₘ⁻ᵀ yₘ (single state xₘ)
        One level below: xₖ₋₁ = - Dₖ⁻ᵀ Uₖᵀ Xₖ  +  Dₖ₋₁⁻ᵀ yₖ₋₁  (conditional xₖ₋₁|Xₖ = [xₖ, ..., xₘ])

        These formulas can be read by inverting the recursion:
        [xₖ₋₁, _] = Lₖ⁻ᵀ [yₖ₋₁, Xₖ]
                  = Pₖ⁻ᵀ [ Dₖ⁻ᵀ  -Dₖ⁻ᵀ Uₖᵀ Lₖ₊₁⁻ᵀ ] [yₖ₋₁] = Pₖ [ Dₖ⁻ᵀ(yₖ₋₁ - Uₖᵀ Lₖ₊₁⁻ᵀ Xₖ) ]
                         [ 0              Lₖ₊₁⁻ᵀ ] [Xₖ]         [ Lₖ₊₁⁻ᵀ Xₖ                  ]

        :param right: the cyclic reduction representation of a vector (or vectors to be broadcasted)
        :return: the value of L⁻ᵀ y
        """
        # fix shape for ytilde
        ytilde = tf.einsum("i...j->...ij", right[-1])
        # get the broadcasting shape
        sample_batch_shape = ytilde.shape[:-2]
        # TODO: remove dummy broadcasting once tf2.2 is ready
        dummy_zero = tf.zeros_like(ytilde[..., None])
        broadcasted_chol = self._fix_shape(self.chols[-1]) + dummy_zero

        # Lₖ₊₁⁻ᵀ y₂
        xs = tf.linalg.triangular_solve(broadcasted_chol, ytilde[..., None], adjoint=True)[..., 0]
        xs.set_shape(sample_batch_shape + [None, self.state_dim])

        for i in tf.range(1, self.tree_depth + 1):
            if tf.greater_equal(right.nrows() - i - 1, 0):

                right_iminus1 = tf.einsum("i...j->...ij", right[-i - 1])
                # y₁ - Uᵀ Lₖ₊₁⁻ᵀ y₂
                ytilde = right_iminus1 - Utx(
                    self._fix_shape(self.Fs[-i]), self._fix_shape(self.Gs[-i]), xs
                )
                # TODO: remove dummy broadcasting once tf2.2 is ready
                dummy_zero = tf.zeros_like(ytilde[..., None])
                broadcasted_chol = self._fix_shape(self.chols[-i - 1]) + dummy_zero
                # D⁻ᵀ(y₁ - Uᵀ Lₖ₊₁⁻ᵀ y₂)
                xtilde = tf.linalg.triangular_solve(
                    broadcasted_chol, ytilde[..., None], adjoint=True
                )[..., 0]
                # Pₖ⁻ᵀ [ D⁻ᵀ(y₁ - Uᵀ Lₖ₊₁⁻ᵀ y₂) ;   Lₖ₊₁⁻ᵀ y₂ ]
                xs = interleave(xtilde, xs, axis=-2)
            else:
                break

        return xs

    def covariance_blocks(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        returns diagonal and lower off-diagonal blocks of J⁻¹

        ((Note that J⁻¹ is NOT sparse, so this doesn't tell you everything
        about the inverse of J))
        """

        D = self._fix_shape(self.chols[-1])
        Sig_diag = tf.linalg.cholesky_solve(
            D, tf.eye(self.state_dim, batch_shape=tf.shape(D)[:-2], dtype=default_float())
        )
        Sig_off = tf.zeros(
            shape=self.batch_shape + (0, self.state_dim, self.state_dim), dtype=default_float()
        )

        for i in tf.range(1, self.tree_depth):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (Sig_diag, self.batch_shape + (None, self.state_dim, self.state_dim)),
                    (Sig_off, self.batch_shape + (None, self.state_dim, self.state_dim)),
                ]
            )

            D = self._fix_shape(self.chols[-i - 1])
            F = self._fix_shape(self.Fs[-i])
            G = self._fix_shape(self.Gs[-i])

            # invert D
            # Di = tf.linalg.inv(D)
            # DtiDi = tf.matmul(Di, Di, transpose_a=True)
            DtiDi = tf.linalg.cholesky_solve(
                D, tf.eye(self.state_dim, batch_shape=tf.shape(D)[:-2], dtype=default_float())
            )

            # compute U D⁻¹
            FDi = tf.linalg.matrix_transpose(
                tf.linalg.triangular_solve(
                    D[..., : tf.shape(F)[-3], :, :], tf.linalg.matrix_transpose(F), adjoint=True
                )
            )
            GDi = tf.linalg.matrix_transpose(
                tf.linalg.triangular_solve(
                    D[..., 1:, :, :], tf.linalg.matrix_transpose(G), adjoint=True
                )
            )

            # compute the diagonal and upper-diagonal parts of Sig UD⁻¹
            SUDi_diag, SUDi_off = SigU(-Sig_diag, -Sig_off, FDi, GDi)

            # compute the diagonal parts of D⁻ᵀ Uᵀ Sig UD⁻¹
            UtSUDi_diag = -UtV_diags(FDi, GDi, SUDi_diag, SUDi_off) + DtiDi

            # stitch everything together
            Sig_diag = interleave(UtSUDi_diag, Sig_diag, axis=-3)
            Sig_off = interleave(SUDi_diag, tf.linalg.matrix_transpose(SUDi_off), axis=-3)

        return Sig_diag, Sig_off

    @property
    @tf.function
    def log_det_precision(self) -> tf.Tensor:
        """
        Calculate the Log Determinant of the Precision matrix.

        :return: a tensor of shape: batch_shape
        """
        det = tf.zeros(shape=self.batch_shape, dtype=default_float())
        # for L in self.chols:
        for i in tf.range(self.tree_depth):
            L = self._fix_shape(self.chols[i])
            det = det + tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))), axis=[-1, -2])
        return det

    def create_trainable_copy(self, jitter: float = 0.0) -> "CyclicReduction":
        """
        Create a trainable version of this Cycic Reduction

        This is primarily for use with variational approaches where we want to optimise
        the parameters of a variable `CyclicReduction`, initialised from a prior `CyclicReduction`.

        """
        trainable_cr = CyclicReduction(self.mean, self.chols, self.Fs, self.Gs, as_param=True)
        check_compatible(self, trainable_cr)
        return trainable_cr

        # with self.name_scope:
        #     Fs = Parameter(self._Fs, name='F')
        #     Gs = Parameter(self._Gs, name='G')
        #     chols = Parameter(self._chols, transform=triangular(), name='chol')
        #     mean = Parameter(self.mean, name='mean', trainable=False)

        #     trainable_cr = CyclicReduction(mean, chols, Fs, Gs

        # return self

    def create_non_trainable_copy(self, jitter: float = 0.0) -> "CyclicReduction":
        """
        Create a non trainable version of this Cycic Reduction

        """
        return self.create_trainable_copy()

    def _build_precision(self) -> SymmetricBlockTriDiagonal:
        """
        Computes the compact banded representation of the Precision matrix.

        :return: The precision as a `SymmetricBlockTriDiagonal` object
        """
        L = self._fix_shape(self.chols[-1])
        diag = tf.matmul(L, L, transpose_b=True)
        off_diag = tf.zeros(
            shape=self.batch_shape + (0, self.state_dim, self.state_dim), dtype=default_float()
        )
        off_diag.set_shape(self.batch_shape + (None, self.state_dim, self.state_dim))

        for i in tf.range(1, self.tree_depth):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (diag, self.batch_shape + (None, self.state_dim, self.state_dim)),
                    (off_diag, self.batch_shape + (None, self.state_dim, self.state_dim)),
                ]
            )

            Ks = self._fix_shape(self.chols[-i - 1])
            F = self._fix_shape(self.Fs[-i])
            G = self._fix_shape(self.Gs[-i])

            UUt_diags, _ = UUt(F, G)
            R = tf.matmul(Ks, Ks, transpose_b=True)

            N2 = tf.shape(F)[-3]
            Os_2 = tf.matmul(Ks[..., 1 : N2 + 1, :, :], G, transpose_b=True)
            Os_1 = tf.matmul(F, Ks[..., :N2, :, :], transpose_b=True)

            diag = interleave(R, diag + UUt_diags, axis=-3)
            off_diag = interleave(Os_1, Os_2, axis=-3)

            diag.set_shape(self.batch_shape + (None, self.state_dim, self.state_dim))
            off_diag.set_shape(self.batch_shape + (None, self.state_dim, self.state_dim))

        # make tf aware of the shapes
        diag = tf.reshape(
            diag,
            tf.concat(
                [self.batch_shape, (self.num_transitions + 1, self.state_dim, self.state_dim)],
                axis=0,
            ),
        )
        off_diag = tf.reshape(
            off_diag,
            tf.concat(
                [self.batch_shape, (self.num_transitions, self.state_dim, self.state_dim)], axis=0
            ),
        )
        return SymmetricBlockTriDiagonal(diag, off_diag)

    def mahalanobis(self, states: TensorType) -> tf.Tensor:
        """
        Calculate the mahalanobis distance with the underlying precision.

        :return: a tensor of shape: batch_shape
        """
        leading_shape = states.shape[:-2]
        ytilde = states - self.mean
        mahal = tf.zeros(shape=leading_shape, dtype=default_float())

        for i in tf.range(self.tree_depth - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(ytilde, leading_shape + (None, self.state_dim))]
            )

            L = self._fix_shape(self.chols[i])
            F = self._fix_shape(self.Fs[i])
            G = self._fix_shape(self.Gs[i])
            y_even, y_odd = ytilde[..., ::2, :], ytilde[..., 1::2, :]
            Ltx = tf.matmul(L, y_even[..., None], transpose_a=True)[..., 0]
            mahal = mahal + tf.reduce_sum(tf.square(Ltx + Utx(F, G, y_odd)), axis=[-1, -2])
            ytilde = y_odd

        mahal = mahal + tf.reduce_sum(
            tf.square(
                tf.matmul(self._fix_shape(self.chols[-1]), ytilde[..., None], transpose_a=True)[
                    ..., 0
                ]
            ),
            axis=[-1, -2],
        )
        return mahal

    def log_pdf(self, states):
        """
        Returns the value of the log of the pdf evaluated at states.
        :states: The state trajectory
            [..., ] + batch_shape + [num_transitions + 1, state_dim]
        :return: The log pdf with shape:
            batch_shape
        """
        c = np.log(2 * np.pi) * tf.cast(
            (self.num_transitions + 1) * self.state_dim, default_float()
        )
        return -0.5 * (c - self.log_det_precision + self.mahalanobis(states))

    def kl_divergence(self, dist: GaussMarkovDistribution) -> tf.Tensor:
        """
        Return the KL divergence of the current Gauss Markov distribution from the specified
        input dist,
        i.e. KL(dist₁ ∥ dist₂).

        To do so we first compute the marginal distributions from the Gauss-Markov form, i.e.

            dist₁ = 𝓝(μ₁, P⁻¹₁)
            dist₂ = 𝓝(μ₂, P⁻¹₂)

        where μᵢ are the marginal means and Pᵢ are the banded precisions.

        The KL divergence is then given by:

            KL(dist₁ ∥ dist₂) = ½(tr(P₂P₁⁻¹) + (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁) - N - log(|P₂|) + log(|P₁|))

            N = (num_transitions + 1) * state_dim (i.e. the dimensionality of the Gaussian)

        :param dist: another similarly parameterised Gauss-Markov distribution.

        :return: a tensor of the KL divergences with shape self.batch_shape
        """
        check_compatible(self, dist)
        batch_shape = self.batch_shape

        marginal_covs_1, marginal_offcovs_1 = self.covariance_blocks()
        precision_2 = dist._build_precision()

        # trace term, we use that for any trace tr(AᵀB) = Σᵢⱼ Aᵢⱼ Bᵢⱼ
        # and since the P₂ is symmetric block tri diagonal, we only need the block diagonal and
        # block sub diagonals from from P₁⁻¹
        # this is the sub diagonal of P₁⁻¹, [..., num_transitions, state_dim, state_dim]
        # trace_sub_diag must be added twice as the matrix is symmetric, [...]
        trace = tf.reduce_sum(
            precision_2.block_diagonal * marginal_covs_1, axis=[-3, -2, -1]
        ) + 2.0 * tf.reduce_sum(
            precision_2.block_sub_diagonal * marginal_offcovs_1, axis=[-3, -2, -1]
        )
        tf.debugging.assert_equal(tf.shape(trace), batch_shape)

        # (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁)
        # [... num_transitions + 1, state_dim]
        mahalanobis = dist.mahalanobis(self.marginal_means)
        tf.debugging.assert_equal(tf.shape(mahalanobis), batch_shape)

        dim = (self.num_transitions + 1) * self.state_dim
        dim = tf.cast(dim, default_float())

        k_l = 0.5 * (trace + mahalanobis - dim - dist.log_det_precision + self.log_det_precision)

        tf.debugging.assert_equal(tf.shape(k_l), batch_shape)

        return k_l


def precision_to_CR(precision: SymmetricBlockTriDiagonal, mean=None) -> CyclicReduction:
    """
    Take a precision class and decompose it into a cyclic reduction
    :param precision: `SymmetricBlockTriDiagonal` matrix with
            inner_dim = state_dim, outer_dim = state_dim x num_time_points
    :param mean: tf.Tensor, the mean tensor of shape:
                batch_shape + [num_time_points, state_dim]
    :return : a CyclicReduction object
    """

    Rs = precision.block_diagonal
    Os = precision.block_sub_diagonal  # type: tf.Tensor
    if mean is None:
        mean = tf.zeros(Rs.shape[:-1], dtype=default_float())

    Ds = []
    Fs = []
    Gs = []

    while Rs.shape[-3] > 1:
        # do our part!
        Ks = tf.linalg.cholesky(Rs[..., ::2, :, :])

        Os_1 = Os[..., ::2, :, :]
        Os_2 = Os[..., 1::2, :, :]
        Os_1T = tf.linalg.matrix_transpose(Os_1)

        N2 = Os_1.shape[-3]
        F = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(Ks[..., :N2, :, :], Os_1T))
        G = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(Ks[..., 1 : N2 + 1, :, :], Os_2))
        UUt_diags, UUt_offdiag = UUt(F, G)

        # collect the information
        Ds.append(Ks)
        Fs.append(F)
        Gs.append(G)

        # get the residual
        Rs = Rs[..., 1::2, :, :] - UUt_diags
        Os = -UUt_offdiag

    Ds = Ds + [tf.linalg.cholesky(Rs)]

    return CyclicReduction(mean, Ds, Fs, Gs)


def covariance_blocks_to_CR(Sig_diag: tf.Tensor, Sig_off: tf.Tensor) -> CyclicReduction:
    """
    Take the block diagonals of the inverse of a block-banded precision and reconsctuct the CR.
    """

    mean = tf.zeros_like(Sig_diag[..., 0])
    Ds = []
    Fs = []
    Gs = []

    while Sig_diag.shape[-3] > 1:
        UtSUDi_diag, Sig_diag = Sig_diag[..., ::2, :, :], Sig_diag[..., 1::2, :, :]
        SUDi_diag, SUDi_off = Sig_off[..., ::2, :, :], Sig_off[..., 1::2, :, :]

        n = SUDi_diag.shape[-3]
        m = SUDi_off.shape[-3]

        # get next Sig_off
        chol_UtSUDi_diag = tf.linalg.cholesky(UtSUDi_diag[..., 2 - (n - m) :, :, :])
        half_Sig_off1 = tf.linalg.triangular_solve(
            chol_UtSUDi_diag, tf.linalg.matrix_transpose(SUDi_diag[..., 1:, :, :])
        )
        half_Sig_off2 = tf.linalg.triangular_solve(
            chol_UtSUDi_diag, SUDi_off[..., 1 - (n - m) :, :, :]
        )
        Sig_off = tf.matmul(half_Sig_off1, half_Sig_off2, transpose_a=True)

        SUDi_off = tf.linalg.matrix_transpose(SUDi_off)
        FDi, GDi = reverse_SigU(Sig_diag, Sig_off, -SUDi_diag, -SUDi_off)

        DtiDi = UtSUDi_diag - UtV_diags(FDi, GDi, -SUDi_diag, -SUDi_off)

        # invert DtiDi and get the cholesky
        DtiDi_rev = tf.reverse(DtiDi, axis=[-2, -1])
        L_rev = tf.linalg.cholesky(DtiDi_rev)
        U = tf.reverse(L_rev, axis=[-2, -1])  # this is upper triangular
        eye = tf.eye(tf.shape(DtiDi)[-1], batch_shape=tf.shape(DtiDi)[:-2], dtype=default_float())
        D = tf.linalg.triangular_solve(U, eye, adjoint=True, lower=False)

        # get current D, F, G
        F = tf.matmul(FDi, D[..., : FDi.shape[-3], :, :])
        G = tf.matmul(GDi, D[..., 1:, :, :])

        Ds.append(D)
        Fs.append(F)
        Gs.append(G)

    # compute the cholesky of the last diagonal element
    eye = tf.eye(tf.shape(Sig_diag)[-1], batch_shape=tf.shape(Sig_diag)[:-2], dtype=default_float())
    Sig_diag_rev = tf.reverse(Sig_diag, axis=[-2, -1])
    L_rev = tf.linalg.cholesky(Sig_diag_rev)
    U = tf.reverse(L_rev, axis=[-2, -1])  # this is upper triangular
    D = tf.linalg.triangular_solve(U, eye, adjoint=True, lower=False)
    Ds.append(D)

    return CyclicReduction(mean, Ds, Fs, Gs)


def deconstruct(a: tf.Tensor, axis: int):
    """
    Recursively split a tensor into even and odd indexed entries.

    Example:
     >>> deconstruct(np.arange(10))
     # [<tf.Tensor: shape=(5,), dtype=int64, numpy=array([0, 2, 4, 6, 8])>,
     #  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 5, 9])>,
     #  <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3])>,
     #  <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>]
    """
    residual = a
    ret = []
    a_shape = a.shape
    start_even = tf.zeros_like(a_shape)
    start_odd = (
        tf.TensorShape(tf.zeros_like(a_shape[:axis]))
        + (1,)
        + tf.TensorShape(tf.zeros_like(a_shape[axis + 1 :]))
    )
    stop = a_shape
    stride = (
        tf.TensorShape(tf.ones_like(a_shape[:axis]))
        + (2,)
        + tf.TensorShape(tf.ones_like(a_shape[axis + 1 :]))
    )
    while residual.shape[axis] > 0:
        evens = tf.strided_slice(residual, start_even, stop, stride)
        residual = tf.strided_slice(residual, start_odd, stop, stride)
        ret.append(evens)
    return ret


def reconstruct(dec_a: List, axis: int):
    """
    The inverse of deconstruct: take a list of tensors and recursively interleave them.
    """
    a = dec_a[-1]
    for i in range(1, len(dec_a)):
        a = interleave(dec_a[-1 - i], a, axis)
    return a


def interleave(a: TensorType, b: TensorType, axis):
    """
    Take two tensors and interleave them.

    This is the tensorflow version of
    V = np.empty(a.shape[:axis] + a.shape[axis] + b.shape[axis],) + a.shape[axis + 1:])
    V[..., ::2, ...] = a
    V[..., 1::2, ...] = b
    return V
    """

    # handle numpy arguments
    a = tf.convert_to_tensor(a, dtype=default_float())
    b = tf.convert_to_tensor(b, dtype=default_float())

    # handle axis argument: we can't cope with negative axis because when we do axis+1, we might get
    # zero, which make the behavious weird
    # if axis < 0:
    if axis < 0:
        axis = a.shape.ndims + axis

    # make sure shapes are okay
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    a_bcast_shape = a_shape[:axis]
    b_bcast_shape = a_shape[:axis]
    tf.debugging.assert_equal(a_bcast_shape, b_bcast_shape)
    a_stack_shape = tf.shape(a)[axis + 1 :]
    b_stack_shape = tf.shape(b)[axis + 1 :]
    tf.debugging.assert_equal(a_stack_shape, b_stack_shape)
    n = a_shape[axis]
    m = b_shape[axis]

    # where to start and strid for strided_slice
    start_zero = tf.zeros_like(a_shape)
    stride_one = tf.ones_like(a_shape)
    zeros_stack_shape = tf.zeros_like(a_stack_shape)
    zeros_bcast_shape = tf.zeros_like(a_bcast_shape)

    if tf.less(n, m):
        b_sliced = tf.strided_slice(b, start_zero, a_shape, stride_one)
        # target_shape = a_bcast_shape + (n * 2,) + a_stack_shape
        target_shape = tf.concat([a_bcast_shape, (n * 2,), a_stack_shape], axis=0)
        first_part = tf.reshape(tf.stack([a, b_sliced], axis=axis + 1), target_shape)

        start = tf.concat([zeros_bcast_shape, [n], zeros_stack_shape], axis=0)
        last_bit = tf.strided_slice(b, start, b_shape, stride_one)
        return tf.concat([first_part, last_bit], axis=axis)

    else:
        # target_shape = a_bcast_shape + (m * 2,) + b_stack_shape
        target_shape = tf.concat([a_bcast_shape, (m * 2,), b_stack_shape], axis=0)
        a_sliced = tf.strided_slice(a, start_zero, b_shape, stride_one)
        first_part = tf.reshape(tf.stack([a_sliced, b], axis=axis + 1), target_shape)
        start = tf.concat([zeros_bcast_shape, [m], zeros_stack_shape], axis=0)
        last_bit = tf.strided_slice(a, start, a_shape, tf.ones_like(a_shape))
        return tf.concat([first_part, last_bit], axis=axis)
