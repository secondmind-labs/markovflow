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
"""Module containing a state space model."""
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.base import Parameter, TensorType
from gpflow.utilities import triangular

from markovflow.base import SampleShape
from markovflow.block_tri_diag import LowerTriangularBlockTriDiagonal, SymmetricBlockTriDiagonal
from markovflow.gauss_markov import GaussMarkovDistribution, check_compatible
from markovflow.utils import tf_scope_class_decorator, tf_scope_fn_decorator

tfd = tfp.distributions


@tf_scope_class_decorator
class StateSpaceModel(GaussMarkovDistribution):
    """
    Implements a state space model. This has the following form:

    .. math:: xₖ₊₁ = Aₖ xₖ + bₖ + qₖ

    ...where:

        * :math:`qₖ ~ 𝓝(0, Qₖ)`
        * :math:`x₀ ~ 𝓝(μ₀, P₀)`
        * :math:`xₖ ∈ ℝ^d`
        * :math:`bₖ ∈ ℝ^d`
        * :math:`Aₖ ∈ ℝ^{d × d}`
        * :math:`Qₖ ∈ ℝ^{d × d}`
        * :math:`μ₀ ∈ ℝ^{d × 1}`
        * :math:`P₀ ∈ ℝ^{d × d}`

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The model samples :math:`x₀` with an initial Gaussian distribution in :math:`ℝ^d`
    (in code :math:`d` is `state_dim`).

    The model then proceeds for :math:`n` (`num_transitions`) to generate :math:`[x₁, ... xₙ]`,
    according to the formula above. The marginal distribution of samples at a point :math:`k`
    is a Gaussian with mean :math:`μₖ, Pₖ`.

    This class allows the user to generate samples from this process as well as to calculate the
    marginal distributions for each transition.
    """

    def __init__(
        self,
        initial_mean: TensorType,
        chol_initial_covariance: TensorType,
        state_transitions: TensorType,
        state_offsets: TensorType,
        chol_process_covariances: TensorType,
    ) -> None:
        """
        :param initial_mean: A :data:`~markovflow.base.TensorType` containing the initial mean,
            with shape ``batch_shape + [state_dim]``.
        :param chol_initial_covariance: A :data:`~markovflow.base.TensorType` containing the
            Cholesky of the initial covariance, with shape
            ``batch_shape + [state_dim, state_dim]``. That is, unless the
            initial covariance is zero, in which case it is zero.
        :param state_transitions: A :data:`~markovflow.base.TensorType` containing state transition
            matrices, with shape ``batch_shape + [num_transitions, state_dim, state_dim]``.
        :param state_offsets: A :data:`~markovflow.base.TensorType` containing the process means
            bₖ, with shape ``batch_shape + [num_transitions, state_dim]``.
        :param chol_process_covariances: A :data:`~markovflow.base.TensorType` containing the
            Cholesky of the noise covariance matrices, with shape
            ``batch_shape + [num_transitions, state_dim, state_dim]``. That is, unless the
            noise covariance is zero, in which case it is zero.
        """

        super().__init__(self.__class__.__name__)

        tf.debugging.assert_shapes(
            [
                (initial_mean, [..., "state_dim"]),
                (chol_initial_covariance, [..., "state_dim", "state_dim"]),
                (state_transitions, [..., "num_transitions", "state_dim", "state_dim"]),
                (state_offsets, [..., "num_transitions", "state_dim"]),
                (chol_process_covariances, [..., "num_transitions", "state_dim", "state_dim"]),
            ]
        )

        # assert batch shapes are exactly matching
        shape = tf.shape(initial_mean)[:-1]
        tf.debugging.assert_equal(shape, tf.shape(chol_initial_covariance)[:-2])
        tf.debugging.assert_equal(shape, tf.shape(state_transitions)[:-3])
        tf.debugging.assert_equal(shape, tf.shape(state_offsets)[:-2])
        tf.debugging.assert_equal(shape, tf.shape(chol_process_covariances)[:-3])

        # store the tensors in self
        self._mu_0 = initial_mean
        self._A_s = state_transitions

        self._chol_P_0 = chol_initial_covariance
        self._chol_Q_s = chol_process_covariances
        self._b_s = state_offsets

    @property
    def event_shape(self) -> tf.Tensor:
        """
        Return the shape of the event.

        :return: The shape is ``[num_transitions + 1, state_dim]``.
        """
        return tf.shape(self.concatenated_state_offsets)[-2:]

    @property
    def batch_shape(self) -> tf.TensorShape:
        """
        Return the shape of any leading dimensions that come before :attr:`event_shape`.
        """
        return self._A_s.shape[:-3]

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension.
        """
        return self._A_s.shape[-2]

    @property
    def num_transitions(self) -> tf.Tensor:
        """
        Return the number of transitions.
        """
        return tf.shape(self._A_s)[-3]

    @property
    def cholesky_process_covariances(self) -> TensorType:
        """
        Return the Cholesky of :math:`[Q₁, Q₂, ....]`.

        :return: A :data:`~markovflow.base.TensorType` with
            shape ``[... num_transitions, state_dim, state_dim]``.
        """
        return self._chol_Q_s

    @property
    def cholesky_initial_covariance(self) -> TensorType:
        """
        Return the Cholesky of :math:`P₀`.

        :return: A :data:`~markovflow.base.TensorType` with shape ``[..., state_dim, state_dim]``.
        """
        return self._chol_P_0

    @property
    def initial_covariance(self) -> tf.Tensor:
        """
        Return :math:`P₀`.

        :return: A :data:`~markovflow.base.TensorType` with shape ``[..., state_dim, state_dim]``.
        """
        return self._chol_P_0 @ tf.linalg.matrix_transpose(self._chol_P_0)

    @property
    def concatenated_cholesky_process_covariance(self) -> tf.Tensor:
        """
        Return the Cholesky of :math:`[P₀, Q₁, Q₂, ....]`.

        :return: A tensor with shape ``[... num_transitions + 1, state_dim, state_dim]``.
        """
        return tf.concat([self._chol_P_0[..., None, :, :], self._chol_Q_s], axis=-3)

    @property
    def state_offsets(self) -> TensorType:
        """
        Return the state offsets :math:`[b₁, b₂, ....]`.

        :return: A :data:`~markovflow.base.TensorType` with
            shape ``[..., num_transitions, state_dim]``.
        """
        return self._b_s

    @property
    def initial_mean(self) -> TensorType:
        """
        Return the initial mean :math:`μ₀`.

        :return: A :data:`~markovflow.base.TensorType` with shape ``[..., state_dim]``.
        """
        return self._mu_0

    @property
    def concatenated_state_offsets(self) -> tf.Tensor:
        """
        Return the concatenated state offsets :math:`[μ₀, b₁, b₂, ....]`.

        :return: A tensor with shape ``[... num_transitions + 1, state_dim]``.
        """
        return tf.concat([self._mu_0[..., None, :], self._b_s], axis=-2)

    @property
    def state_transitions(self) -> TensorType:
        """
        Return the concatenated state offsets :math:`[A₀, A₁, A₂, ....]`.

        :return: A :data:`~markovflow.base.TensorType` with
            shape ``[... num_transitions, state_dim, state_dim]``.
        """
        return self._A_s

    @property
    def marginal_means(self) -> tf.Tensor:
        """
        Return the mean of the marginal distributions at each time point. If:

        .. math:: xₖ ~ 𝓝(μₖ, Kₖₖ)

        ...then return :math:`μₖ`.

        If we let the concatenated state offsets be :math:`m = [μ₀, b₁, b₂, ....]` and :math:`A`
        be defined as in equation (5) of the SpInGP paper (see class docstring), then:

        .. math:: μ = A m = (A⁻¹)⁻¹ m

        ...which we can do quickly using :meth:`a_inv_block`.

        :return: The marginal means of the joint Gaussian, with shape
            ``batch_shape + [num_transitions + 1, state_dim]``.
        """
        # (A⁻¹)⁻¹ m: batch_shape + [num_transitions + 1, state_dim]
        return self.a_inv_block.solve(self.concatenated_state_offsets)

    @property
    def marginal_covariances(self) -> tf.Tensor:
        """
        Return the ordered covariances :math:`Σₖₖ` of the multivariate normal marginal
        distributions over consecutive states :math:`xₖ`.

        :return: The marginal covariances of the joint Gaussian, with shape
            ``batch_shape + [num_transitions + 1, state_dim, state_dim]``.
        """
        return self.precision.cholesky.block_diagonal_of_inverse()

    def covariance_blocks(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Return the diagonal and lower off-diagonal blocks of the covariance.

        :return: A tuple of tensors with respective shapes
                ``batch_shape + [num_transitions + 1, state_dim]``,
                ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        return (
            self.marginal_covariances,
            self.subsequent_covariances(self.marginal_covariances),
        )

    @property
    def a_inv_block(self) -> LowerTriangularBlockTriDiagonal:
        """
        Return :math:`A⁻¹`.

        This has the form::

            A⁻¹ =  [ I             ]
                   [-A₁, I         ]
                   [    -A₂, I     ]
                   [         ᨞  ᨞  ]
                   [         -Aₙ, I]

        ...where :math:`[A₁, ..., Aₙ]` are the state transition matrices.
        """
        # create the diagonal of A⁻¹
        batch_shape = tf.concat([self.batch_shape, self.event_shape[:-1]], axis=0)
        identities = tf.eye(self.state_dim, dtype=default_float(), batch_shape=batch_shape)
        # A⁻¹
        return LowerTriangularBlockTriDiagonal(identities, -self._A_s)

    def sample(self, sample_shape: SampleShape) -> tf.Tensor:
        """
        Return sample trajectories.

        :param sample_shape: The shape (and hence number of) trajectories to sample from
            the state space model.
        :return: A tensor containing state samples, with shape
            ``sample_shape + self.batch_shape + self.event_shape``.
        """
        sample_shape = tf.TensorShape(sample_shape)
        full_sample_shape = tf.concat(
            [sample_shape, self.batch_shape, self.event_shape, tf.TensorShape([1])], axis=0
        )

        epsilons = tf.random.normal(full_sample_shape, dtype=default_float())
        b = self.concatenated_state_offsets
        z = tf.matmul(self.concatenated_cholesky_process_covariance, epsilons)[..., 0]
        conditional_epsilons = b + z

        # handle the case of zero sample size: this array has no elements!
        if conditional_epsilons.shape.num_elements() == 0:
            return conditional_epsilons

        # (A⁻¹)⁻¹ m: sample_shape + self.batch_shape + self.event_shape
        samples = self.a_inv_block.solve(conditional_epsilons)

        return samples

    def subsequent_covariances(self, marginal_covariances: tf.Tensor) -> tf.Tensor:
        """
        For each pair of subsequent states :math:`xₖ, xₖ₊₁`, return the covariance of their joint
        distribution. That is:

        .. math:: Cov(xₖ₊₁, xₖ) = AₖPₖ

        :param marginal_covariances: The marginal covariances of each state in the model,
            with shape ``batch_shape + [num_transitions + 1, state_dim, state_dim]``.
        :return: The covariance between subsequent state, with shape
                 ``batch_shape + [num_transitions, state_dim, state_dim]``.
        """
        subsequent_covs = self._A_s @ marginal_covariances[..., :-1, :, :]

        tf.debugging.assert_equal(tf.shape(subsequent_covs), tf.shape(self.state_transitions))
        return subsequent_covs

    def log_det_precision(self) -> tf.Tensor:
        r"""
        Calculate the log determinant of the precision matrix. This uses the precision as
        defined in the SpInGP paper (see class summary above).

        Precision is defined as:

        .. math:: K⁻¹ = (AQAᵀ)⁻¹

        so:

        .. math::
            log |K⁻¹| &=  log | Q⁻¹ | (since |A| = 1)\\
                      &= - log |P₀| - Σₜ log |Qₜ|\\
                      &= - 2 * (log |chol_P₀| + Σₜ log |chol_Qₜ|)

        :return: A tensor with shape ``batch_shape``.
        """
        # shape: [...]
        log_det = -(
            tf.reduce_sum(
                input_tensor=tf.math.log(tf.square(tf.linalg.diag_part(self._chol_P_0))), axis=-1,
            )
            + tf.reduce_sum(
                input_tensor=tf.math.log(tf.square(tf.linalg.diag_part(self._chol_Q_s))),
                axis=[-1, -2],
            )
        )

        tf.debugging.assert_equal(tf.shape(log_det), self.batch_shape)
        return log_det

    def create_non_trainable_copy(self) -> "StateSpaceModel":
        """
        Create a non-trainable version of :class:`~markovflow.gauss_markov.GaussMarkovDistribution`.

        This is to convert a trainable version of this class back to being non-trainable.

        :return: A Gauss-Markov distribution that is a copy of this one.
        """
        initial_mean = tf.stop_gradient(self.initial_mean)
        state_transitions = tf.stop_gradient(self.state_transitions)
        chol_initial_covariance = tf.stop_gradient(self.cholesky_initial_covariance)
        chol_process_covariances = tf.stop_gradient(self.cholesky_process_covariances)
        state_offsets = tf.stop_gradient(self.state_offsets)
        return StateSpaceModel(
            initial_mean,
            chol_initial_covariance,
            state_transitions,
            state_offsets,
            chol_process_covariances,
        )

    def create_trainable_copy(self) -> "StateSpaceModel":
        """
        Create a trainable version of this state space model.

        This is primarily for use with variational approaches where we want to optimise
        the parameters of a state space model that is initialised from a prior state space model.

        The initial mean and state transitions are the same.

        The initial and process covariances are 'flattened'. Since they are lower triangular, we
        only want to parametrise this part of the matrix. For this purpose we use the
        `params.triangular` constraint which is the `tfp.bijectors.FillTriangular` bijector that
        converts between a triangular matrix :math:`[dim, dim]` and a flattened vector of
        shape :math:`[dim (dim + 1) / 2]`.

        :return: A state space model that is a copy of this one and a dataclass containing the
                 variables that can be trained.
        """
        trainable_ssm = StateSpaceModel(
            initial_mean=Parameter(self._mu_0, name="initial_mean"),
            chol_initial_covariance=Parameter(
                self._chol_P_0, transform=triangular(), name="chol_initial_covariance"
            ),
            state_transitions=Parameter(self._A_s, name="state_transitions"),
            state_offsets=Parameter(self._b_s, name="state_offsets"),
            chol_process_covariances=Parameter(
                self._chol_Q_s, transform=triangular(), name="chol_process_covariances"
            ),
        )

        # check that the state space models are the same
        check_compatible(trainable_ssm, self)

        return trainable_ssm

    def _build_precision(self) -> SymmetricBlockTriDiagonal:
        """
        Compute the compact banded representation of the Precision matrix using state space model
        parameters.

        We construct matrix:
            K⁻¹ = A⁻ᵀQ⁻¹A⁻¹

        Using Q⁻¹ and A⁻¹ defined in equations (6) and (8) in the SpInGP paper (see class docstring)

        It can be shown that

        K⁻¹ = | P₀⁻¹ + A₁ᵀ Q₁⁻¹ A₁ | -A₁ᵀ Q₁⁻¹          | 0...
              | -Q₁⁻¹ A₁          | Q₁⁻¹ + A₂ᵀ Q₂⁻¹ A₂ |  -A₂ᵀ Q₂⁻¹         | 0...
              |   0               | -Q₂⁻¹ A₂          | Q₂⁻¹ + A₃ᵀ Q₃⁻¹ A₃ | -A₃ᵀ Q₃⁻¹| 0...
        ....

        :return: The precision as a `SymmetricBlockTriDiagonal` object
        """
        # [Q₁⁻¹A₁, Q₂⁻¹A₂, ....Qₙ⁻¹Aₙ]
        # [... num_transitions, state_dim, state_dim]
        inv_q_a = tf.linalg.cholesky_solve(self._chol_Q_s, self._A_s)
        # [A₁ᵀQ₁⁻¹A₁, A₂ᵀQ₂⁻¹A₂, .... AₙQₙ⁻¹Aₙᵀ]
        # [... num_transitions, state_dim, state_dim]
        aqa = tf.matmul(self._A_s, inv_q_a, transpose_a=True)
        # need to pad aqa to make it the same length
        # [... 1, state_dim, state_dim]
        padding_zeros = tf.zeros_like(self.cholesky_initial_covariance, dtype=default_float())[
            ..., None, :, :
        ]

        # Calculate [P₀⁻¹, Q₁⁻¹, Q₂⁻¹, ....]
        # First create the identities
        # [... num_transitions, state_dim, state_dim]
        identities = tf.eye(
            self.state_dim,
            dtype=default_float(),
            batch_shape=tf.concat([self.batch_shape, self.event_shape[:-1]], axis=0),
        )
        # now use cholesky solve with the identities to create [P₀⁻¹, Q₁⁻¹, Q₂⁻¹, ....]
        # [... num_transitions + 1, state_dim, state_dim]
        concatted_inv_q_s = tf.linalg.cholesky_solve(
            self.concatenated_cholesky_process_covariance, identities
        )

        # [P₀⁻¹ + A₁ᵀQ₁⁻¹A₁, Q₁⁻¹ + A₂ᵀQ₂⁻¹A₂, .... Qₙ₋₁⁻¹ + AₙQₙ⁻¹Aₙᵀ, Qₙ⁻¹]
        # [... num_transitions + 1, state_dim, state_dim]
        diag = concatted_inv_q_s + tf.concat([aqa, padding_zeros], axis=-3)

        shape = tf.shape(self.concatenated_cholesky_process_covariance)
        tf.debugging.assert_equal(tf.shape(diag), shape)

        return SymmetricBlockTriDiagonal(diag, -inv_q_a)

    def _log_pdf_factors(self, states: tf.Tensor) -> tf.Tensor:
        """
        Return the value of the log of the factors of the probability density function (PDF)
        evaluated at a state trajectory::

            [log p(x₀), log p(x₁|x₀), ..., log p(xₖ₊₁|xₖ)]

        ...with x₀ ~ 𝓝(μ₀, P₀) and xₖ₊₁|xₖ ~ 𝓝(Aₖ xₖ + bₖ, Qₖ)

        :states: The state trajectory has shape:
            sample_shape + self.batch_shape + self.event_shape
        :return: The log PDF of the factors with shape:
            sample_shape + self.batch_shape + [self.num_transitions + 1]
        """
        tf.debugging.assert_equal(tf.shape(states)[-2:], self.event_shape)

        # log p(x₀)
        initial_pdf = tfd.MultivariateNormalTriL(
            loc=self.initial_mean, scale_tril=self.cholesky_initial_covariance,
        ).log_prob(states[..., 0, :])

        # [A₁ x₀ + b₁, A₂ x₁ + b₂, ..., Aₖ₊₁ μₖ + bₖ₊₁]
        conditional_means = tf.matmul(self._A_s, states[..., :-1, :, None])[..., 0] + self._b_s

        # [log p(x₁|x₀), ..., etc]
        remaining_pdfs = tfd.MultivariateNormalTriL(
            loc=conditional_means, scale_tril=self.cholesky_process_covariances
        ).log_prob(states[..., 1:, :])
        return tf.concat([initial_pdf[..., None], remaining_pdfs], axis=-1)

    def log_pdf(self, states) -> tf.Tensor:
        """
        Return the value of the log of the probability density function (PDF)
        evaluated at states. That is:

        .. math:: log p(x) = log p(x₀) + Σₖ log p(xₖ₊₁|xₖ)  (for 0 ⩽ k < n)

        :param states: The state trajectory, with shape
            ``sample_shape + self.batch_shape + self.event_shape``.
        :return: The log PDF, with shape ``sample_shape + self.batch_shape``.
        """
        return tf.reduce_sum(self._log_pdf_factors(states), axis=-1)

    def kl_divergence(self, dist: GaussMarkovDistribution) -> tf.Tensor:
        r"""
        Return the KL divergence of the current Gauss-Markov distribution from the specified
        input `dist`. That is:

        .. math:: KL(dist₁ ∥ dist₂)

        To do so we first compute the marginal distributions from the Gauss-Markov form:

        .. math::
            dist₁ = 𝓝(μ₁, P⁻¹₁)\\
            dist₂ = 𝓝(μ₂, P⁻¹₂)

        ...where:

            * :math:`μᵢ` are the marginal means
            * :math:`Pᵢ` are the banded precisions

        The KL divergence is thus given by:

        .. math::
            KL(dist₁ ∥ dist₂) = ½(tr(P₂P₁⁻¹) + (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁) - N - log(|P₂|) + log(|P₁|))

        ...where :math:`N = (\verb |num_transitions| + 1) * \verb |state_dim|` (that is,
        the dimensionality of the Gaussian).

        :param dist: Another similarly parameterised Gauss-Markov distribution.
        :return: A tensor of the KL divergences, with shape ``self.batch_shape``.
        """
        check_compatible(self, dist)
        batch_shape = self.batch_shape

        marginal_covs_1 = self.marginal_covariances
        precision_2 = dist.precision

        # trace term, we use that for any trace tr(AᵀB) = Σᵢⱼ Aᵢⱼ Bᵢⱼ
        # and since the P₂ is symmetric block tri diagonal, we only need the block diagonal and
        # block sub diagonals from from P₁⁻¹
        # this is the sub diagonal of P₁⁻¹, [..., num_transitions, state_dim, state_dim]
        subsequent_covs_1 = self.subsequent_covariances(marginal_covs_1)
        # trace_sub_diag must be added twice as the matrix is symmetric, [...]
        trace = tf.reduce_sum(
            input_tensor=precision_2.block_diagonal * marginal_covs_1, axis=[-3, -2, -1]
        ) + 2.0 * tf.reduce_sum(
            input_tensor=precision_2.block_sub_diagonal * subsequent_covs_1, axis=[-3, -2, -1]
        )
        tf.debugging.assert_equal(tf.shape(trace), batch_shape)

        # (μ₂ - μ₁)ᵀP₂(μ₂ - μ₁)
        # [... num_transitions + 1, state_dim]
        mean_diff = dist.marginal_means - self.marginal_means
        # if P₂ = LLᵀ, calculate [Lᵀ(μ₂ - μ₁)] [... num_transitions + 1, state_dim]
        l_mean_diff = precision_2.cholesky.dense_mult(mean_diff, transpose_left=True)
        mahalanobis = tf.reduce_sum(input_tensor=l_mean_diff * l_mean_diff, axis=[-2, -1])  # [...]
        tf.debugging.assert_equal(tf.shape(mahalanobis), batch_shape)

        dim = (self.num_transitions + 1) * self.state_dim
        dim = tf.cast(dim, default_float())

        k_l = 0.5 * (
            trace + mahalanobis - dim - dist.log_det_precision() + self.log_det_precision()
        )

        tf.debugging.assert_equal(tf.shape(k_l), batch_shape)

        return k_l

    def normalizer(self):
        """
        Conputes the normalizer
        Page 36 of Thang Bui
        :return:
        """
        dim = (self.num_transitions + 1) * self.state_dim
        dim = tf.cast(dim, default_float())
        cst = dim * np.log(2.0 * np.pi)
        log_det = -self.log_det_precision()
        # if P₂ = LLᵀ, calculate [Lᵀ(μ₂ - μ₁)] [... num_transitions + 1, state_dim]
        l_mean = self.precision.cholesky.dense_mult(self.marginals[0], transpose_left=True)
        mahalanobis = tf.reduce_sum(input_tensor=l_mean * l_mean, axis=[-2, -1])  # [...]

        return 0.5 * (cst + log_det + mahalanobis)


@tf_scope_fn_decorator
def state_space_model_from_covariances(
    initial_mean: tf.Tensor,
    initial_covariance: tf.Tensor,
    state_transitions: tf.Tensor,
    state_offsets: tf.Tensor,
    process_covariances: tf.Tensor,
) -> StateSpaceModel:
    """
    Construct a state space model using the full covariance matrices for convenience.

    :param initial_mean: The initial mean, with shape ``batch_shape + [state_dim]``.
    :param initial_covariance: Initial covariance, with shape
        ``batch_shape + [state_dim, state_dim]``.
    :param state_transitions: State transition matrices, with shape
        ``batch_shape + [num_transitions, state_dim, state_dim]``.
    :param state_offsets: The process means :math:`bₖ`, with shape
        ``batch_shape + [num_transitions, state_dim]``.
    :param process_covariances: Noise covariance matrices, with shape
        ``batch_shape + [num_transitions, state_dim, state_dim]``.
    """

    def cholesky_or_zero(covariance: tf.Tensor) -> tf.Tensor:
        """
        This function takes a number of covariance matrices which have been stacked in the batch
        dimensions and, for each matrix if it non-zero computes the Cholesky of the matrix,
        otherwise leaves as-is (i.e. a matrix of zeros).

        :param covariance: tiled covariance matrices, shape
                batch_shape + [dim, dim]
        :return: tiled matrices each of which is either a Cholesky, or Zero matrix, shape
                batch_shape + [dim, dim]
        """
        zeros = tf.zeros_like(covariance)
        tf.debugging.assert_greater_equal(tf.size(covariance), 1)

        mask = tf.reduce_all(tf.math.equal(covariance, zeros), axis=(-2, -1))
        dim = covariance.shape[-1]
        mask_expanded = tf.stack([tf.stack([mask] * dim, axis=-1)] * dim, axis=-1)
        batch_identity = tf.broadcast_to(tf.eye(dim, dtype=default_float()), tf.shape(covariance))
        # As all arguments to tf.where are evaluated we need to make sure the Cholesky does not
        # fail, even if it is unused. This is all the following line does, is does not affect the
        # computation.
        fix = tf.where(mask_expanded, batch_identity, tf.zeros_like(batch_identity))
        return tf.where(mask_expanded, zeros, tf.linalg.cholesky(covariance + fix))

    return StateSpaceModel(
        initial_mean=initial_mean,
        chol_initial_covariance=cholesky_or_zero(initial_covariance),
        state_transitions=state_transitions,
        state_offsets=state_offsets,
        chol_process_covariances=cholesky_or_zero(process_covariances),
    )
