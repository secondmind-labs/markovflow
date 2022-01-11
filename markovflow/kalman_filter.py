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
"""Module containing a Kalman filter."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.base import TensorType

from markovflow.block_tri_diag import SymmetricBlockTriDiagonal
from markovflow.emission_model import EmissionModel
from markovflow.state_space_model import StateSpaceModel
from markovflow.utils import tf_scope_class_decorator


class BaseKalmanFilter(tf.Module, ABC):
    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with given observations.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R` is the observation covariance
        * :math:`Σ = I_N ⊗ R`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(self, state_space_model: StateSpaceModel, emission_model: EmissionModel,) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        """
        super().__init__(self.__class__.__name__)
        # verify observation shape
        self.prior_ssm = state_space_model
        self.emission = emission_model

    @property
    @abstractmethod
    def _r_inv(self):
        """ Precision of observation model """
        raise NotImplementedError

    @property
    @abstractmethod
    def observations(self):
        """ Observation vector """
        raise NotImplementedError

    @property
    def _k_inv_prior(self):
        """ Prior precision """
        return self.prior_ssm.precision

    @property
    def _k_inv_post(self):
        """ Posterior precision """

        # construct the likelihood precision: Gᵀ Σ⁻¹ G
        # HᵀR⁻¹H [state_dim, state_dim]
        h_t_r_h = tf.einsum(
            "...ji,...jk,...kl->...il",
            self.emission.emission_matrix,
            self._r_inv,
            self.emission.emission_matrix,
        )
        # The emission matrix is tiled across the time_points, so for a time invariant matrix
        # this is equivalent to Gᵀ Σ⁻¹ G = (I_N ⊗ HᵀR⁻¹H),
        likelihood_precision = SymmetricBlockTriDiagonal(h_t_r_h)
        _k_inv_prior = self.prior_ssm.precision
        # K⁻¹ + GᵀΣ⁻¹G
        return _k_inv_prior + likelihood_precision

    @property
    def _log_det_observation_precision(self):
        """ Sum of log determinant of the precisions of the observation model """
        num_data = self.prior_ssm.num_transitions + 1
        return tf.cast(num_data, default_float()) * tf.linalg.logdet(self._r_inv)

    def posterior_state_space_model(self) -> StateSpaceModel:
        r"""
        Return the posterior as a state space model.

        The marginal means and covariances are given by:

        .. math::
            &μ(Χ) = (K⁻¹ + GᵀΣ⁻¹G)⁻¹[GᵀΣ⁻¹y + K⁻¹μ]\\
            &P(X) = K⁻¹ + GᵀΣ⁻¹G

        ...where :math:`μ` is a block vector of the marginal means.

        We can derive the state transitions :math:`aₖ` and process noise covariances :math:`qₖ`
        from the block tridiagonal matrix (see
        :meth:`~markovflow.block_tri_diag.SymmetricBlockTriDiagonal.upper_diagonal_lower`).
        Lower case is used to attempt to distinguish the posterior and prior parameters.

        We then need to calculate :math:`μ₀` and :math:`bₖ` (this is what most of the code in
        this function does). This can be calculated from:

        .. math:: K⁻¹ₚₒₛₜμₚₒₛₜ = GᵀΣ⁻¹y + K⁻¹ₚᵣᵢₒᵣμₚᵣᵢₒᵣ

        Firstly, we use that for any :class:`~markovflow.state_space_model.StateSpaceModel`:

        .. math:: K⁻¹μ = A⁻ᵀ Q⁻¹ m

        ...where :math:`m = [μ₀, b₁,... bₙ]` and::

            A⁻¹ =  [ I             ]      Q⁻¹ =  [ P₀⁻¹          ]
                   [-A₁, I         ]            [    Q₁⁻¹       ]
                   [    -A₂, I     ]            [       ᨞      ]
                   [         ᨞  ᨞  ]            [         ᨞    ]
                   [         -Aₙ, I]            [           Qₙ⁻¹]

        So:

        .. math:: mₚₒₛₜ = Qₚₒₛₜ Aₚₒₛₜᵀ [GᵀΣ⁻¹y + Kₚᵣᵢₒᵣ⁻¹mₚᵣᵢₒᵣ]

        :return: The posterior as a state space model.
        """
        a_inv_post, chol_q_inv_post = self._k_inv_post.upper_diagonal_lower()
        assert a_inv_post.block_sub_diagonal is not None

        # (GᵀΣ⁻¹)y [..., num_transitions + 1, state_dim]
        obs_proj = self._back_project_y_to_state(self.observations)

        # Kₚᵣᵢₒᵣ⁻¹μₚᵣᵢₒᵣ (prior parameters) [..., num_transitions + 1, state_dim]
        k_inv_mu_prior = self._k_inv_prior.dense_mult(self.prior_ssm.marginal_means)

        # mₚₒₛₜ = Qₚₒₛₜ Aₚₒₛₜᵀ [GᵀΣ⁻¹y + Kₚᵣᵢₒᵣ⁻¹mₚᵣᵢₒᵣ] [..., num_transitions + 1, state_dim]
        m_post = chol_q_inv_post.solve(
            chol_q_inv_post.solve(a_inv_post.solve(obs_proj + k_inv_mu_prior, transpose_left=True)),
            transpose_left=True,
        )

        # [..., num_transitions + 1, state_dim, state_dim]
        batch_shape = tf.concat(
            [self.prior_ssm.batch_shape, tf.TensorShape([self.prior_ssm.num_transitions + 1])],
            axis=0,
        )
        identities = tf.eye(self.prior_ssm.state_dim, dtype=m_post.dtype, batch_shape=batch_shape)

        # cholesky of [P₀, Q₁, Q₂, ....] [..., num_transitions + 1, state_dim, state_dim]
        concatted_qs = tf.linalg.cholesky(
            tf.linalg.cholesky_solve(chol_q_inv_post.block_diagonal, identities)
        )

        return StateSpaceModel(
            initial_mean=m_post[..., 0, :],
            chol_initial_covariance=concatted_qs[..., 0, :, :],
            state_transitions=-a_inv_post.block_sub_diagonal,
            state_offsets=m_post[..., 1:, :],
            chol_process_covariances=concatted_qs[..., 1:, :, :],
        )

    def log_likelihood(self) -> tf.Tensor:
        r"""
        Construct a TensorTlow function to compute the likelihood.

        We set :math:`y = obs - Hμ` (where :math:`μ` is the vector of marginal state means):

        .. math::
            log p(obs|params) = &- ᴺ⁄₂log(2π) - ½(log |K⁻¹ + GᵀΣ⁻¹G| - log |K⁻¹| - log |Σ⁻¹|)\\
                                &- ½ yᵀ(Σ⁻¹ - Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y

        ...where :math:`N` is the dimensionality of the precision object, that is
        ``state_dim * (num_transitions + 1)``.

        We break up the log likelihood as: cst + term1 + term2 + term3. That is, as:

            * cst: :math:`- ᴺ⁄₂log(2π)`
            * term 1: :math:`- ½ yᵀΣ⁻¹y`
            * term 2:

              .. math::
                 ½ yᵀΣ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y = ½ yᵀΣ⁻¹G(LLᵀ)⁻¹GᵀΣ⁻¹)y = ½|L⁻¹(GᵀΣ⁻¹)y|²

            * term 3:

              .. math::
                 - ½(log |K⁻¹ + GᵀΣ⁻¹G| - log |K⁻¹| - log |Σ⁻¹|) = ½log |K⁻¹| - log |L| + ½log |Σ⁻¹|

        Note that there are a couple of mistakes in the SpinGP paper for this formula (18):

            * They have :math:`- ½(... + log |Σ⁻¹|)`. It should be :math:`- ½(... - log |Σ⁻¹|)`
            * They have :math:`- ½ yᵀ(... Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹)y`. It should
              be :math:`- ½ yᵀ(... Σ⁻¹G(K⁻¹ + GᵀΣ⁻¹G)⁻¹GᵀΣ⁻¹)y`

        :return: The likelihood as a scalar tensor (we sum over the `batch_shape`).
        """
        # K⁻¹ + GᵀΣ⁻¹G = LLᵀ.
        l_post = self._k_inv_post.cholesky
        num_data = self.prior_ssm.num_transitions + 1
        # Hμ [..., num_transitions + 1, output_dim]
        marginal = self.emission.project_state_to_f(self.prior_ssm.marginal_means)

        # y = obs - Hμ [..., num_transitions + 1, output_dim]
        disp = self.observations - marginal

        # cst is the constant term for a gaussian log likelihood
        cst = (
            -0.5 * np.log(2 * np.pi) * tf.cast(self.emission.output_dim * num_data, default_float())
        )

        # term1 is: - ½ yᵀΣ⁻¹y shape [...]
        term1 = -0.5 * tf.reduce_sum(
            input_tensor=tf.einsum("...op,...p,...o->...o", self._r_inv, disp, disp), axis=[-1, -2]
        )

        # term 2 is: ½|L⁻¹(GᵀΣ⁻¹)y|²
        # (GᵀΣ⁻¹)y [..., num_transitions + 1, state_dim]
        obs_proj = self._back_project_y_to_state(disp)

        # ½|L⁻¹(GᵀΣ⁻¹)y|² [...]
        term2 = 0.5 * tf.reduce_sum(
            input_tensor=tf.square(l_post.solve(obs_proj, transpose_left=False)), axis=[-1, -2]
        )

        ## term 3 is: ½log |K⁻¹| - log |L| + ½ log |Σ⁻¹|
        # where log |Σ⁻¹| = num_data * log|R⁻¹|
        term3 = (
            0.5 * self.prior_ssm.log_det_precision()
            - l_post.abs_log_det()
            + 0.5 * self._log_det_observation_precision
        )

        return tf.reduce_sum(cst + term1 + term2 + term3)

    def _back_project_y_to_state(self, observations: tf.Tensor) -> tf.Tensor:
        """
        Back project from the observation space to the state_space, i.e. calculate (GᵀΣ⁻¹)y.

        :param observations: a tensor y of shape
                    batch_shape + [num_data, output_dim]
        :return: a tensor (GᵀΣ⁻¹)y of shape
                    batch_shape + [num_data, state_dim]
        """
        # GᵀΣ⁻¹, batch_shape + [num_data, output_dim, state_dim]
        back_projection = tf.einsum(
            "...ij,...ki->...kj", self.emission.emission_matrix, self._r_inv
        )
        # (GᵀΣ⁻¹) y
        return tf.einsum("...ij,...i->...j", back_projection, observations)

    def forward_filter_scan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the forward filter. i.e. return the mean and variance for
            p(xₜ | y₁ ... yₜ₋₁), this is called 'pred' and
            p(xₜ | y₁ ... yₜ) this is called 'filter'
        also return log likelihoods i.e. log p(yₜ | y₁ ... yₜ₋₁)

        :param observations: batch_shape + [num_timesteps, output_dim]
        :return: log_liks: batch_shape + [num_timesteps]
                 filter_mus: batch_shape + [num_timesteps, state_dim]
                 filter_covs: batch_shape + [num_timesteps, state_dim, state_dim]
                 pred_mus: batch_shape + [num_timesteps, state_dim]
                 pred_covs: batch_shape + [num_timesteps, state_dim, state_dim]
        """
        # [..., state_dim, state_dim]
        P_0 = tf.matmul(self.prior_ssm._chol_P_0, self.prior_ssm._chol_P_0, transpose_b=True)
        # [..., state_dim]
        mu_0 = self.prior_ssm._mu_0
        # [..., num_transitions, state_dim, state_dim]
        Q_s = tf.matmul(self.prior_ssm._chol_Q_s, self.prior_ssm._chol_Q_s, transpose_b=True)
        A_s = self.prior_ssm.state_transitions
        b_s = self.prior_ssm.state_offsets
        H_s = self.emission.emission_matrix
        R = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(self._r_inv),
            tf.eye(self.emission.output_dim, dtype=default_float())
        )
        y_s = self.observations
        indices = tf.range(self.prior_ssm.num_transitions)

        def body(carry, counter):
            filter_mean, filter_cov, pred_mean, pred_cov = carry

            A_k = A_s[..., counter, :, :]  # [... state_dim, state_dim]
            b_k = b_s[..., counter, :]  # [...  1, state_dim]
            Q_k = Q_s[..., counter, :, :]  # [... state_dim, state_dim]
            H_k = H_s[..., counter, :, :]
            y_k = y_s[..., counter, :]

            # correct
            S = H_k @ tf.matmul(pred_cov, H_k, transpose_b=True) + R
            chol = tf.linalg.cholesky(S)
            Kt = tf.linalg.cholesky_solve(chol, H_k @ pred_cov)
            filter_mean = pred_mean + tf.linalg.matvec(Kt, y_k - tf.linalg.matvec(H_k, pred_mean), transpose_a=True)
            filter_cov = pred_cov - tf.matmul(Kt, S, transpose_a=True) @ Kt

            # propagate
            pred_mean = tf.linalg.matvec(A_k, filter_mean) + b_k
            pred_cov = A_k @ tf.matmul(filter_cov, A_k, transpose_b=True) + Q_k

            return filter_mean, filter_cov, pred_mean, pred_cov

        return tf.scan(body, indices, (mu_0, P_0, mu_0, P_0))
        #
        # def body(carry, counter):
        #     filter_mean, filter_cov = carry
        #     A_k = A_s[..., counter, :, :]  # [... state_dim, state_dim]
        #     b_k = b_s[..., counter, :]  # [...  1, state_dim]
        #     Q_k = Q_s[..., counter, :, :]  # [... state_dim, state_dim]
        #     H_k = H_s[..., counter, :, :]
        #     y_k = y_s[..., counter, :]
        #
        #     # correct
        #     S = H_k @ tf.matmul(filter_cov, H_k, transpose_b=True) + R
        #     chol = tf.linalg.cholesky(S)
        #     Kt = tf.linalg.cholesky_solve(chol, H_k @ filter_cov)
        #     filter_mean = filter_mean + tf.linalg.matvec(Kt, y_k - tf.linalg.matvec(H_k, filter_mean), transpose_a=True)
        #     filter_cov = filter_cov - tf.matmul(Kt, S, transpose_a=True) @ Kt
        #
        #     # propagate
        #     filter_mean = tf.linalg.matvec(A_k, filter_mean) + b_k
        #     filter_cov = A_k @ tf.matmul(filter_cov, A_k, transpose_b=True) + Q_k
        #
        #     return filter_mean, filter_cov
        #
        # return tf.scan(body, indices, (mu_0, P_0))

    def forward_filter(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        """
        # [..., state_dim, state_dim]
        P_0 = tf.matmul(self.prior_ssm._chol_P_0, self.prior_ssm._chol_P_0, transpose_b=True)
        # [..., state_dim]
        mu_0 = self.prior_ssm._mu_0
        # [..., num_transitions, state_dim, state_dim]
        Q_s = tf.matmul(self.prior_ssm._chol_Q_s, self.prior_ssm._chol_Q_s, transpose_b=True)
        A_s = self.prior_ssm.state_transitions
        b_s = self.prior_ssm.state_offsets
        H_s = self.emission.emission_matrix

        y_s = self.observations
        R = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(self._r_inv),
            tf.eye(self.emission.output_dim, dtype=default_float())
        )
        y_s = self.observations


        # first correction step
        H_0 = H_s[..., 0, :, :]
        y_0 = y_s[..., 0, :]
        S = H_0 @ tf.matmul(P_0, H_0, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)
        Kt = tf.linalg.cholesky_solve(chol, H_0 @ P_0)
        mu_1 = mu_0 + tf.linalg.matvec(Kt, y_0 - tf.linalg.matvec(H_0, mu_0), transpose_a=True)
        P_1 = P_0 - tf.matmul(Kt, S, transpose_a=True) @ Kt

        def step(filter_mus, filter_covs, pred_mus, pred_covs, counter)\
                -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """
            Step the mean and the variance forward one timestep.
            These are the standard equations for linear transformations of a Gaussian

            μₖ₊₁ = Aₖμₖ + bₖ
            Sₖ₊₁ = AₖSₖAₖᵀ + Qₖ
            """
            A_k = A_s[..., counter, :, :]  # [... state_dim, state_dim]
            b_k = b_s[..., counter, :]  # [...  1, state_dim]
            Q_k = Q_s[..., counter, :, :]  # [... state_dim, state_dim]
            H_k = H_s[..., counter + 1, :, :]
            y_k = y_s[..., counter + 1, :]
            filter_cov = filter_covs[..., -1, :, :]
            filter_mean = filter_mus[..., -1, :]
            pred_cov = pred_covs[..., -1, :, :]
            pred_mean = pred_mus[..., -1, :]

            # # Aₖμₖ + bₖ[... 1, state_dim]
            # mu_t = tf.matmul(mus[..., -1:, :], A_k, transpose_b=True) + b_k
            #
            # # [... state_dim, state_dim]
            # # AₖSₖAₖᵀ + Qₖ
            # cov_t = tf.matmul((A_k @ covs[..., -1, :, :]), A_k, transpose_b=True) + Q_k

            # propagate
            pred_mean = tf.linalg.matvec(A_k, filter_mean) + b_k
            pred_cov = A_k @ tf.matmul(filter_cov, A_k, transpose_b=True) + Q_k

            # correct
            S = H_k @ tf.matmul(pred_cov, H_k, transpose_b=True) + R
            chol = tf.linalg.cholesky(S)
            Kt = tf.linalg.cholesky_solve(chol, H_k @ pred_cov)
            filter_mean = pred_mean + tf.linalg.matvec(Kt, y_k - tf.linalg.matvec(H_k, pred_mean), transpose_a=True)
            filter_cov = pred_cov - tf.matmul(Kt, S, transpose_a=True) @ Kt


            # stick the new mean and covariance to their accumulators and increment the counter
            return (tf.concat([filter_mus, filter_mean[..., None, :]], axis=-2),
                    tf.concat([filter_covs, filter_cov[..., None, :, :]], axis=-3),
                    tf.concat([pred_mus, pred_mean[..., None, :]], axis=-2),
                    tf.concat([pred_covs, pred_cov[..., None, :, :]], axis=-3),
                    counter + 1)

        # set up the loop variables and shape invariants
        # [... 1, state_dim] and [... 1, state_dim, state_dim]



        loop_vars = (mu_1[..., None, :], P_1[..., None, :, :],
                     mu_0[..., None, :], P_0[..., None, :, :],
                     tf.constant(0, tf.int32))

        batch_shape = self.prior_ssm.batch_shape
        state_dim = self.prior_ssm.state_dim
        num_transitions = self.prior_ssm.num_transitions
        shape_invars = (tf.TensorShape(batch_shape + (None, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim, state_dim)),
                        tf.TensorShape([]))

        filter_mus, filter_covs, pred_mus, pred_covs, _ = tf.while_loop(cond=lambda _, __, ___, ____, counter: counter < num_transitions,
                                     body=step,
                                     loop_vars=loop_vars,
                                     shape_invariants=shape_invars)

        # first ensure the shape is compatible
        mus_shape = batch_shape + (num_transitions + 1, state_dim)
        tf.ensure_shape(filter_mus, mus_shape)
        tf.ensure_shape(filter_covs, mus_shape + (state_dim,))

        return filter_mus, filter_covs, pred_mus, pred_covs


    def backward_filter(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        """
        # [..., num_transitions, state_dim, state_dim]
        Q_s = tf.matmul(self.prior_ssm._chol_Q_s, self.prior_ssm._chol_Q_s, transpose_b=True)
        A_s = self.prior_ssm.state_transitions
        b_s = self.prior_ssm.state_offsets
        H_s = self.emission.emission_matrix

        y_s = self.observations
        R = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(self._r_inv),
            tf.eye(self.emission.output_dim, dtype=default_float())
        )
        y_s = self.observations


        # first correction
        P_0 = self.prior_ssm.marginal_covariances[..., -1, :, :]
        mu_0 = self.prior_ssm.marginal_means[..., -1, :]
        H_0 = H_s[..., -1, :, :]
        y_0 = y_s[..., -1, :]
        S = H_0 @ tf.matmul(P_0, H_0, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)
        Kt = tf.linalg.cholesky_solve(chol, H_0 @ P_0)
        mu_1 = mu_0 + tf.linalg.matvec(Kt, y_0 - tf.linalg.matvec(H_0, mu_0), transpose_a=True)
        P_1 = P_0 - tf.matmul(Kt, S, transpose_a=True) @ Kt

        def step(filter_mus, filter_covs, pred_mus, pred_covs, counter)\
                -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """
            Step the mean and the variance forward one timestep.
            These are the standard equations for linear transformations of a Gaussian

            μₖ₊₁ = Aₖμₖ + bₖ
            Sₖ₊₁ = AₖSₖAₖᵀ + Qₖ
            """
            A_k = A_s[..., counter, :, :]  # [... state_dim, state_dim]
            b_k = b_s[..., counter, :]  # [...  1, state_dim]
            Q_k = Q_s[..., counter, :, :]  # [... state_dim, state_dim]
            H_k = H_s[..., counter, :, :]
            y_k = y_s[..., counter, :]
            filter_cov = filter_covs[..., 0, :, :]
            filter_mean = filter_mus[..., 0, :]
            # # Aₖμₖ + bₖ[... 1, state_dim]
            # mu_t = tf.matmul(mus[..., -1:, :], A_k, transpose_b=True) + b_k
            #
            # # [... state_dim, state_dim]
            # # AₖSₖAₖᵀ + Qₖ
            # cov_t = tf.matmul((A_k @ covs[..., -1, :, :]), A_k, transpose_b=True) + Q_k

            # propagate
            # Q_k = A_k @ tf.matmul(Q_k, A_k, transpose_b=True)

            # pred_mean = tf.linalg.solve(A_k, (filter_mean - b_k)[..., None])[..., 0]
            # # pred_cov = A_k @ tf.matmul(filter_cov, A_k, transpose_b=True) + Q_k
            # pred_cov = tf.linalg.solve(A_k,
            #                            tf.linalg.matrix_transpose(tf.linalg.solve(A_k, filter_cov + Q_k)))
            #            pred_cov = A_k @ tf.matmul(filter_cov + Q_k, A_k, transpose_b=True) + Q_k
            iA_k = tf.linalg.inv(A_k)
            pred_mean = tf.linalg.matvec(iA_k, filter_mean - b_k)
            pred_cov = iA_k @ tf.matmul(filter_cov + Q_k, iA_k, transpose_b=True)
            # correct
            S = H_k @ tf.matmul(pred_cov, H_k, transpose_b=True) + R
            chol = tf.linalg.cholesky(S)
            Kt = tf.linalg.cholesky_solve(chol, H_k @ pred_cov)
            filter_mean = pred_mean + tf.linalg.matvec(Kt, y_k - tf.linalg.matvec(H_k, pred_mean), transpose_a=True)
            filter_cov = pred_cov - tf.matmul(Kt, S, transpose_a=True) @ Kt



            # stick the new mean and covariance to their accumulators and increment the counter
            return (tf.concat([filter_mean[..., None, :], filter_mus], axis=-2),
                    tf.concat([filter_cov[..., None, :, :], filter_covs], axis=-3),
                    tf.concat([pred_mean[..., None, :], pred_mus], axis=-2),
                    tf.concat([pred_cov[..., None, :, :], pred_covs], axis=-3),
                    counter - 1)

        batch_shape = self.prior_ssm.batch_shape
        state_dim = self.prior_ssm.state_dim
        num_transitions = self.prior_ssm.num_transitions



        loop_vars = (mu_1[..., None, :], P_1[..., None, :, :],
                     mu_0[..., None, :], P_0[..., None, :, :],
                     tf.constant(num_transitions - 1, tf.int32))

        shape_invars = (tf.TensorShape(batch_shape + (None, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim)),
                        tf.TensorShape(batch_shape + (None, state_dim, state_dim)),
                        tf.TensorShape([]))

        filter_mus, filter_covs, pred_mus, pred_covs, _ = tf.while_loop(
                                     cond=lambda _, __, ___, ____, counter: counter > -1,
                                     body=step,
                                     loop_vars=loop_vars,
                                     shape_invariants=shape_invars)

        # first ensure the shape is compatible
        mus_shape = batch_shape + (num_transitions + 1, state_dim)
        tf.ensure_shape(filter_mus, mus_shape)
        tf.ensure_shape(filter_covs, mus_shape + (state_dim,))

        return filter_mus, filter_covs, pred_mus, pred_covs



@tf_scope_class_decorator
class KalmanFilter(BaseKalmanFilter):
    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with given observations.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R` is the observation covariance
        * :math:`Σ = I_N ⊗ R`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(
        self,
        state_space_model: StateSpaceModel,
        emission_model: EmissionModel,
        observations: tf.Tensor,
        chol_obs_covariance: TensorType,
    ) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        :param observations: Data with shape ``[num_transitions + 1, output_dim]``.
        :param chol_obs_covariance: A :data:`~markovflow.base.TensorType` with shape
            ``[output_dim, output_dim]`` for the Cholesky factor of the covariance to be
            applied to :math:`f` from `emission_model`.
        """
        super().__init__(state_space_model, emission_model)

        assert isinstance(observations, tf.Tensor)

        # verify observation covariance shape
        shape = tf.convert_to_tensor([emission_model.output_dim, emission_model.output_dim])
        message = """The shape of the observation covaraiance matrix and the emission
                     matrix are not compatible"""
        tf.debugging.assert_equal(tf.shape(chol_obs_covariance), shape, message=message)

        # verify observation shape
        message = """The shape of the observations and the state-space-model parameters
                     are not compatible"""
        shape = tf.concat(
            [
                state_space_model.batch_shape,
                [state_space_model.num_transitions + 1, emission_model.output_dim],
            ],
            axis=0,
        )
        tf.debugging.assert_equal(tf.shape(observations), shape, message=message)

        self._chol_obs_covariance = chol_obs_covariance  # To collect tf.Module trainables
        self._observations = observations  # batch_shape + [num_transitions + 1, output_dim]

    @property
    def _r_inv(self):
        """ Precision of the observation model """
        # [output_dim, output_dim]
        return tf.linalg.cholesky_solve(
            self._chol_obs_covariance,
            tf.eye(self.emission.output_dim, dtype=self._chol_obs_covariance.dtype),
        )

    @property
    def observations(self):
        """ Observation vector """
        return self._observations


class GaussianSites(tf.Module, ABC):
    """
    This class is a wrapper around the parameters specifying multiple independent
    Gaussian distributions.
    """

    @property
    def means(self):
        """
        Return the means of the Gaussians.
        """
        raise NotImplementedError

    @property
    def precisions(self):
        """
        Return the precisions of the Gaussians.
        """
        raise NotImplementedError

    @property
    def log_det_precisions(self):
        """ Return the sum of the log determinant of the observation precisions."""
        raise NotImplementedError


class UnivariateGaussianSitesNat(GaussianSites):
    """
    This class is a wrapper around parameters of univariate Gaussian distributions
    in the natural form. That is:

    .. math:: p(f) = exp(𝞰ᵀφ(f) - A(𝞰))

    ...where :math:`𝞰=[η₁,η₂]` and :math:`𝛗(f)=[f,f²]`.

    The mean :math:`μ` and variance :math:`σ²` parameterization is such that:

    .. math:: μ = -½η₁/η₂, σ²=-½η₂⁻¹
    """

    def __init__(self, nat1, nat2, log_norm=None):
        """
        :param nat1: first natural parameter [N, D]
        :param nat2: second natural parameter [N, D, D]
        :param log_norm: normalizer parameter [N, D]
        """
        super().__init__()
        shape_constraints = [
            (nat1, ["N", 1]),
            (nat2, ["N", 1, 1]),
        ]
        if log_norm is not None:
            shape_constraints += [(log_norm, ["N", 1])]
        tf.debugging.assert_shapes(shape_constraints)

        self.num_data, self.output_dim = nat1.shape
        self.nat1 = nat1
        self.nat2 = nat2
        self.log_norm = log_norm

    @property
    def means(self):
        """
        Return the means of the Gaussians.
        """
        return -0.5 * self.nat1 / self.nat2[..., 0]

    @property
    def precisions(self):
        """
        Return the precisions of the Gaussians.
        """
        return -2 * self.nat2

    @property
    def log_det_precisions(self):
        """ Return the sum of the log determinant of the observation precisions. """
        return tf.math.log(-2 * self.nat2)


@tf_scope_class_decorator
class KalmanFilterWithSites(BaseKalmanFilter):

    r"""
    Performs a Kalman filter on a :class:`~markovflow.state_space_model.StateSpaceModel` and
    :class:`~markovflow.emission_model.EmissionModel`, with Gaussian sites,
    that is time dependent Gaussian Likelihood terms.

    The key reference is::

        @inproceedings{grigorievskiy2017parallelizable,
            title={Parallelizable sparse inverse formulation Gaussian processes (SpInGP)},
            author={Grigorievskiy, Alexander and Lawrence, Neil and S{\"a}rkk{\"a}, Simo},
            booktitle={Int'l Workshop on Machine Learning for Signal Processing (MLSP)},
            pages={1--6},
            year={2017},
            organization={IEEE}
        }

    The following notation from the above paper is used:

        * :math:`G = I_N ⊗ H`, where :math:`⊗` is the Kronecker product
        * :math:`R = [R₁, R₂, ... Rₙ]` is the observation covariance
        * :math:`Σ = blockdiag[R]`
        * :math:`K⁻¹ = A⁻ᵀQ⁻¹A⁻¹` is the precision, where :math:`A⁻ᵀ =  [Aᵀ]⁻¹ = [A⁻¹]ᵀ`
        * :math:`L` is the Cholesky of :math:`K⁻¹ + GᵀΣ⁻¹G`. That is, :math:`LLᵀ = K⁻¹ + GᵀΣ⁻¹G`
        * :math:`y` is the observation matrix
    """

    def __init__(
        self,
        state_space_model: StateSpaceModel,
        emission_model: EmissionModel,
        sites: GaussianSites,
    ) -> None:
        """
        :param state_space_model: Parametrises the latent chain.
        :param emission_model: Maps the latent chain to the observations.
        :param sites: Gaussian sites parameterizing the Gaussian likelihoods.
        """
        # verify site shape
        message = """The shape of the site matrices and the emission
                     matrix are not compatible"""
        tf.debugging.assert_equal(sites.output_dim, emission_model.output_dim, message=message)
        self.sites = sites

        super().__init__(state_space_model, emission_model)

    @property
    def _r_inv(self):
        """ Precisions of the observation model """
        return self.sites.precisions

    @property
    def _log_det_observation_precision(self):
        """ Sum of log determinant of the precisions of the observation model """
        return tf.reduce_sum(tf.linalg.logdet(self._r_inv), axis=-1)

    @property
    def observations(self):
        """ Observation vector """
        return self.sites.means
