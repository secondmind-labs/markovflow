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
"""Module for evaluating conditional distributions."""
from typing import Optional, Tuple

import tensorflow as tf
from gpflow import default_float

from markovflow.base import APPROX_INF
from markovflow.gauss_markov import GaussMarkovDistribution
from markovflow.kernels import SDEKernel
from markovflow.utils import tf_scope_fn_decorator


@tf_scope_fn_decorator
def conditional_predict(
    new_time_points: tf.Tensor,
    training_time_points: tf.Tensor,
    kernel: SDEKernel,
    training_pairwise_means: tf.Tensor,
    training_pairwise_covariances: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Given :math:`∀ xₜ ∈` `new_time_points`, compute the means and
    covariances of the marginal densities:

    .. math:: p(xₜ) = ∫ d[x₋, x₊] p(xₜ|x₋, x₊) q(x₋, x₊; mₜ, Sₜ) = 𝓝(Pₜ mₜ, Tₜ + Pₜ Sₜ Pₜᵀ)

    Or, if :math:`Sₜ` is not given, compute the conditional density:

    .. math:: p(xₜ|[x₋, x₊] = mₜ) = 𝓝(Pₜ @ [x₋, x₊], Tₜ)

    .. note:: `new_time_points` and `training_time_points` must be sorted.

    Where:

      - :math:`p` is the density over state trajectories specified by the kernel
      - :math:`∀ xₜ ∈` `new_time_points`:

        .. math::
            x₊ = arg minₓ \{|x-xₜ|, x ∈ \verb|training_time_point and |x>xₜ\}\\
            x₋ = arg minₓ \{|x-xₜ|, x ∈ \verb|training_time_point and |x⩽xₜ\}

    Details of the computation of :math:`Pₜ` and :math:`Tₜ` are found
    in :func:`conditional_statistics`.

    :param new_time_points: Sorted time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
    :param training_time_points: Sorted time points to condition on, with shape
            ``batch_shape + [num_training_time_points,]``.
    :param kernel: A kernel.
    :param training_pairwise_means: Pairs of states to condition on, with shape
            ``batch_shape + [num_training_time_points, 2 * state_dim]``.
    :param training_pairwise_covariances: Covariances of the pairs of states to condition on, with
            shape ``batch_shape + [num_training_time_points, 2 * state_dim, 2 * state_dim]``.
    :return: Predicted mean and covariance for the new time points, with respective shapes
            ``batch_shape + [num_new_time_points, state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``.
    """
    P, T, indices = _conditional_statistics(new_time_points, training_time_points, kernel)
    # projection and marginalization (if Sₜ given)
    batch_dims = len(new_time_points.shape[:-1])
    pairwise_means = tf.gather(training_pairwise_means, indices, batch_dims=batch_dims)

    if training_pairwise_covariances is None:
        pairwise_covs = None
    else:
        pairwise_covs = tf.gather(training_pairwise_covariances, indices, batch_dims=batch_dims)

    return base_conditional_predict(P, T, pairwise_means, pairwise_state_covariances=pairwise_covs)


@tf_scope_fn_decorator
def conditional_statistics(
    new_time_points: tf.Tensor, training_time_points: tf.Tensor, kernel: SDEKernel
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Given :math:`∀ xₜ ∈` `new_time_points`, compute the statistics :math:`Pₜ` and :math:`Tₜ`
    of the conditional densities:

    .. math:: p(xₜ|x₋, x₊) = 𝓝(Pₜ @ [x₋, x₊], Tₜ)

    ...where:

        - :math:`p` is the density over state trajectories specified by the kernel
        - :math:`∀ xₜ ∈` `new_time_points`:

          .. math::
                x₊ = arg minₓ \{ |x-xₜ|, x ∈ \verb|training_time_point and |x>xₜ \}\\
                x₋ = arg minₓ \{ |x-xₜ|, x ∈ \verb|training_time_point and |x⩽xₜ \}

    .. note:: `new_time_points` and `training_time_points` must be sorted.

    :param new_time_points: Sorted time points to generate observations for, with shape
            ``batch_shape + [num_new_time_points,]``.
    :param training_time_points: Sorted time points to condition on, with shape
            ``batch_shape + [num_training_time_points,]``.
    :param kernel: A kernel.
    :return: Parameters for the conditional mean and covariance, with respective shapes
            ``batch_shape + [num_new_time_points, state_dim, 2 * state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``.
    """
    #  remove the `indices` output from `_conditional_statistics`
    P, T, _ = _conditional_statistics(new_time_points, training_time_points, kernel)
    return P, T


@tf_scope_fn_decorator
def _conditional_statistics_from_transitions(
    state_transitions_to_t: tf.Tensor,
    process_covariances_to_t: tf.Tensor,
    state_transitions_from_t: tf.Tensor,
    process_covariances_from_t: tf.Tensor,
    return_precision: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Implementation details:
        Given consecutive time differences Δ₁ = t - t₋ and Δ₂ = t₊ - t
        of ordered triplets t₋ < t < t₊, we denote their values as
        x₋, xₜ, x₊ and their conditional distributions as
        p(x₊ | xₜ) = 𝓝(x₊; Aₜ₊xₜ, Qₜ₊)    where     [Aₜ₊ == A_tp, Qₜ₊ == Q_tp]
        p(xₜ | x₋) = 𝓝(xₜ; A₋ₜx₋, Q₋ₜ)    where     [A₋ₜ == A_mt, Q₋ₜ == Q_mt]

        This computes Dₜ, Eₜ, Tₜ (or Tₜ⁻¹) such that
        p(xₜ | x₋, x₊) = 𝓝(xₜ; Dₜ @ x₋ + Eₜ @ x₊, Tₜ)

        p(x₊|xₜ, x₋) = p(x₊|xₜ) = 𝓝(Aₜ₊xₜ, Qₜₚ)
        p(xₜ|x₋) = 𝓝(A₋ₜx₋, Q₋ₜ)
        p(x₊| x₋) = 𝓝(A₋₊x₋, Q₋₊ = Qₜ₊ + Aₜ₊Q₋ₜAₜ₊ᵀ)

        p([xₜ, x₊]| x₋) = p(x₊| xₜ)p(xₜ|x₋)
                      = 𝓝([A₋ₜx₋, A₋₊x₋]ᵀ, [[ Q₋ₜ, Q₋ₜAₜ₊ᵀ],
                                             [ Aₜ₊Q₋ₜ, Q₋₊ ]]

        Given this joint distribution we can obtain the mean and covariance of the
        conditional distribution of
        p(xₜ|[x₋, x₊]) = 𝓝(xₜ; A₋ₜx₋ + Q₋ₜAₜ₊ᵀQ₋₊⁻¹(x₊ - A₋₊x₋), Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ)
                       = 𝓝(xₜ; (A₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹A₋₊)x₋ + Q₋ₜAₜ₊ᵀQ₋₊⁻¹x₊,
                                (Q₋ₜ⁻¹ + Aₜ₊ᵀQₜ₊⁻¹Aₜ₊)⁻¹)

    :param state_transitions_to_t: the state transitions from t₋ to t - A₋ₜ
            ``batch_shape + [num_time_points, state_dim, state_dim]``
    :param process_covariances_to_t: the process covariances from t₋ to t - Q₋ₜ
            ``batch_shape + [num_time_points, state_dim, state_dim]``
    :param state_transitions_from_t: the state transitions from t to t₊ - A₋ₜ
            ``batch_shape + [num_time_points, state_dim, state_dim]``
    :param process_covariances_from_t: the process covariances from t to t₊ - Qₜ₊
            ``batch_shape + [num_time_points, state_dim, state_dim]``
    :param return_precision: bool, defaults to False.
            if True (resp. False), conditional precision (resp. covariance) is returned
    :return: parameters for the conditional mean and covariance
            ``batch_shape + [num_time_points, state_dim, state_dim]``
            ``batch_shape + [num_time_points, state_dim, state_dim]``
            ``batch_shape + [num_time_points, state_dim, state_dim]``
    """
    A_tp_Q_mt = tf.matmul(state_transitions_from_t, process_covariances_to_t)
    Q_mt = process_covariances_from_t + tf.matmul(
        state_transitions_from_t, A_tp_Q_mt, transpose_b=True
    )
    chol_Q_mt = tf.linalg.cholesky(Q_mt)  # L

    # V = L⁻¹ Aₜ₊Q₋ₜ
    L_inv_A_tp_Q_mt = tf.linalg.triangular_solve(
        chol_Q_mt, state_transitions_from_t @ process_covariances_to_t
    )

    # E = Q₋ₜAₜ₊ᵀQ₋₊⁻¹
    E = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(chol_Q_mt, L_inv_A_tp_Q_mt, adjoint=True)
    )
    D = state_transitions_to_t - E @ state_transitions_from_t @ state_transitions_to_t

    # Return the parameters for the conditional density:
    #   p(xₜ|x₋, x₊) = 𝓝(Pₙ @ [x₋, x₊], T [or T⁻¹])
    if return_precision:
        chol_Q_mt = tf.linalg.cholesky(process_covariances_to_t)
        chol_Q_tp = tf.linalg.cholesky(process_covariances_from_t)
        state_dim = state_transitions_to_t.shape[-1]
        identities = tf.broadcast_to(
            tf.eye(state_dim, dtype=default_float()), tf.shape(process_covariances_to_t)
        )
        Q_mt_inv = tf.linalg.cholesky_solve(chol_Q_mt, identities)
        L_tp_inv_A_tp = tf.linalg.triangular_solve(chol_Q_tp, state_transitions_from_t)
        # The conditional_precision T⁻¹ = Q₋ₜ⁻¹ + Aₜ₊ᵀQₜ₊⁻¹Aₜ₊
        T_inv = Q_mt_inv + tf.matmul(L_tp_inv_A_tp, L_tp_inv_A_tp, transpose_a=True)
        return D, E, T_inv
    else:
        # The conditional_covariance T = Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ == Q₋ₜ - Q₋ₜᵀAₜ₊ᵀL⁻ᵀL⁻¹Aₜ₊Q₋ₜ
        T = process_covariances_to_t - tf.matmul(L_inv_A_tp_Q_mt, L_inv_A_tp_Q_mt, transpose_a=True)
        return D, E, T


@tf_scope_fn_decorator
def _conditional_statistics(
    new_time_points: tf.Tensor, training_time_points: tf.Tensor, kernel: SDEKernel
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    ∀ xₜ ∈ new_time_points, computes the statistics Pₜ and Tₜ of the conditional densities:
        p(xₜ|x₋, x₊) = 𝓝(Pₜ @ [x₋, x₊], Tₜ)
    where
      - p is the density over state trajectories specified by the kernel
      - ∀ xₜ ∈ new_time_points
             x₊ = arg minₓ { |x-xₜ|, x ∈ training_time_point and x>xₜ }
             x₋ = arg minₓ { |x-xₜ|, x ∈ training_time_point and x⩽xₜ }

    Warning: `new_time_points` and `training_time_points` must be sorted

    :param new_time_points: sorted time points to generate observations for
            ``batch_shape + [num_new_time_points,]``
    :param training_time_points: sorted time points to condition on
            ``batch_shape + [num_training_time_points,]``
    :param kernel: a Markovian Kernel
    :return: parameters for the conditional mean and covariance, and the insertion indices
            ``batch_shape + [num_new_time_points, state_dim, 2*state_dim]``
            ``batch_shape + [num_new_time_points, state_dim, state_dim]``
            ``batch_shape + [num_new_time_points,]``
    """
    batch_shape = new_time_points.shape[:-1]
    batch_dims = len(batch_shape)
    # Indices of where the intermediate points would be inserted into the
    # existing time points. This will be slow if new_time_points are not sorted.
    # WARNING: tf.searchsorted will be slow if `new_time_points` are not sorted
    indices = tf.searchsorted(training_time_points, new_time_points)
    # HACK - arbitrary far away point
    inf = APPROX_INF * tf.ones_like(training_time_points[..., -1:])
    time_points_augmented = tf.concat([-inf, training_time_points, inf], axis=-1)

    # For all intermediate_time_points calculate the time deltas from the previous time_point
    # (delta_time_points_1) and to the next time_point (delta_time_points_2)
    inducing_plus = tf.gather(time_points_augmented, indices + 1, batch_dims=batch_dims)
    inducing_minus = tf.gather(time_points_augmented, indices, batch_dims=batch_dims)

    dX_mt = new_time_points - inducing_minus
    dX_tp = inducing_plus - new_time_points
    A_mt, Q_mt = kernel.transition_statistics(transition_times=inducing_minus, time_deltas=dX_mt)
    A_tp, Q_tp = kernel.transition_statistics(transition_times=new_time_points, time_deltas=dX_tp)
    F, G, T = _conditional_statistics_from_transitions(A_mt, Q_mt, A_tp, Q_tp)
    P = tf.concat([F, G], axis=-1)
    # Return the parameters for the conditional density:
    #   p(xₜ|x₋, x₊) = 𝓝(Pₙ @ [x₋, x₊], T)
    return P, T, indices


@tf_scope_fn_decorator
def cyclic_reduction_conditional_statistics(
    explained_time_points: tf.Tensor, conditioning_time_points: tf.Tensor, kernel: SDEKernel
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    r"""
    Compute :math:`Fₜ, Gₜ, Lₜ`. Such that:

    .. math:: p(xᵉₜ | xᶜₜ₋₁,  xᶜₜ₊₁) = 𝓝(xᵉₜ; Fₜ @ xᶜₜ₋₁ + Gₜ @ xᶜₜ₊₁, Tₜ = (Lₜ Lₜᵀ)⁻¹ = Lₜ⁻ᵀLₜ⁻¹)

    ...where superscripts :math:`e` and :math:`c` refer to explained and conditioning respectively.

    .. note:: :math:`xᵉ` and :math:`xᶜ` must be sorted, such that:

        .. math:: xᵉ₀ < xᶜ₀ < xᵉ₁ < ...  < xᵉₜ < xᶜₜ < xᶜₜ₊₁ < xᶜₜ₊₁ < ...

        ...where :math:`len(xᵉ) = len(xᶜ)` or :math:`len(xᵉ) = len(xᶜ) + 1`.

    This computes the conditional statistics :math:`Fₜ, Gₜ, Lₜ`, where
    :math:`𝔼 xᵉ|xᶜ = - L⁻ᵀ Uᵀ xᶜ`, with::

        Uᵀ = | F₁ᵀ                 [      |]  and  L⁻ᵀ = |L₁⁻ᵀ               |
             |  G₁ᵀ, F₂ᵀ           [      |]             |   L₂⁻ᵀ            |
             |     , G₂ᵀ,⋱         [      |]             |      L₃⁻ᵀ         |
             |           ⋱ ⋱       [      |]             |        ⋱          |
             |             ⋱ Fₙ₋₁ᵀ [      |]             |          ⋱        |
             |               Gₙ₋₁ᵀ [ Fₙᵀ  |]             |            Lₙ⁻ᵀ   |

    :math:`L` is the Cholesky factor of the conditional precision :math:`xᵉ|xᶜ`.

    Statistics :math:`F` and :math:`G` are computed from the conditional mean projection
    parameters :math:`D` and :math:`E` defined by :math:`𝔼 xᵉₙ|xᶜ = Dₙ @ xᶜₙ₋₁ + Eₙ @ xᶜₙ`.

    Solving the system :math:`- (L⁻ᵀ Uᵀ xᶜ)ₙ = Dₙ @ xᶜₙ₋₁ + Eₙ @ xᶜₙ`,
    we get :math:`Gₙ₋₁ᵀ = -Lₙᵀ Dₙ` and :math:`Fₙᵀ = -Lₙᵀ Eₙ`.

    Details of the system::

        -| L₁⁻ᵀF₁ᵀ xᶜ₁                     | = | E₁ xᶜ₁
         | L₂⁻ᵀG₁ᵀxᶜ₁  + L₂⁻ᵀ F₂ᵀ xᶜ₂      |   | D₂ xᶜ₁ +   E₂ xᶜ₂
         | L₃⁻ᵀ G₂ᵀ xᶜ₂ , L₃⁻ᵀ F₃ᵀ xᶜ₃,    |   | D₃ xᶜ₂ +   E₃ xᶜ₃
         | ⋮                               |   | ⋮
         | Lₙ⁻ᵀ Gₙ₋₁ᵀxᶜₙ₋₁,   Lₙ⁻ᵀ [Fₙᵀ]xᶜₙ|   | Dₙ xᶜₙ₋₁ +   [Eₙ] xᶜₙ

    Remarks on size:

    * When splitting :math:`x` of size :math:`n` into odd and even, you get
      :math:`nᵉ = (n+1)//2` and :math:`nᶜ = n//2`
    * At each level, cyclic reduction statistics have shape:

      .. math::
         &- F : nᶜ\\
         &- G : nᵉ - 1\\
         &- L : nᵉ\\

    Note that:

        * :math:`F₀` is not defined (there is no time point below :math:`xᵉ₀`)
        * The last element :math:`G` may not be defined if :math:`len(xᵉ) = len(xᶜ)`

    :param explained_time_points: Sorted time points to generate observations for, with shape
            ``batch_shape + [num_time_points_1,]``.
    :param conditioning_time_points: Sorted time points to condition on, with shape
            ``batch_shape + [num_time_points_2,]``.
    :param kernel:  A kernel.
    :return: Parameters for the conditional, with respective shapes
            ``batch_shape + [num_conditioning, state_dim, state_dim]``
            ``batch_shape + [num_explained - 1, state_dim, state_dim]``
            ``batch_shape + [num_explained, state_dim, state_dim]``.
    """
    delta_t1 = explained_time_points[..., 1:] - explained_time_points[..., :-1]
    delta_t2 = conditioning_time_points[..., 1:] - conditioning_time_points[..., :-1]
    tf.debugging.assert_non_negative(delta_t1, message="explained_time_points must be sorted")
    tf.debugging.assert_non_negative(delta_t2, message="conditioning_time_points must be sorted")

    num_explained = tf.shape(explained_time_points)[-1]
    num_conditioning = tf.shape(conditioning_time_points)[-1]

    tf.debugging.assert_greater_equal(
        num_explained,
        num_conditioning,
        message="explained_time_points must be longer than conditioning_time_points",
    )
    tf.debugging.assert_less_equal(
        num_explained - num_conditioning,
        1,
        message="explained_time_points must be as long as conditioning_time_points"
        " or 1 entry longer",
    )

    # Indices of where the intermediate points would be inserted into the
    # existing time points. This will be slow if new_time_points are not sorted.
    # HACK - arbitrary far away point
    inf = APPROX_INF * tf.ones_like(conditioning_time_points[..., :1])
    time_points_augmented = tf.concat([-inf, conditioning_time_points, inf], axis=-1)

    # For all intermediate_time_points calculate the time deltas from the previous time_point
    # (delta_time_points_1) and to the next time_point (delta_time_points_2)

    # this will span the range -inf to either the penultimate or last time points
    left_conditioning_time_points = time_points_augmented[..., :num_explained]
    dX_mt = explained_time_points - left_conditioning_time_points
    # this will span the range from the first time point to either the last or inf
    right_conditioning_time_points = time_points_augmented[..., 1 : num_explained + 1]
    dX_tp = right_conditioning_time_points - explained_time_points
    A_mt, Q_mt = kernel.transition_statistics(
        transition_times=left_conditioning_time_points, time_deltas=dX_mt
    )
    A_tp, Q_tp = kernel.transition_statistics(
        transition_times=explained_time_points, time_deltas=dX_tp
    )
    D, E, T_inv = _conditional_statistics_from_transitions(
        A_mt, Q_mt, A_tp, Q_tp, return_precision=True
    )
    L = tf.linalg.cholesky(T_inv)

    # Return the parameters for the cyclic reduction parameters:
    F = -tf.matmul(E, L, transpose_a=True)
    G = -tf.matmul(D, L, transpose_a=True)

    return F[..., 1:], G[..., :num_conditioning], L


@tf_scope_fn_decorator
def base_conditional_predict(
    conditional_projections: tf.Tensor,
    conditional_covariances: tf.Tensor,
    adjacent_states: tf.Tensor,
    pairwise_state_covariances: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Predict state at new time points given conditional statistics.

    Given conditionals statistics :math:`Pₜ, Tₜ` of :math:`p(xₜ|x₋, x₊) = 𝓝(Pₜ @ [x₋, x₊], Tₜ)`
    and pairwise marginals :math:`p(xₜ₋, xₜ₊) = 𝓝(mₜ, Sₜ)`,
    compute marginal mean and covariance of the marginal density:

    .. math:: p(xₜ) = 𝓝(Pₜ mₜ, Tₜ + Pₜ Sₜ Pₜᵀ)

    If :math:`Sₜ` is not provided, compute the conditional mean and covariance of the
    conditional density:

    .. math:: p(xₜ|[xₜ₋, xₜ₊] = mₜ) = 𝓝(Pₜ mₜ, Tₜ)

    :param conditional_projections: :math:`Pₜ` with shape
            ``batch_shape + [num_time_points, state_dim]``.
    :param conditional_covariances: :math:`Tₜ` with shape
            ``batch_shape + [num_time_points, state_dim, state_dim]``.
    :param adjacent_states: Pairs of states to condition on, with shape
            ``batch_shape + [num_time_points, 2 * state_dim]``.
    :param pairwise_state_covariances: Covariances of the pairs of states to condition on,
            with shape ``batch_shape + [num_time_points, 2 * state_dim, 2 * state_dim]``.
    :return: Predicted mean and covariance for the time points, with respective shapes
            ``batch_shape + [num_time_points, state_dim]``
            ``batch_shape + [num_time_points, state_dim, state_dim]``.
    """
    means = (conditional_projections @ tf.expand_dims(adjacent_states, axis=-1))[..., 0]
    covs = conditional_covariances
    if pairwise_state_covariances is not None:
        covs += tf.matmul(
            conditional_projections @ pairwise_state_covariances,
            conditional_projections,
            transpose_b=True,
        )
    return means, covs


@tf_scope_fn_decorator
def pairwise_marginals(
    dist: GaussMarkovDistribution, initial_mean: tf.Tensor, initial_covariance: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    TODO(sam): figure out what the initial mean and covariance should be for non-stationary kernels

    For each pair of subsequent states :math:`xₖ, xₖ₊₁`, return the mean and covariance of
    their joint distribution. This is assuming we start from, and revert to, the prior:

    .. math::
        &p(xₖ) = 𝓝(μₖ, Pₖ)  \verb|(we can get this from the marginals method)|\\
        &p(xₖ₊₁ | xₖ) = 𝓝(Aₖμₖ, Qₖ)

    Then:

    .. math::
        p(xₖ₊₁, xₖ) = 𝓝([μₖ, μₖ₊₁], [Pₖ, Pₖ Aₖᵀ])\\
                                     [Aₖ Pₖ, Pₖ₊₁]

    :param dist: The distribution.
    :param initial_mean:  The prior mean (used to extend the pairwise marginals
        of the distribution).
    :param initial_covariance: The prior covariance (used to extend the pairwise marginal of
        the state space model).
    :return: Mean and covariance pairs for the marginals, with respective shapes
            ``batch_shape + [num_transitions + 2, state_dim]``
            ``batch_shape + [num_transitions + 2, state_dim, state_dim]``.
    """
    means, covariances = dist.marginals
    covariances, subsequent_covariances = dist.covariance_blocks()

    initial_mean = tf.expand_dims(initial_mean, axis=-2)
    extended_means = tf.concat([initial_mean, means, initial_mean], axis=-2)

    joint_mean = tf.concat([extended_means[..., :-1, :], extended_means[..., 1:, :]], axis=-1)

    initial_covariance = tf.expand_dims(initial_covariance, axis=-3)
    extended_cov = tf.concat([initial_covariance, covariances, initial_covariance], axis=-3)
    extended_subsequent_cov = tf.concat(
        [
            tf.zeros_like(initial_covariance),
            subsequent_covariances,
            tf.zeros_like(initial_covariance),
        ],
        axis=-3,
    )

    joint_cov_0 = tf.concat(
        [extended_cov[..., :-1, :, :], tf.linalg.matrix_transpose(extended_subsequent_cov)],
        axis=-1,
    )

    joint_cov_1 = tf.concat([extended_subsequent_cov, extended_cov[..., 1:, :, :]], axis=-1)
    joint_cov = tf.concat([joint_cov_0, joint_cov_1], axis=-2)

    shape = tf.concat([dist.batch_shape, [dist.num_transitions + 2, 2 * dist.state_dim]], axis=0)
    shape_cov = tf.concat([shape, [2 * dist.state_dim]], axis=0)

    tf.debugging.assert_equal(tf.shape(joint_mean), shape)
    tf.debugging.assert_equal(tf.shape(joint_cov), shape_cov)

    return joint_mean, joint_cov
