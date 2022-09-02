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
"""Utility functions for SDE"""

import tensorflow as tf

from gpflow.base import TensorType
from gpflow.quadrature import NDiagGHQuadrature

from markovflow.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.likelihoods import MultivariateGaussian


def euler_maruyama(sde: SDE, x0: tf.Tensor, time_grid: tf.Tensor) -> tf.Tensor:
    """
    Numerical Simulation of SDEs of type: dx(t) = f(x,t)dt + L(x,t)dB(t) using the Euler-Maruyama Method.

    ..math:: x(t+1) = x(t) + f(x,t)dt + L(x,t)*sqrt(dt*q)*N(0,I)

    :param sde: Object of SDE class
    :param x0: state at start time, t0, with shape (num_batch, state_dim)
    :param time_grid: A homogeneous time grid for simulation, (num_transitions, )

    :return: Simulated SDE values, (num_batch, num_transitions+1, state_dim)

    Note: evaluation time grid is [t0, tn], x0 value is appended for t0 time.
    Thus, simulated values are (num_transitions+1).
    """

    DTYPE = x0.dtype
    num_time_points = time_grid.shape[0]
    state_dim = x0.shape[-1]
    num_batch = x0.shape[0]

    f = sde.drift
    l = sde.diffusion

    def _step(current_state_time, next_state_time):
        x, t = current_state_time
        _, t_next = next_state_time
        dt = t_next[0] - t[0]  # As time grid is homogeneous
        diffusion_term = tf.cast(l(x, t) * tf.math.sqrt(dt), dtype=DTYPE)
        x_next = x + f(x, t) * dt + tf.random.normal(x.shape, dtype=DTYPE) @ diffusion_term
        return x_next, t_next

    # [num_data-1, batch_shape, state_dim] for tf.scan. -1 is for removing the initial state from the count.
    sde_values = tf.zeros((num_time_points-1, num_batch, state_dim), dtype=DTYPE)

    # Adding time for batches for tf.scan
    t0 = tf.zeros((num_batch, 1), dtype=DTYPE)
    time_grid = tf.reshape(time_grid[1:], (-1, 1, 1))  # [1:] is to remove t0 from the list
    time_grid = tf.repeat(time_grid, num_batch, axis=1)

    sde_values, _ = tf.scan(_step, (sde_values, time_grid), (x0, t0))

    # Append the initial state
    sde_values = tf.concat((tf.reshape(x0, (1, num_batch, state_dim)), sde_values), axis=0)

    # [batch_shape, num_data, state_dim]
    sde_values = tf.transpose(sde_values, [1, 0, 2])

    shape_constraints = [
        (sde_values, [num_batch, num_time_points, state_dim]),
        (x0, [num_batch, state_dim]),
    ]
    tf.debugging.assert_shapes(shape_constraints)

    return sde_values


def linearize_sde(
    sde: SDE,
    transition_times: TensorType,
    q_mean: TensorType,
    q_covar: TensorType,
    initial_mean: TensorType,
    initial_chol_covariance: TensorType,
) -> StateSpaceModel:
    """
    Linearizes the SDE (with fixed diffusion) on the basis of the Gaussian over states

    Note: this currently only works for sde with a state dimension of 1.

    ..math:: q(\cdot) \sim N(q_{mean}, q_{covar})

    ..math:: A_{i}^{*} = E_{q(.)}[d f(x)/ dx]
    ..math:: b_{i}^{*} = E_{q(.)}[f(x)] - A_{i}^{*}  E_{q(.)}[x]

    :param sde: SDE to be linearized.
    :param transition_times: Transition_times, (num_transitions, )
    :param q_mean: mean of Gaussian over states with shape (num_batch, num_states, state_dim).
    :param q_covar: covariance of Gaussian over states with shape (num_batch, num_states, state_dim, state_dim).
    :param initial_mean: The initial mean, with shape ``[num_batch, state_dim]``.
    :param initial_chol_covariance: Cholesky of the initial covariance, with shape ``[num_batch, state_dim, state_dim]``.
    :param process_chol_covariances: Cholesky of the noise covariance matrices, with shape
        ``[num_batch, num_states, state_dim, state_dim]``.

    :return: the state-space model of the linearized SDE.
    """

    assert sde.state_dim == 1
    E_f = sde.expected_drift(q_mean, q_covar)
    E_x = q_mean

    A = sde.expected_gradient_drift(q_mean, q_covar)
    b = E_f - A * E_x
    A = tf.linalg.diag(A)

    transition_deltas = tf.reshape(transition_times[1:] - transition_times[:-1], (1, -1, 1))
    state_transitions = A * tf.expand_dims(transition_deltas, -1) + tf.eye(sde.state_dim, dtype=A.dtype)

    state_offsets = b * transition_deltas
    chol_process_covariances = sde.diffusion(q_mean, transition_times[:-1]) * tf.expand_dims(
        tf.sqrt(transition_deltas), axis=-1
    )

    return StateSpaceModel(
        initial_mean=initial_mean,
        chol_initial_covariance=initial_chol_covariance,
        state_transitions=state_transitions,
        state_offsets=state_offsets,
        chol_process_covariances=chol_process_covariances,
    )


def KL_sde(sde_p: SDE, A_q, b_q, m, S, dt: float, quadrature_pnts: int = 20):
    """
    Calculate KL between two SDEs i.e. KL[q(x(.) || p(x(.)))]

    p(x(.)) : d x_t = f(x_t, t) dt   + dB_t  ; Q
    q(x(.)) : d x_t = f_L(x_t, t) dt + dB_t  ; Q  ; f_L(x_t, t) = - A_t * x_t + b_t.

    KL[q(x(.) || p(x(.)))] = 0.5 * \int <(f-f_L)^T Q^{-1} (f-f_L)>_{q_t} dt

    NOTE:
        1. Both the SDE have same diffusion i.e. Q.
        2. SDE q(x(.)) has a linear drift i.e. f_L(x_t, t) = - A_t * x_t + b_t
        3. Parameter A_t is "WITHOUT" the negative sign.
        4. A_q, b_q are the DRIFT parameters of the SDE and should not be confused with the state transitions of the SSM model.

    Apply Gaussian quadrature method to approximate the Expectation and integral is approximated as Riemann sum.

    """
    assert sde_p.state_dim == 1
    assert m.shape[0] == A_q.shape[0] == b_q.shape[0]

    def func(x, t=None, A_q=A_q, b_q=b_q):
        # Adding N information
        x = tf.transpose(x, perm=[1, 0, 2])
        n_pnts = x.shape[1]

        A_q = tf.repeat(A_q, n_pnts, axis=1)
        b_q = tf.repeat(b_q, n_pnts, axis=1)
        b_q = tf.expand_dims(b_q, axis=-1)

        A_q = tf.stop_gradient(A_q)
        b_q = tf.stop_gradient(b_q)

        prior_drift = sde_p.drift(x=x, t=t)
        # if stop_drift_gradient:
        #     prior_drift = tf.stop_gradient(prior_drift)

        tmp = prior_drift + ((x * A_q) - b_q)
        tmp = tmp * tmp

        sigma = sde_p.q
        sigma = tf.stop_gradient(sigma)

        val = tmp * (1 / sigma)

        return tf.transpose(val, perm=[1, 0, 2])

    diag_quad = NDiagGHQuadrature(sde_p.state_dim, quadrature_pnts)
    kl_sde = diag_quad(func, m, tf.squeeze(S, axis=-1))

    kl_sde = 0.5 * tf.reduce_sum(kl_sde) * dt
    return kl_sde


def gaussian_log_predictive_density(mean: tf.Tensor, chol_covariance: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """
        Compute the log probability density for a Gaussian.
    """
    x = tf.reshape(x, (-1, 1))
    mean = tf.reshape(mean, (-1, 1))

    mvn = MultivariateGaussian(chol_covariance=chol_covariance)
    log_pd = mvn.log_probability_density(mean, x)

    return log_pd
