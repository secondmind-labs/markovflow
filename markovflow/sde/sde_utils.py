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
from gpflow.quadrature import mvnquad

from markovflow.sde import SDE
from markovflow.state_space_model import StateSpaceModel


def euler_maruyama(sde: SDE, x0: tf.Tensor, time_interval: tf.Tensor) -> tf.Tensor:
    """
    Numerical Simulation of SDEs of type: dx(t) = f(x,t)dt + L(x,t)dB(t) using the Euler-Maruyama Method.

    ..math:: x(t+1) = x(t) + f(x,t)dt + L(x,t)*sqrt(dt*q)*N(0,I)

    :param sde: Object of SDE class
    :param x0: state at start time, t0, with shape (n_batch, state_dim)
    :param time_interval: Time grid for simulation, (num_transitions, )

    :return: Simulated SDE values, (n_batch, num_transitions+1, state_dim)

    Note: evaluation time interval is [t0, tn], x0 value is appended for t0 time.
    Thus, simulated values are (num_transitions+1).
    """

    DTYPE = x0.dtype
    num_data = time_interval.shape[0]
    state_dim = x0.shape[-1]
    n_batch = x0.shape[0]

    f = sde.drift
    l = sde.diffusion

    def _step(current_val, nxt_val):
        x, t = current_val
        _, t_nxt = nxt_val
        dt = t_nxt - t
        diffusion_term = l(x, t) * tf.math.sqrt(dt)
        x_nxt = x + f(x, t) * dt + tf.random.normal(x.shape, dtype=DTYPE) @ diffusion_term
        return x_nxt, t_nxt

    # [num_data, batch_shape, state_dim] for tf.scan
    sde_values = tf.zeros((num_data, n_batch, state_dim), dtype=DTYPE)

    # Adding time for batches for tf.scan
    t0 = tf.zeros((n_batch, 1), dtype=DTYPE)
    time_interval = tf.reshape(time_interval, (-1, 1, 1))
    time_interval = tf.repeat(time_interval, n_batch, axis=1)

    sde_values, _ = tf.scan(_step, (sde_values, time_interval), (x0, t0))

    # [batch_shape, num_data, state_dim]
    sde_values = tf.transpose(sde_values, [1, 0, 2])

    # Appending the initial value
    sde_values = tf.concat([tf.expand_dims(x0, axis=1), sde_values], axis=1)

    return sde_values


def E_sde_drift(sde: SDE, q_mean: TensorType, q_covar: TensorType) -> TensorType:
    """
    Calculates the Expectation of the SDE's drift under the provided Gaussian over states.

    ..math:: E_q(x(t))[f(x(t))]

    :param sde: SDE to be linearized.
    :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
    :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

    :return: the expectation value with shape (n_batch, num_states, state_dim).
    """
    fx = lambda x: sde.drift(x=x, t=tf.zeros(x.shape[0], 1))

    n_batch, n_states, state_dim = q_mean.shape
    q_mean = tf.reshape(q_mean, (-1, state_dim))
    q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))

    val = mvnquad(fx, q_mean, q_covar, H=10)

    val = tf.reshape(val, (n_batch, n_states, state_dim))
    return val


def E_sde_drift_gradient(sde: SDE, q_mean: TensorType, q_covar: TensorType) -> TensorType:
    """
     Calculates the Expectation of the gradient of the SDE's drift under the provided Gaussian over states

    ..math:: E_q(.)[f'(x(t))]

    :param sde: SDE to be linearized.
    :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
    :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).

    :return: the expectation value with shape (n_batch, num_states, state_dim).
    """
    n_batch, n_states, state_dim = q_mean.shape
    q_mean = tf.reshape(q_mean, (-1, state_dim))
    q_covar = tf.reshape(q_covar, (-1, state_dim, state_dim))
    val = mvnquad(sde.sde_drift_gradient, q_mean, q_covar, H=10)

    val = tf.reshape(val, (n_batch, n_states, state_dim))
    return val


def linearize_sde(sde: SDE, q_mean: TensorType, q_covar: TensorType, initial_mean: TensorType,
                  initial_chol_covariance: TensorType, process_chol_covariances: TensorType) -> StateSpaceModel:
    """
    Linearizes the SDE (with fixed diffusion) on the basis of the Gaussian over states

    ..math:: q(\cdot) \sim N(q_{mean}, q_{covar})

    ..math:: A_{i}^{*} = E_{q(.)}[d f(x)/ dx]
    ..math:: b_{i}^{*} = E_{q(.)}[f(x)] - A_{i}^{*}  E_{q(.)}[x]

    :param sde: SDE to be linearized.
    :param q_mean: mean of Gaussian over states with shape (n_batch, num_states, state_dim).
    :param q_covar: covariance of Gaussian over states with shape (n_batch, num_states, state_dim, state_dim).
    :param linearize_points: points used for the piecewise linearization ``[num_states,]``.
    :param initial_mean: The initial mean, with shape ``[n_batch, state_dim]``.
    :param initial_chol_covariance: Cholesky of the initial covariance, with shape ``[n_batch, state_dim, state_dim]``.
    :param process_chol_covariances: Cholesky of the noise covariance matrices, with shape
        ``[n_batch, num_states, state_dim, state_dim]``.

    :return: the state-space model of the linearized SDE.
    """

    E_f = E_sde_drift(sde, q_mean, q_covar)
    E_x = q_mean

    A = E_sde_drift_gradient(sde, q_mean, q_covar)
    b = E_f - A * E_x
    A = tf.linalg.diag(A)

    return StateSpaceModel(initial_mean=initial_mean, chol_initial_covariance=initial_chol_covariance,
                           state_transitions=A, state_offsets=b,
                           chol_process_covariances=process_chol_covariances)
