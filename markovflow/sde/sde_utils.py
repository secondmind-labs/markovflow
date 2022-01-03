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

from markovflow.sde import SDE


def euler_maruyama(sde: SDE, x0: tf.Tensor, time_interval: tf.Tensor) -> tf.Tensor:
    """
    Numerical Simulation of SDEs of type: dx(t) = f(x,t)dt + L(x,t)dB(t) using the Euler-Maruyama Method.

    ..math:: x(t+1) = x(t) + f(x,t)dt + L(x,t)*sqrt(dt*q)*N(0,I)

    :param sde: Object of SDE class
    :param x0: state at start time, t0, with shape (batch_shape, state_dim)
    :param time_interval: Time grid for simulation, (N, )

    :return: Simulated SDE values, (batch_shape, N+1, D)

    Note: evaluation time interval is [t0, tn], x0 value is appended for t0 time. Thus, simulated values are (N+1).
    """

    DTYPE = x0.dtype
    num_data = time_interval.shape[0]
    state_dim = x0.shape[-1]
    n_batch = x0.shape[0]

    dt = tf.convert_to_tensor([[time_interval[1] - time_interval[0]]], dtype=DTYPE)
    f = sde.drift
    l = sde.diffusion

    def _step(current_val, nxt_val):
        x, t = current_val
        _, t_nxt = nxt_val
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
