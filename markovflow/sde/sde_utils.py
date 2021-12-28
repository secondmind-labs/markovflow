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

import math

import tensorflow as tf
import numpy as np

from markovflow.sde import SDE


def euler_maruyama(sde: SDE, x0: tf.Tensor, time_interval: tf.Tensor) -> tf.Tensor:
    """
    Numerical Simulation of SDEs of type: dx(t) = f(x,t)dt + L(x,t)dB(t) using the Euler-Maruyama Method.

    ..math:: x(t+1) = x(t) + f(x,t)dt + L(x,t)*sqrt(dt*q)*N(0,I)

    :param sde: Object of SDE class
    :param x0: value at start time, t0, (1, D)
    :param time_interval: Time grid for simulation, (N, )

    :return: Simulated SDE values, (N+1, D)

    Note: evaluation time interval is [t0, tn], x0 value is appended for t0 time. Thus, simulated values are (N+1).
    """

    DTYPE = x0.dtype
    N = time_interval.shape[0]
    D = x0.shape[-1]

    dt = float(time_interval[1] - time_interval[0])
    f = sde.drift
    l = sde.diffusion

    sde_values = np.zeros((N+1, D), dtype=np.float64)
    sde_values[0] = x0
    for t_idx in range(N):
        t = time_interval[t_idx]
        x_last = tf.cast(sde_values[t_idx], dtype=DTYPE)

        diffusion_term = l(x_last, t) * math.sqrt(dt)
        x_tmp = x_last + f(x_last, t) * dt + tf.random.normal(x_last.shape, dtype=DTYPE) * diffusion_term

        sde_values[t_idx+1] = x_tmp

    sde_values = tf.convert_to_tensor(sde_values, dtype=DTYPE)
    return sde_values
