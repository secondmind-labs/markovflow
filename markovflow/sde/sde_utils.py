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
from gpflow.probability_distributions import Gaussian

from markovflow.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.drift import LinearDrift


def euler_maruyama(sde: SDE, x0: tf.Tensor, time_grid: tf.Tensor) -> tf.Tensor:
    """
    Numerical Simulation of SDEs of type: dx(t) = f(x,t)dt + L(x,t)dB(t) using the Euler-Maruyama Method.

    ..math:: x(t+1) = x(t) + f(x,t)dt + L(x,t)*sqrt(dt*q)*N(0,I)

    :param sde: Object of SDE class
    :param x0: state at start time, t0, with shape ```[num_batch, state_dim]```
    :param time_grid: A homogeneous time grid for simulation, ```[num_transitions, ]```

    :return: Simulated SDE values, ```[num_batch, num_transitions+1, state_dim]```

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
        x_next = x + f(x, t) * dt + tf.squeeze(diffusion_term @ tf.random.normal(x.shape, dtype=DTYPE)[..., None],
                                               axis=-1)
        return x_next, t_next

    # [num_data, batch_shape, state_dim] for tf.scan
    sde_values = tf.zeros((num_time_points, num_batch, state_dim), dtype=DTYPE)

    # Adding time for batches for tf.scan
    t0 = tf.zeros((num_batch, 1), dtype=DTYPE)
    time_grid = tf.reshape(time_grid, (-1, 1, 1))
    time_grid = tf.repeat(time_grid, num_batch, axis=1)

    sde_values, _ = tf.scan(_step, (sde_values, time_grid), (x0, t0))

    # [batch_shape, num_data, state_dim]
    sde_values = tf.transpose(sde_values, [1, 0, 2])

    # Appending the initial value
    sde_values = tf.concat([tf.expand_dims(x0, axis=1), sde_values[..., :-1, :]], axis=1)

    shape_constraints = [
        (sde_values, [num_batch, num_time_points, state_dim]),
        (x0, [num_batch, state_dim]),
    ]
    tf.debugging.assert_shapes(shape_constraints)

    return sde_values


def handle_tensor_shape(tensor: tf.Tensor, desired_dimensions=2):
    """
    Handle shape of the tensor according to the desired dimensions.

    * if the shape is 1 more and at dimension 0 there is nothing then drop it.
    * if the shape is 1 less then add a dimension at the start.
    * else raise an Exception

    """
    tensor_shape = tensor._shape_as_list()
    if len(tensor_shape) == desired_dimensions:
        return tensor
    elif (len(tensor_shape) == (desired_dimensions + 1)) and tensor_shape[0] == 1:
        return tf.squeeze(tensor, axis=0)
    elif len(tensor_shape) == (desired_dimensions - 1):
        return tf.expand_dims(tensor, axis=0)
    else:
        raise Exception("Batch present!")


def linearize_sde(
        sde: SDE,
        transition_times: TensorType,
        linearization_path: Gaussian,
        initial_state: Gaussian,
) -> StateSpaceModel:
    """
    Linearizes the SDE (with fixed diffusion) on the basis of the Gaussian over states.

    Note: this currently only works for sde with a state dimension of 1.

    ..math:: q(\cdot) \sim N(q_{mean}, q_{covar})

    ..math:: A_{i}^{*} = (E_{q(.)}[d f(x)/ dx]) * dt + I
    ..math:: b_{i}^{*} = (E_{q(.)}[f(x)] - A_{i}^{*}  E_{q(.)}[x]) * dt

    :param sde: SDE to be linearized.
    :param transition_times: Transition_times, ``[num_transitions, ]``
    :param linearization_path: Gaussian of the states over transition times.
    :param initial_state: Gaussian over the initial state.

    :return: the state-space model of the linearized SDE.
    """

    assert sde.state_dim == 1

    q_mean = handle_tensor_shape(linearization_path.mu, desired_dimensions=3)  # (B, N, D)
    q_covar = handle_tensor_shape(linearization_path.cov, desired_dimensions=4)  # (B, N, D, D)
    initial_mean = handle_tensor_shape(initial_state.mu, desired_dimensions=2)  # (B, D, )
    initial_chol_covariance = handle_tensor_shape(tf.linalg.cholesky(initial_state.cov),
                                                  desired_dimensions=3)  # (B, D, D)

    B, N, D = q_mean.shape

    assert q_mean.shape == (B, N, D)
    assert q_covar.shape == (B, N, D, D)
    assert initial_mean.shape == (B, D,)
    assert initial_chol_covariance.shape == (B, D, D)

    E_f = sde.expected_drift(q_mean, q_covar)
    E_x = q_mean

    A = sde.expected_gradient_drift(q_mean, q_covar)
    b = E_f - A * E_x
    A = tf.linalg.diag(A)

    q = sde.diffusion(q_mean, transition_times[:-1])

    linear_drift = LinearDrift(A=A, b=b)
    linear_drift_ssm = linear_drift.to_ssm(q=q, transition_times=transition_times, initial_mean=initial_mean,
                                           initial_chol_covariance=initial_chol_covariance)
    return linear_drift_ssm


def squared_drift_difference_along_Gaussian_path(sde_p: SDE, linear_drift: LinearDrift, q: Gaussian,
                                                 dt: float, quadrature_pnts: int = 20) -> tf.Tensor:
    """
    Expected Square Drift difference between two SDEs
        * a first one denoted by p, that can be any arbitrary SDE.
        * a second which is linear, denoted by p_L, with a drift defined as f_L(x(t)) = A_L(t) x(t) + b_L(t)

    Where the expectation is over a third distribution over path summarized by its mean (m) and covariance (S)
    for all times given by a Gaussian `q`.

    Formally, the function calculates:
        0.5 * E_{q}[||f_L(x(t)) - f_p(x(t))||^{2}_{Σ^{-1}}].

    This function corresponds to the expected log density ratio:  E_q log [p_L || p].

    When the linear drift is of `q`, then the function returns the KL[q || p].

    NOTE:
        1. The function assumes that both the SDEs have same diffusion.

    Gaussian quadrature method is used to approximate the expectation and integral over time is approximated
    as Riemann sum.

    :param sde_p: SDE p.
    :param linear_drift: Linear drift representing the drift of the second SDE.
    :param q: Gaussian states of the path along which the drift difference is calculated.
    :param dt: Time-step value, float.
    :param quadrature_pnts: Number of quadrature points used.

    Note: Batching isn't supported.
    """
    assert sde_p.state_dim == 1

    m = handle_tensor_shape(q.mu, desired_dimensions=2)  # (N, D)
    S = handle_tensor_shape(q.cov, desired_dimensions=3)  # (N, D, D)

    A = linear_drift.A
    b = linear_drift.b

    assert m.shape[0] == S.shape[0] == A.shape[0] == b.shape[0]
    assert len(m.shape) == len(b.shape) == 2
    assert len(A.shape) == len(S.shape) == 3

    def func(x, t=None, A=A, b=b):
        # Adding N information
        x = tf.transpose(x, perm=[1, 0, 2])
        n_pnts = x.shape[1]

        A = tf.repeat(A, n_pnts, axis=1)
        b = tf.repeat(b, n_pnts, axis=1)
        b = tf.expand_dims(b, axis=-1)

        prior_drift = sde_p.drift(x=x, t=t)

        tmp = ((x * A) + b) - prior_drift
        tmp = tmp * tmp

        sigma = sde_p.q

        val = tmp * (1 / sigma)

        return tf.transpose(val, perm=[1, 0, 2])

    diag_quad = NDiagGHQuadrature(sde_p.state_dim, quadrature_pnts)
    drift_difference = diag_quad(func, m, tf.squeeze(S, axis=-1))

    drift_difference = 0.5 * tf.reduce_sum(drift_difference) * dt
    return drift_difference
