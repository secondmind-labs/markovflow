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
import tensorflow_probability as tfp

from gpflow.base import TensorType
from gpflow.quadrature import NDiagGHQuadrature
from gpflow.probability_distributions import Gaussian

from markovflow.gauss_markov import BTDGaussian
from markovflow.ssm_gaussian_transformations import ssm_to_expectations, expectations_to_ssm_params
from markovflow.sde import SDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.sde.drift import LinearDrift
from markovflow.likelihoods import MultivariateGaussian
from markovflow.ssm_gaussian_transformations import ssm_to_naturals


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
        x_next = (
                x
                + f(x, t) * dt
                + tf.squeeze(
            diffusion_term @ tf.random.normal(x.shape, dtype=DTYPE)[..., None], axis=-1
        )
        )
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
        sde: SDE, transition_times: TensorType, linearization_path: Gaussian, initial_state: Gaussian
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
    initial_chol_covariance = handle_tensor_shape(
        tf.linalg.cholesky(initial_state.cov), desired_dimensions=3
    )  # (B, D, D)

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

    chol_q = sde.diffusion(q_mean, transition_times[:-1])

    linear_drift = LinearDrift(A=A, b=b)

    linear_drift_ssm = linear_drift.to_ssm(
        q=tf.square(chol_q),
        transition_times=transition_times,
        initial_mean=initial_mean,
        initial_chol_covariance=initial_chol_covariance
    )

    return linear_drift_ssm


def squared_drift_difference_along_Gaussian_path(
        sde_p: SDE, linear_drift: LinearDrift, q: Gaussian, dt: float, quadrature_pnts: int = 20
) -> tf.Tensor:
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


def gaussian_log_predictive_density(mean: tf.Tensor, chol_covariance: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """
        Compute the log probability density of observations x under a Gaussian distribution
        parameterized by its mean and covariance.
    """
    mvn = MultivariateGaussian(chol_covariance=chol_covariance)
    log_pd = mvn.log_probability_density(mean, x)

    return log_pd


def SSM_KL_along_Gaussian_path(
        func_q,
        func_p,
        ssm_q_process_covar: tf.Tensor,
        ssm_p_process_covar: tf.Tensor,
        ssm_q_marginals_mean: tf.Tensor,
        ssm_q_marginals_covar_diag: tf.Tensor,
        quadrature_pnts: int = 20,
) -> tf.Tensor:
    """
    The function calculates the KL divergence between two SSMs, KL[SSM-q || SSM-p].

    SSM-p: x_{t+1} = f^p(x_t, t) + e^p_t ; e^p_t ~ N(0, Q_t^p)
           f^p(x_t, t) =  A^p_t x_t + b^p_t

    SSM-q: x_{t+1} = f^q(x_t, t) + e^q_t ; e^q_t ~ N(0, Q_t^q)
           f^q(x_t, t) = A^q_t x_t + b^q_t

    KL[SSM-q || SSM-p] = KL[q(x0) || p(x0)] + 0.5 * ( log(|Q_t^p|/|Q_t^q|)
    + \sum_{t=0}^{N-1} E_{q(x_{t+1}, x_t)} [||x_{t+1} - f^p(x_t, t)||^{2}_{Q_t^{p}^{-1}}
    - ||x_{t+1} - f^q(x_t, t)||^{2}_{Q_t^{q}^{-1}}])

    Rather than doing the 2D quadrature on the join distribution of q(x_{t+1}, x_t) the term is further simplified to
    reduce to 1D quadrature.

    We use Gaussian quadrature method to approximate the expectation.

    :param func_q: f^q(x_t, t) = A^q_t x_t + b^q_t, with signature of the function as f(x_t).
    :param func_p: f^p(x_t, t) = A^p_t x_t + b^p_t,  with signature of the function as f(x_t).
    :param ssm_q_process_covar: Process covariance of SSM-q, Q_t^q with shape ``[num_transitions, 1, 1]``.
    :param ssm_p_process_covar: Process covariance of SSM-p, Q_t^p with shape ``[num_transitions, 1, 1]``.
    :param ssm_q_marginals_mean: The marginal means of SSM-q with shape ``[num_transitions+1, 1]``.
    :param ssm_q_marginals_covar_diag: The diagonal of the marginal covariance of SSM-q with shape ``[num_transitions+1, 1]``.
    :param quadrature_pnts: Number of quadrature points to use, integer.

    Note:
        1. This function currently doesn't support batching.
        2. func_q and func_p should support batching as quadrature points are passed to it.
        3. The function only supports state-dim=1 currently.

    """

    n_transitions = ssm_p_process_covar.shape[0]
    tf.debugging.assert_equal(
        ssm_q_process_covar.get_shape(), tf.TensorShape([n_transitions, 1, 1])
    )
    tf.debugging.assert_equal(
        ssm_p_process_covar.get_shape(), tf.TensorShape([n_transitions, 1, 1])
    )
    tf.debugging.assert_equal(
        ssm_q_marginals_mean.get_shape(), tf.TensorShape([n_transitions + 1, 1])
    )
    tf.debugging.assert_equal(
        ssm_q_marginals_covar_diag.get_shape(), tf.TensorShape([n_transitions + 1, 1])
    )

    ssm_p_process_covar_inv = tf.linalg.cholesky_solve(
        tf.linalg.cholesky(ssm_p_process_covar),
        tf.eye(
            ssm_p_process_covar.shape[-1],
            dtype=ssm_p_process_covar.dtype,
            batch_shape=[ssm_p_process_covar.shape[0]],
        ),
    )

    C_log_det = tf.linalg.logdet(ssm_q_process_covar) - tf.linalg.logdet(ssm_p_process_covar)
    C_term_trace = tf.linalg.trace(ssm_p_process_covar_inv * ssm_q_process_covar)
    C_term_D = ssm_q_marginals_mean.shape[-1]

    C = - C_log_det - C_term_D + C_term_trace

    def func(x):
        func_q_val = func_q(x=x)  # [n_pnts, num_transitions, 1]
        func_p_val = func_p(x=x)  # [n_pnts, num_transitions, 1]

        tf.debugging.assert_shapes(
            [
                (func_q_val, ("B", "N", "D")),
                (func_p_val, ("B", "N", "D")),
                (ssm_p_process_covar_inv[:, :, 0], ("N", "D")),
            ]
        )

        val = (func_p_val - func_q_val) * (func_p_val - func_q_val) * ssm_p_process_covar_inv[:, :, 0]

        return val

    m = ssm_q_marginals_mean[:-1]
    S = ssm_q_marginals_covar_diag[:-1]

    fn_difference = NDiagGHQuadrature(ssm_q_marginals_mean.shape[-1], quadrature_pnts)(func, mean=m, var=S)

    vals = fn_difference + C[..., None]

    tf.debugging.assert_shapes(
        [
            (m, ("N", "D")),
            (vals, ("N", "D")),
        ]
    )

    vals = 0.5 * tf.reduce_sum(vals)
    return vals


def ssm_to_btd_nat(ssm: StateSpaceModel) -> BTDGaussian:
    """
    Get natural parameters from a SSM model as a BTDGaussian.
    """
    assert ssm.batch_shape.is_compatible_with(
        tf.TensorShape([])), "ssm_to_btd_nat function currently does not support batch!"

    nat1, nat_diag, nat_subdiag = ssm_to_naturals(ssm)

    nat_btd = BTDGaussian(nat1, nat_diag, nat_subdiag)

    return nat_btd


def SSM_KL_with_grads_wrt_exp_params(
        ssm_q: StateSpaceModel, ssm_p: StateSpaceModel
) -> [tf.Tensor, [tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    The function calculates the KL divergence between two SSMs, KL[SSM-q || SSM-p], and returns the gradients wrt the
    expectation parameters of SSM-q.
    Note:
        1. The function only supports SSMs with state-dim=1.
        2. Currently, batching isn't supported.
    """

    tf.debugging.assert_equal(ssm_q.state_dim, 1)
    tf.debugging.assert_equal(ssm_p.state_dim, 1)
    assert ssm_q.batch_shape.is_compatible_with(
        tf.TensorShape([])), "SSM_KL_with_grads_wrt_exp_params function currently does not support batch"
    assert ssm_p.batch_shape.is_compatible_with(
        tf.TensorShape([])), "SSM_KL_with_grads_wrt_exp_params function currently does not support batch"

    exp1, exp_diag, exp_sub_diag = ssm_to_expectations(ssm_q)

    def dist_p_forward(x: tf.Tensor) -> tf.Tensor:
        """
        Evaluates f_p(x_t) = A(t)x(t) + b(t) where the shape of x(t) is `[num_batch, num_transitions, state_dim]`.
        """
        A_p = ssm_p.state_transitions
        b_p = ssm_p.state_offsets
        n_batch = x.shape[0]

        A_p_n = tf.squeeze(
            tf.repeat(A_p[None, ...], n_batch, axis=0), -1
        )  # [num_batch, n_transitions, 1]
        b_p_n = tf.repeat(b_p[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, 1]

        return A_p_n * x + b_p_n

    with tf.GradientTape() as g:
        g.watch([exp1, exp_diag, exp_sub_diag])

        (
            A,
            b,
            chol_initial_covariance,
            chol_process_covariances,
            initial_mean,
        ) = expectations_to_ssm_params(exp1, exp_diag, exp_sub_diag)

        def dist_q_forward(x: tf.Tensor) -> tf.Tensor:
            """
            Evaluates f_q(x_t) = A(t)x(t) + b(t) where the shape of x(t) is `[num_batch, num_transitions, state_dim]`.
            """
            n_batch = x.shape[0]

            A_n = tf.squeeze(
                tf.repeat(A[None, ...], n_batch, axis=0), -1
            )  # [num_batch, n_transitions, 1]
            b_n = tf.repeat(b[None, ...], n_batch, axis=0)  # [num_batch, n_transitions, 1]

            return A_n * x + b_n

        process_covariance_q = tf.square(chol_process_covariances)
        process_covariance_p = tf.square(ssm_p.cholesky_process_covariances)

        # create a diagonal covariance using expectation params
        diag_cov = exp_diag - (exp1 * exp1)[..., None]
        diag_cov = tf.squeeze(diag_cov, axis=-1)

        kl_q_p = SSM_KL_along_Gaussian_path(
            func_q=dist_q_forward,
            func_p=dist_p_forward,
            ssm_q_process_covar=process_covariance_q,
            ssm_p_process_covar=process_covariance_p,
            ssm_q_marginals_mean=exp1,
            ssm_q_marginals_covar_diag=diag_cov,
        )

        kl_q_0 = tfp.distributions.Normal(loc=initial_mean, scale=chol_initial_covariance)
        kl_p_0 = tfp.distributions.Normal(
            loc=ssm_p.initial_mean, scale=ssm_p.cholesky_initial_covariance
        )
        kl_0 = tfp.distributions.kl_divergence(kl_q_0, kl_p_0)

        kl_val = kl_0 + kl_q_p

    grads = g.gradient(kl_val, [exp1, exp_diag, exp_sub_diag])

    return kl_val, grads

