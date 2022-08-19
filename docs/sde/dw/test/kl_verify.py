"""
Test the expansion of KL
   KL[q_{L} || p] = KL[q_{L} || p_{L}] + u(q_{L}, p, p_{L})
"""

import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from gpflow.quadrature import NDiagGHQuadrature

from markovflow.sde.sde_utils import KL_sde, euler_maruyama
from markovflow.sde.sde import PriorOUSDE, PriorDoubleWellSDE

DTYPE = default_float()

tf.random.set_seed(33)
np.random.seed(33)


def cross_term(A_q, b_q, A_p_l, b_p_l, dt, q_mean, q_covar, p_sde):
    """
    Calculate
                E_{q(x)[(f_q - f_L)\Sigma^-1(f_L - f_p)] .
    """

    state_dim = 1

    def func(x, t=None, A_q=A_q, b_q=b_q, A_p_l=A_p_l, b_p_l=b_p_l, sde_p=p_sde):
        # Adding N information
        x = tf.transpose(x, perm=[1, 0, 2])
        n_pnts = x.shape[1]

        A_q = tf.repeat(A_q, n_pnts, axis=1)
        b_q = tf.repeat(b_q, n_pnts, axis=1)
        b_q = tf.expand_dims(b_q, axis=-1)
        A_q = tf.stop_gradient(A_q)
        b_q = tf.stop_gradient(b_q)

        A_p_l = tf.repeat(A_p_l, n_pnts, axis=1)
        b_p_l = tf.repeat(b_p_l, n_pnts, axis=1)
        b_p_l = tf.expand_dims(b_p_l, axis=-1)
        A_p_l = tf.stop_gradient(A_p_l)
        b_p_l = tf.stop_gradient(b_p_l)

        prior_drift = sde_p.drift(x=x, t=t)

        fq_fL = ((x * A_q) + b_q) - ((x * A_p_l) + b_p_l)  # (f_q - f_L)
        fl_fp = ((x * A_p_l) + b_p_l) - prior_drift  # (f_l - f_p)

        sigma = sde_p.q
        sigma = tf.stop_gradient(sigma)

        val = fq_fL * (1 / sigma) * fl_fp

        return tf.transpose(val, perm=[1, 0, 2])

    diag_quad = NDiagGHQuadrature(state_dim, 20)

    val = diag_quad(func, q_mean, tf.squeeze(q_covar, axis=-1))
    val = tf.reduce_sum(val) * dt

    return val


if __name__ == '__main__':
    t0 = 0
    t1 = 10
    dt = 0.001

    # define the x scale
    t = tf.convert_to_tensor(np.arange(t0, t1, dt), dtype=DTYPE)

    # Define q
    q = tf.ones((1, 1), dtype=DTYPE)

    # OU linear SDE, q_{L}
    sde_q = PriorOUSDE(initial_val=-0.5, q=q)

    # DW non-linear SDE, p
    sde_p = PriorDoubleWellSDE(q=q)

    # OU linear SDE, p_{L}
    sde_p_l = PriorOUSDE(initial_val=-2., q=q)

    # Get m, S of q for KL expectation
    m = euler_maruyama(sde_q, x0=tf.ones((1, 1), dtype=DTYPE), time_grid=t)
    S = 0.1 * tf.ones_like(m)

    m = tf.reshape(m, (-1, 1))
    S = tf.reshape(S, (-1, 1, 1))

    # statistics of q_{L} SDE
    A_q = tf.reshape(sde_q.decay * tf.ones_like(m), (-1, 1, 1))
    b_q = tf.zeros_like(m)

    # statistics of p_{L} SDE
    A_pl = tf.reshape(sde_p_l.decay * tf.ones_like(m), (-1, 1, 1))
    b_pl = tf.zeros_like(m)

    # KL[q_{L} || p] via Girsanov
    kl_girsanov_val = KL_sde(sde_p, -1 * A_q, b_q, m, S, dt=dt, quadrature_pnts=20)
    print(f"Girsanov value (KL[q_L || p]) = {kl_girsanov_val}")

    # KL[q_{L} || p_{L}]
    kl_ql_pl = KL_sde(sde_p_l, -1 * A_q, b_q, m, S, dt=dt, quadrature_pnts=20)
    print(f"KL[q_L || p_L] = {kl_ql_pl}")

    #  0.5 * E_{q(x) [||f_{L} - f_{p}||^2_{\Sigma^{-1}}]
    lin_loss = KL_sde(sde_p, -1 * A_pl, b_pl, m, S, dt=dt, quadrature_pnts=20)
    print(f"Lin loss = {lin_loss}")

    cross_term_val = cross_term(A_q, b_q, A_pl, b_pl, dt, m, S, sde_p)
    print(f"Cross term = {cross_term_val}")

    print(f"KL value is : {kl_ql_pl + lin_loss + cross_term_val}")
