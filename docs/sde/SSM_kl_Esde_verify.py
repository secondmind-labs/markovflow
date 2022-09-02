"""
    Script to check KL calculated using SSM and E_sde term of VI SDE using Reimann Sum.
"""

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions

from markovflow.sde.sde import OrnsteinUhlenbeckSDE
from markovflow.kernels import OrnsteinUhlenbeck
from markovflow.sde.sde_utils import KL_sde


if __name__ == '__main__':

    t0 = 0.
    t1 = 1.
    dt = 0.01

    Q = .3

    time_grid = np.arange(t0, t1, dt).reshape((-1,))

    """
        "q" SSM and SDE.
    """
    decay_q = .5

    ou_sde_q = OrnsteinUhlenbeckSDE(decay=decay_q * np.ones((1, 1)), q=Q * np.ones((1, 1)))
    kernel_q = OrnsteinUhlenbeck(decay=decay_q, diffusion=Q)
    kernel_ssm_q = kernel_q.state_space_model(time_grid)

    """
        "p" SSM and SDE
    """
    decay_p = .8

    ou_sde_p = OrnsteinUhlenbeckSDE(decay=decay_p * np.ones((1, 1)), q=Q * np.ones((1, 1)))
    kernel_p = OrnsteinUhlenbeck(decay=decay_p, diffusion=Q)
    kernel_ssm_p = kernel_p.state_space_model(time_grid)

    """
    SSM KL
    """
    kl_q_p = kernel_ssm_q.kl_divergence(kernel_ssm_p)
    print(f" KL SSM : {kl_q_p.numpy().item()}")

    """
    E_sde
    """
    m, S = kernel_ssm_q.marginals
    m = m[:-1]  # remove the last state
    S = S[:-1]  # remove the last state

    A_q = decay_q * tf.ones_like(m)[..., None]
    b_q = tf.zeros_like(m)

    kl_sde = KL_sde(ou_sde_p, A_q, b_q, m, S, dt)

    """
    KL of the initial state
    """
    dist_qx0 = distributions.Normal(kernel_ssm_q.initial_mean, kernel_ssm_q.cholesky_initial_covariance)
    dist_px0 = distributions.Normal(kernel_ssm_p.initial_mean, kernel_ssm_p.cholesky_initial_covariance)
    kl_q0_p0 = distributions.kl_divergence(dist_qx0, dist_px0)

    print(f" E_sde : {(kl_sde + kl_q0_p0).numpy().item()}")
