"""
Test the expansion of KL
   KL[q_{L} || p] = KL[q_{L} || p_{L}] + u(q_{L}, p, p_{L})

using t-VGP and VGP models.
"""
import wandb
import tensorflow as tf
import numpy as np
from gpflow.config import default_float
from gpflow.likelihoods import Gaussian

from docs.sde.sde_exp_utils import generate_dw_data
from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.state_space_model import StateSpaceModel
from markovflow.models.cvi_sde import SDESSM
from markovflow.models.vi_sde import VariationalMarkovGP

DTYPE = default_float()
wandb.init()


def setup():
    t0 = 0
    t1 = 10
    dt = 0.001

    # Define q
    q = tf.ones((1, 1), dtype=DTYPE)

    # DW non-linear SDE, p
    sde_p = PriorDoubleWellSDE(q=q)

    # Generate observations.
    # Observations and likelihoods are not really required but need to pass them
    obs_val, obs_t, _, t, _, _, _ = generate_dw_data(q=q, x0=1., t0=t0, t1=t1, simulation_dt=dt,
                                                     noise_stddev=1., n_observations=20, dtype=DTYPE)
    obs_val = tf.reshape(obs_val, (-1, 1))
    observations = (obs_t, obs_val)
    likelihood = Gaussian()

    return sde_p, observations, t, likelihood, dt


if __name__ == '__main__':

    sde_p, observations, t, likelihood, dt = setup()
    t_vgp_model = SDESSM(prior_sde=sde_p, grid=t, input_data=observations, likelihood=likelihood, learning_rate=0.9)
    t_vgp_model.update_sites()
    print(f"t-VGP model : {t_vgp_model.classic_elbo()}")

    # Get drift parameters from q posterior SSM
    q_A = (tf.reshape(t_vgp_model.dist_q.state_transitions, (-1, 1, 1)) - tf.eye(1, dtype=t_vgp_model.dist_q.state_transitions.dtype)) / dt
    q_b = tf.reshape(t_vgp_model.dist_q.state_offsets, (-1, 1)) / dt

    # Compare Q of the posterior SSM, should be equal/ close to true q.
    dist_q_Q = tf.square(t_vgp_model.dist_q.cholesky_process_covariances) / dt
    print(dist_q_Q)

    vgp_model = VariationalMarkovGP(observations, sde_p, t, likelihood)

    # Initialize VGP model's posterior same as t-VGP
    vgp_model.p_initial_cov = tf.reshape(tf.square(t_vgp_model.initial_chol_cov), vgp_model.p_initial_cov.shape)
    vgp_model.p_initial_mean = t_vgp_model.initial_mean + tf.zeros_like(vgp_model.p_initial_mean)
    vgp_model.q_initial_mean = tf.reshape(t_vgp_model.dist_q.initial_mean, vgp_model.q_initial_mean.shape)
    vgp_model.q_initial_cov = tf.reshape(t_vgp_model.dist_q.initial_covariance, shape=vgp_model.q_initial_cov.shape)

    # -1 because of how VGP is parameterized
    vgp_model.A = -1 * tf.concat([q_A, tf.ones((1, 1, 1), dtype=q_A.dtype)], axis=0)
    vgp_model.b = tf.concat([q_b, tf.zeros((1, 1), dtype=q_b.dtype)], axis=0)

    # Test for means and covariances of the posterior of VGP and t-VGP
    np.testing.assert_array_almost_equal(t_vgp_model.dist_q.marginal_means.numpy().reshape(-1),
                                         vgp_model.forward_pass[0].numpy().reshape(-1), decimal=4)

    np.testing.assert_array_almost_equal(t_vgp_model.dist_q.marginal_covariances.numpy().reshape(-1),
                                         vgp_model.forward_pass[1].numpy().reshape(-1), decimal=4)

    print(f"VGP model : {vgp_model.elbo()}")
