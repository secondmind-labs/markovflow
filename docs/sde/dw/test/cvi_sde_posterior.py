"""
Compare posterior SSM of CVI-SDE model obtained via 2 methods:
    1. posterior_kalman.posterior_state_space_model()
    2. SSM(naturals_to_ssm_params())
"""

import wandb
import tensorflow as tf
import numpy as np
from gpflow.config import default_float
from gpflow.likelihoods import Gaussian

from docs.sde.sde_exp_utils import generate_dw_data
from markovflow.sde.sde import PriorDoubleWellSDE
from markovflow.models.cvi_sde import SDESSM

DTYPE = default_float()
wandb.init()


def setup():
    t0 = 0
    t1 = 10
    dt = 0.001
    q = tf.ones((1, 1), dtype=DTYPE)

    sde_p = PriorDoubleWellSDE(q=q)

    # Generate observations.
    obs_val, obs_t, _, t, _, _, _ = generate_dw_data(q=q, x0=1., t0=t0, t1=t1, simulation_dt=dt,
                                                     noise_stddev=1., n_observations=20, dtype=DTYPE)
    obs_val = tf.reshape(obs_val, (-1, 1))
    observations = (obs_t, obs_val)
    likelihood = Gaussian()

    return sde_p, observations, t, likelihood


if __name__ == '__main__':

    sde_p, observations, t, likelihood = setup()
    t_vgp_model = SDESSM(prior_sde=sde_p, grid=t, input_data=observations, likelihood=likelihood, learning_rate=0.9)
    t_vgp_model.update_sites()

    kf_ssm = t_vgp_model.posterior_kalman.posterior_state_space_model()

    np.testing.assert_allclose(t_vgp_model.dist_q.marginal_means, kf_ssm.marginal_means)
    np.testing.assert_allclose(t_vgp_model.dist_q.marginal_covariances, kf_ssm.marginal_covariances)

    np.testing.assert_allclose(t_vgp_model.dist_q.state_transitions, kf_ssm.state_transitions)
    np.testing.assert_allclose(t_vgp_model.dist_q.state_offsets, kf_ssm.state_offsets)
